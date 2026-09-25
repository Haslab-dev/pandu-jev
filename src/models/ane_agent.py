"""Pandu ANE Agent Runtime: Host tokenization + Core ML Apple Neural Engine (ANE) Inference.

Provides a unified, drop-in replacement for PanduJevNLP and LayaAgent, delivering
sub-millisecond typed decisions on Apple Silicon with 0 generated output tokens.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.jev_nlp import (
    TEMP_MAX,
    TEMP_MIN,
    build_sequence,
    clamp_temperature,
    normalized_entropy_confidence,
    temp_bucket,
)


class PanduANEAgent:
    """High-performance Core ML / Apple Neural Engine Runtime for Pandu Typed Decisions."""

    def __init__(
        self,
        package_dir: Union[str, Path] = "checkpoints/pandu_ane",
        compute_units: str = "cpu_ne",
        fallback_threshold: float = 0.85,
    ):
        self.package_dir = Path(package_dir)
        mlpackage_path = self.package_dir / "model.mlpackage"
        host_weights_path = self.package_dir / "host_weights.pt"
        metadata_path = self.package_dir / "metadata.json"

        if not mlpackage_path.exists():
            raise FileNotFoundError(f"Core ML package not found at {mlpackage_path}")

        # 1. Load Core ML Model onto Neural Engine
        units = {
            "cpu_ne": ct.ComputeUnit.CPU_AND_NE,
            "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU,
            "all": ct.ComputeUnit.ALL,
            "cpu": ct.ComputeUnit.CPU_ONLY,
        }
        if compute_units not in units:
            raise ValueError(f"compute_units must be one of {list(units.keys())}")

        self.compute_units = compute_units
        self.model = ct.models.MLModel(str(mlpackage_path), compute_units=units[compute_units])

        # 2. Load host weights and configurations
        host_data = torch.load(host_weights_path, map_location="cpu")
        self.tok_embeddings = host_data["tok_embeddings"].numpy()  # [Vocab, D]
        self.type_emb = host_data["type_emb"].numpy()  # [3, D]
        self.temperatures = host_data.get("temperatures", {})
        self.cfg = host_data.get("config", {})
        self.hidden_size = self.cfg.get("hidden_size", 256)
        self.length = self.cfg.get("length", 96)
        self.max_options = self.cfg.get("max_options", 32)
        self.fallback_threshold = fallback_threshold

        # Load action head in PyTorch CPU for fast post-processing
        from torch import nn
        self.act_head = nn.Sequential(
            nn.Linear(self.hidden_size + 4, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        self.act_head.load_state_dict(host_data["act_head"])
        self.act_head.eval()

        # 3. Load Offline Fast Rust Tokenizer
        tokenizer_dir = self.package_dir / "tokenizer"
        if not tokenizer_dir.exists():
            tokenizer_dir = Path("/Users/hy4-mac-002/hasdev/research/laya-coreml/models/snake/tokenizer")

        from tokenizers import Tokenizer as FastTokenizer
        backend = FastTokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        backend.no_padding()
        backend.no_truncation()
        cfg_tok = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())

        class StandaloneTokenizer:
            def __init__(self, b, cfg):
                self.backend = b
                for name in ("cls_token", "sep_token", "pad_token", "mask_token"):
                    val = cfg.get(name)
                    if isinstance(val, dict):
                        val = val.get("content")
                    tok_id = self.backend.token_to_id(val) if isinstance(val, str) else None
                    setattr(self, name, val)
                    setattr(self, name + "_id", tok_id)

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": self.backend.encode(text, add_special_tokens=add_special_tokens).ids}

        self.tokenizer = StandaloneTokenizer(backend, cfg_tok)

    def prepare_single_question_inputs(
        self,
        state: Union[str, Dict, List],
        question: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], int, int]:
        """Convert state + question into fixed-shape BC1L numpy tensors for Core ML."""
        qtype_str = question.get("type", question.get("t", "choice"))
        qtypes = {"choice": 0, "score": 1, "noul": 2}
        qtype_int = qtypes.get(qtype_str, 0)

        # Build token sequence and find option marker indices
        input_ids, marker_positions = build_sequence(
            self.tokenizer,
            state,
            question,
            max_len=self.length,
            head_max_len=min(48, self.length // 2),
        )

        actual_len = len(input_ids)
        num_options = min(len(marker_positions), self.max_options)

        # Pad input_ids to fixed self.length
        pad_id = self.tokenizer.pad_token_id or 0
        padded_ids = input_ids + [pad_id] * (self.length - actual_len)

        # 1. Embedding lookup: [1, D, 1, L]
        # self.tok_embeddings is [Vocab, D] -> slice: [L, D] -> transpose: [D, L]
        emb = self.tok_embeddings[padded_ids].T[None, :, None, :].astype(np.float16)

        # 2. Additive attention mask: [1, L, 1, L]
        # 0.0 for active tokens (< actual_len), -1e4 for padded tokens
        mask = np.full((1, self.length, 1, self.length), -1e4, dtype=np.float16)
        mask[:, :, :, :actual_len] = 0.0

        # 3. Type vector: [1, D, 1, 1]
        type_vec = self.type_emb[qtype_int][None, :, None, None].astype(np.float16)

        # 4. Marker map: [1, L, 1, MaxOptions]
        marker_map = np.zeros((1, self.length, 1, self.max_options), dtype=np.float16)
        for opt_idx in range(num_options):
            pos = marker_positions[opt_idx]
            if pos < self.length:
                marker_map[0, pos, 0, opt_idx] = 1.0

        coreml_inputs = {
            "embeddings": emb,
            "attention_mask": mask,
            "type_vectors": type_vec,
            "marker_map": marker_map,
        }

        return coreml_inputs, qtype_int, num_options

    def predict_single(
        self,
        state: Union[str, Dict, List],
        question: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute single-question prediction on Apple Neural Engine."""
        t0 = time.perf_counter()
        inputs, qtype_int, num_options = self.prepare_single_question_inputs(state, question)

        # Run inference on Apple Neural Engine
        t_infer0 = time.perf_counter()
        raw_outputs = self.model.predict(inputs)
        infer_ms = (time.perf_counter() - t_infer0) * 1000.0

        logits = raw_outputs["logits"][0, :num_options].astype(np.float32)
        cls_hidden = raw_outputs["cls_hidden"][0].astype(np.float32)  # [D]

        # Temperature calibration
        qtype_names = {0: "choice", 1: "score", 2: "noul"}
        bucket = temp_bucket(qtype_int, num_options)
        temp = self.temperatures.get(bucket, 1.0)
        calibrated_logits = logits / max(0.1, temp)

        # Probabilities
        exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits))
        probs = (exp_logits / np.sum(exp_logits)).tolist()

        # Telemetry
        k = max(2, num_options)
        conf = normalized_entropy_confidence(np.array(probs), k)
        top_idx = int(np.argmax(probs))

        # CPU Action Head
        p_arr = np.array(probs)
        sorted_p = np.sort(p_arr)[::-1]
        top1 = float(sorted_p[0])
        margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0
        entropy = -float(np.sum(p_arr * np.log(np.clip(p_arr, 1e-9, 1.0)))) / math.log(k)
        norm_k = float(k) / 32.0

        telemetry = torch.tensor([top1, margin, entropy, norm_k], dtype=torch.float32)
        act_input = torch.cat([torch.from_numpy(cls_hidden), telemetry], dim=0).unsqueeze(0)

        with torch.no_grad():
            act_logits = self.act_head(act_input)
            act_probs = F.softmax(act_logits, dim=-1)[0].tolist()

        should_fallback = bool(conf < self.fallback_threshold or act_probs[1] > 0.5)
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "selected_index": top_idx,
            "probabilities": [round(p, 4) for p in probs],
            "confidence": round(conf, 4),
            "fallback_recommended": should_fallback,
            "inference_ms": round(infer_ms, 2),
            "total_ms": round(total_ms, 2),
            "engine": f"Core ML ({self.compute_units.upper()})",
        }

    def predict(
        self,
        state: Union[str, Dict, List],
        questions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Predict answers across all questions in typed decision schema (Zero Generated Tokens)."""
        answers = {}
        total_infer_ms = 0.0
        t0 = time.perf_counter()

        for q_id, q_data in questions.items():
            ans = self.predict_single(state, q_data)
            total_infer_ms += ans["inference_ms"]
            answers[q_id] = ans

        total_wall_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "answers": answers,
            "usage": {
                "input_tokens": self.length,
                "output_tokens": 0,  # Zero-Token Generation Guarantee
            },
            "telemetry": {
                "inference_ms": round(total_infer_ms, 2),
                "total_wall_ms": round(total_wall_ms, 2),
                "device": "Apple Neural Engine (ANE)",
            },
        }
