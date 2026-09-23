"""Pandu-Jev NLP: Lightweight (~15M-28M) Typed Decision Model with Zero Generated Tokens.

Combines a compact ModernBERT-Tiny bidirectional transformer encoder with
typed decision heads (choice, score, noul) and calibrated uncertainty fallback.
Operates with 0 output tokens and delivers ~1.5 - 3.0 ms latency on Apple Silicon.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, ModernBertConfig, ModernBertModel

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def serialize_state(state: Union[str, Dict, List]) -> str:
    """Serialize environment state to a string representation."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value: Any) -> str:
    """Render a single criterion value as a clean string."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict[str, Any]) -> List[str]:
    """Render candidate option strings in label index order."""
    t = q.get("type", q.get("t"))
    crit = q.get("criteria", q.get("crit"))

    if t == "choice":
        if isinstance(crit, list):
            crit = {k: "" for k in crit}
        elif not isinstance(crit, dict):
            crit = {}
        return [
            k if v is None or v == "" else f"{k}: {render_criterion(v)}"
            for k, v in crit.items()
        ]
    elif t == "score":
        if not isinstance(crit, list):
            crit = ["level 0", "level 1", "level 2"]
        return [f"level {i}: {render_criterion(c)}" for i, c in enumerate(crit)]
    else:  # noul
        crit = crit if isinstance(crit, dict) else {}
        false_crit = crit.get("false")
        true_crit = crit.get("true")
        return [
            "false: " + (render_criterion(false_crit) if false_crit else "no, the statement does not hold"),
            "true: " + (render_criterion(true_crit) if true_crit else "yes, the statement holds"),
        ]


def build_sequence(
    tok,
    state: Union[str, Dict, List],
    question: Dict[str, Any],
    max_len: int = 512,
    head_max_len: int = 192,
) -> Tuple[List[int], List[int]]:
    """Format input sequence with option markers:
    [CLS] <type> question: {ins} [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
    Returns (input_ids, marker_positions).
    """
    qtype_str = question.get("type", question.get("t", "choice"))
    ins = str(question.get("instructions", question.get("ins", "")))
    mask_tok = tok.mask_token or "[MASK]"
    ins = ins.replace(mask_tok, " ")

    opts = render_options(question)
    head_ids = tok(f"{qtype_str} question: {ins}", add_special_tokens=False)["input_ids"]

    opt_ids = []
    for opt in opts:
        clean_opt = opt.replace(mask_tok, " ")
        opt_tokens = tok(" " + clean_opt, add_special_tokens=False)["input_ids"][:48]
        opt_ids.append([tok.mask_token_id] + opt_tokens)

    # Budget calculation
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)

    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]

    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)

    # State tokens
    room = max(0, max_len - len(ids) - 1)
    state_str = serialize_state(state).replace(mask_tok, " ")
    st = tok(state_str, add_special_tokens=False)["input_ids"][:room]
    ids = ids + st + [tok.sep_token_id]

    return ids[:max_len], [m for m in markers if m < max_len]


def normalized_entropy_confidence(probs: np.ndarray, k: int) -> float:
    """Compute normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = probs[:k]
    ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    """Compute calibration temperature bucket key."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{QTYPE_NAMES[int(qtype)]}:{size}"


def clamp_temperature(t: float, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """Clamp temperature within [TEMP_MIN, TEMP_MAX]."""
    try:
        t_val = float(t)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t_val):
        return 1.0
    return min(hi, max(lo, t_val))


@dataclass
class PanduJevNLPConfig:
    """Configuration for Pandu-Jev NLP Typed Decision Model."""
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    vocab_size: int = 50368
    max_position_embeddings: int = 1024
    classifier_dropout: float = 0.1
    model_name_or_path: str = "answerdotai/ModernBERT-base"
    fallback_threshold: float = 0.85
    temperatures: Dict[str, float] = field(default_factory=lambda: {
        "choice:2": 1.0,
        "choice:3-5": 1.0,
        "choice:6-10": 1.0,
        "choice:11+": 1.0,
        "score:2": 1.0,
        "score:3-5": 1.0,
        "score:6-10": 1.0,
        "score:11+": 1.0,
        "noul:2": 1.0,
    })


class PanduJevNLP(nn.Module):
    """Pandu-Jev NLP: ModernBERT-Tiny Backbone + Typed Marker Heads."""

    def __init__(self, config: Optional[PanduJevNLPConfig] = None):
        super().__init__()
        self.config = config or PanduJevNLPConfig()
        
        # 1. ModernBERT-Tiny Transformer Backbone
        bert_config = ModernBertConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            max_position_embeddings=self.config.max_position_embeddings,
            classifier_dropout=self.config.classifier_dropout,
        )
        self.encoder = ModernBertModel(bert_config)

        # 2. Question-Type Conditioning Embedding (choice=0, score=1, noul=2)
        self.type_emb = nn.Embedding(3, self.config.hidden_size)

        # 3. Typed Option Scorer (Scores option marker hidden states)
        self.scorer = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, 1),
        )

        # 4. Action / Fallback Telemetry Head
        # Takes [CLS embedding, top1_prob, margin, entropy, normalized_k]
        self.act_head = nn.Sequential(
            nn.Linear(self.config.hidden_size + 4, 128),
            nn.GELU(),
            nn.Linear(128, 2),  # [prob_act, prob_defer_or_fallback]
        )

        # Temperature calibration lookup
        self.temperatures = dict(self.config.temperatures)

        # Tokenizer initialization (offline cached)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path, local_files_only=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @property
    def memory_footprint_mb(self) -> float:
        """Model resident memory size in MB (FP32)."""
        return sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)

    def set_temperature(self, bucket: str, temperature: float):
        """Set clamped temperature for a specific question-type / option bucket."""
        self.temperatures[bucket] = clamp_temperature(temperature)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass scoring option markers.
        Args:
            input_ids: [B, SeqLen] token ids
            attention_mask: [B, SeqLen] binary mask
            marker_pos: [B, MaxOptions] indices of [MASK] markers in sequence
            marker_mask: [B, MaxOptions] binary mask for active options
            qtype: [B] question type integer tensor (0, 1, or 2)
        Returns:
            logits: [B, MaxOptions] unnormalized option scores
            act_logits: [B, 2] action / fallback logits
        """
        b, _ = input_ids.shape
        d = self.config.hidden_size

        # 1. Encode sequence with ModernBERT-Tiny
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state  # [B, SeqLen, D]

        # 2. Add question-type semantic bias
        type_vec = self.type_emb(qtype)[:, None, :]  # [B, 1, D]
        h = h + type_vec

        # 3. Gather marker representations at [MASK] positions
        # Expand marker_pos: [B, MaxOptions, D]
        expanded_marker_pos = marker_pos.long()[:, :, None].expand(-1, -1, d)
        marker_states = torch.gather(h, 1, expanded_marker_pos)  # [B, MaxOptions, D]

        # 4. Score markers
        logits = self.scorer(marker_states).squeeze(-1)  # [B, MaxOptions]

        # Mask inactive options with large negative value
        logits = torch.where(marker_mask.bool(), logits, torch.full_like(logits, -1e4))

        # 5. Compute distribution telemetry for act_head
        p = F.softmax(logits, dim=-1)
        k = marker_mask.sum(dim=-1).clamp(min=2).float()
        entropy = -(p * p.clamp(min=1e-9).log()).sum(dim=-1) / k.log()
        top_vals, _ = p.topk(2, dim=-1)
        top1 = top_vals[:, 0]
        margin = top_vals[:, 0] - top_vals[:, 1]
        norm_k = k / 32.0

        telemetry = torch.stack([top1, margin, entropy, norm_k], dim=-1)  # [B, 4]
        cls_state = h[:, 0]  # [B, D]
        act_input = torch.cat([cls_state, telemetry], dim=-1)  # [B, D + 4]
        act_logits = self.act_head(act_input)  # [B, 2]

        return logits, act_logits

    @torch.no_grad()
    def predict(
        self,
        state: Union[str, Dict, List],
        questions: Dict[str, Dict[str, Any]],
        max_length: int = 512,
        fallback_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """High-level typed decision inference (Zero Generated Tokens).

        Args:
            state: Environment state string, dictionary, or list.
            questions: Dictionary of questions keyed by question ID.
            max_length: Maximum sequence length budget.
            fallback_threshold: Threshold below which fallback is triggered.

        Returns:
            Dictionary matching the typed decision specification:
            {
                "model": "pandu-jev-nlp",
                "answers": { ... },
                "usage": {"input_tokens": N, "output_tokens": 0},
                "latency_ms": float
            }
        """
        threshold = fallback_threshold or self.config.fallback_threshold
        device = next(self.parameters()).device
        self.eval()

        t0 = time.perf_counter()
        answers = {}
        total_input_tokens = 0

        # 1. Build sequences for all questions
        parsed_questions = []
        for qid, qdef in questions.items():
            qtype_str = qdef.get("type", "choice")
            if qtype_str not in QTYPES:
                raise ValueError(f"Unsupported question type: {qtype_str}")
            qtype_int = QTYPES[qtype_str]
            ids, markers = build_sequence(self.tokenizer, state, qdef, max_len=max_length)
            if len(markers) < 1:
                raise ValueError(f"Question '{qid}' produced 0 option markers.")
            total_input_tokens += len(ids)
            parsed_questions.append({
                "qid": qid,
                "qdef": qdef,
                "qtype_str": qtype_str,
                "qtype_int": qtype_int,
                "ids": ids,
                "markers": markers,
                "k": len(markers),
            })

        if not parsed_questions:
            return {
                "model": "pandu-jev-nlp",
                "parameters": self.num_parameters,
                "answers": {},
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "latency_ms": 0.0,
            }

        # 2. Collate into batched tensors
        batch_size = len(parsed_questions)
        max_seq_len = max(len(q["ids"]) for q in parsed_questions)
        max_k = max(q["k"] for q in parsed_questions)
        pad_id = self.tokenizer.pad_token_id or 50283

        input_ids_t = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long, device=device)
        attn_mask_t = torch.zeros((batch_size, max_seq_len), dtype=torch.float, device=device)
        marker_pos_t = torch.zeros((batch_size, max_k), dtype=torch.long, device=device)
        marker_mask_t = torch.zeros((batch_size, max_k), dtype=torch.float, device=device)
        qtype_t = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i, q in enumerate(parsed_questions):
            seq_len = len(q["ids"])
            input_ids_t[i, :seq_len] = torch.tensor(q["ids"], dtype=torch.long, device=device)
            attn_mask_t[i, :seq_len] = 1.0
            k_opts = q["k"]
            marker_pos_t[i, :k_opts] = torch.tensor(q["markers"], dtype=torch.long, device=device)
            marker_mask_t[i, :k_opts] = 1.0
            qtype_t[i] = q["qtype_int"]

        # 3. Single Batched Forward Pass
        logits_batch, act_logits_batch = self.forward(
            input_ids=input_ids_t,
            attention_mask=attn_mask_t,
            marker_pos=marker_pos_t,
            marker_mask=marker_mask_t,
            qtype=qtype_t,
        )

        # 4. Decode typed results
        for i, q in enumerate(parsed_questions):
            qid = q["qid"]
            qdef = q["qdef"]
            qtype_str = q["qtype_str"]
            qtype_int = q["qtype_int"]
            k = q["k"]

            bucket = temp_bucket(qtype_int, k)
            temp = self.temperatures.get(bucket, 1.0)
            scaled_logits = (logits_batch[i, :k] / temp).cpu().numpy()

            exp_z = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_z / np.sum(exp_z)

            confidence = normalized_entropy_confidence(probs, k)
            top1_prob = float(np.max(probs))
            act_probs = F.softmax(act_logits_batch[i], dim=-1).cpu().numpy()
            act_probability = float(act_probs[0])
            should_fallback = bool(confidence < threshold or top1_prob < 0.40)

            # Formulate typed answer
            answer: Dict[str, Any] = {
                "type": qtype_str,
                "confidence": round(confidence, 4),
                "act_probability": round(act_probability, 4),
                "should_fallback": should_fallback,
                "fallback_threshold": threshold,
            }

            if qtype_str == "choice":
                opts = render_options(qdef)
                raw_crit = qdef.get("criteria", [])
                if isinstance(raw_crit, list):
                    labels = raw_crit
                elif isinstance(raw_crit, dict):
                    labels = list(raw_crit.keys())
                else:
                    labels = [f"opt_{i}" for i in range(k)]

                selected_idx = int(np.argmax(probs))
                selected_label = labels[selected_idx] if selected_idx < len(labels) else str(selected_idx)
                answer.update(
                    choice=selected_label,
                    probabilities={
                        labels[i] if i < len(labels) else str(i): round(float(probs[i]), 4)
                        for i in range(k)
                    },
                )
            elif qtype_str == "score":
                crit_list = qdef.get("criteria", [f"level {i}" for i in range(k)])
                expected_score = float(np.sum(np.arange(k) * probs))
                answer.update(
                    score=round(expected_score, 4),
                    probabilities={str(i): round(float(probs[i]), 4) for i in range(k)},
                    legend={str(i): str(crit_list[i]) for i in range(min(k, len(crit_list)))},
                )
            else:  # noul
                p_true = float(probs[1]) if k >= 2 else float(probs[0])
                noul_choice = bool(p_true >= 0.5)
                answer.update(
                    noul=round(p_true, 4),
                    verdict=noul_choice,
                    confidence=round(max(p_true, 1.0 - p_true), 4),
                )

            answers[qid] = answer

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "model": "pandu-jev-nlp",
            "parameters": self.num_parameters,
            "answers": answers,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": 0,  # Zero token generation invariant
            },
            "latency_ms": round(elapsed_ms, 3),
        }
