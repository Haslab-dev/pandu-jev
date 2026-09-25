"""Core ML and Apple Neural Engine (ANE) Export Pipeline for Pandu-Jev.

Converts PanduANEBody into a native Core ML MLProgram package (.mlpackage)
targeting Apple Neural Engine (ANE) with FP16 precision.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import coremltools as ct
import numpy as np
import torch

from models.ane_transformer import PanduANEBody
from models.jev_nlp import PanduJevNLP, PanduJevNLPConfig


def export_pandu_to_coreml(
    model: PanduJevNLP,
    output_path: Path,
    length: int = 96,
    max_options: int = 32,
    minimum_deployment_target: Any = ct.target.macOS15,
) -> Tuple[ct.models.MLModel, Dict[str, Any]]:
    """Export PanduJevNLP to an ANE-optimized Core ML MLProgram package.

    Args:
        model: Trained or initialized PanduJevNLP instance.
        output_path: Directory path where model.mlpackage will be saved.
        length: Fixed sequence length budget (default 96, divisible by 32).
        max_options: Fixed maximum number of candidate options (default 32).
        minimum_deployment_target: Core ML deployment target (macOS 15 / iOS 18).

    Returns:
        tuple of (compiled_mlmodel, conversion_metadata)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hidden_size = model.hidden_size

    # 1. Instantiate PanduANEBody
    ane_body = PanduANEBody(model, length=length, max_options=max_options)
    ane_body.eval()

    # 2. Prepare sample tracing tensors
    dummy_emb = torch.randn(1, hidden_size, 1, length, dtype=torch.float32)
    dummy_mask = torch.zeros(1, length, 1, length, dtype=torch.float32)
    dummy_type = torch.zeros(1, hidden_size, 1, 1, dtype=torch.float32)
    dummy_markers = torch.zeros(1, length, 1, max_options, dtype=torch.float32)

    # 3. PyTorch TorchScript Tracing
    with torch.no_grad():
        traced_model = torch.jit.trace(
            ane_body,
            (dummy_emb, dummy_mask, dummy_type, dummy_markers),
            strict=True,
            check_trace=True,
        )

    # 4. Core ML Conversion
    inputs = [
        ct.TensorType(name="embeddings", shape=(1, hidden_size, 1, length), dtype=np.float16),
        ct.TensorType(name="attention_mask", shape=(1, length, 1, length), dtype=np.float16),
        ct.TensorType(name="type_vectors", shape=(1, hidden_size, 1, 1), dtype=np.float16),
        ct.TensorType(name="marker_map", shape=(1, length, 1, max_options), dtype=np.float16),
    ]

    outputs = [
        ct.TensorType(name="logits", dtype=np.float16),
        ct.TensorType(name="cls_hidden", dtype=np.float16),
    ]

    t0 = time.perf_counter()
    converted_model = ct.convert(
        traced_model,
        source="pytorch",
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=minimum_deployment_target,
        skip_model_load=False,
    )
    conversion_time = time.perf_counter() - t0

    # Save package
    converted_model.save(str(output_path))

    # Save host weights needed by ANEAgent (token embeddings, type embeddings, action head)
    host_weights_path = output_path.parent / "host_weights.pt"
    host_data = {
        "tok_embeddings": model.encoder.embeddings.tok_embeddings.weight.detach().cpu(),
        "type_emb": model.type_emb.weight.detach().cpu(),
        "act_head": model.act_head.state_dict(),
        "temperatures": model.temperatures,
        "config": {
            "hidden_size": hidden_size,
            "length": length,
            "max_options": max_options,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
        },
    }
    torch.save(host_data, host_weights_path)

    # Copy tokenizer files if available
    source_tok = Path("checkpoints/pandu_ane/tokenizer")
    dest_tok = output_path.parent / "tokenizer"
    if source_tok.exists() and not dest_tok.exists():
        import shutil
        shutil.copytree(source_tok, dest_tok)

    metadata = {
        "format": "pandu-ane-mlpackage",
        "length": length,
        "max_options": max_options,
        "hidden_size": hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "num_parameters": model.num_parameters,
        "conversion_seconds": round(conversion_time, 2),
        "package_path": str(output_path),
        "host_weights_path": str(host_weights_path),
    }

    metadata_path = output_path.parent / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return converted_model, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Pandu-Jev NLP to Core ML / ANE")
    parser.add_argument("--output", type=str, default="checkpoints/pandu_ane/model.mlpackage")
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--max-options", type=int, default=32)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--base", action="store_true", help="Export full 149M ModernBERT-Base")
    args = parser.parse_args()

    if args.base:
        print("Configuring full ModernBERT-Base (149M, 22 layers, d=768, 12 heads)...")
        cfg = PanduJevNLPConfig(
            use_pretrained_base=True,
            hidden_size=768,
            intermediate_size=1152,
            num_hidden_layers=22,
            num_attention_heads=12,
        )
        model = PanduJevNLP(cfg)
    else:
        cfg = PanduJevNLPConfig(num_hidden_layers=args.layers, hidden_size=256, num_attention_heads=4)
        model = PanduJevNLP(cfg)
        ckpt = Path("checkpoints/pandu_snake_nlp.pt")
        if ckpt.exists():
            print(f"Loading checkpoint from {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)

    print(f"Exporting Pandu-Jev ({model.num_parameters:,} params) to Core ML...")
    mlmodel, meta = export_pandu_to_coreml(
        model, Path(args.output), length=args.length, max_options=args.max_options
    )
    print(f"Export complete in {meta['conversion_seconds']}s -> {meta['package_path']}")
