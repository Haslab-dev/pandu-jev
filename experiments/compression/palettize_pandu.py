"""Screen conv-only weight palettization for Pandu ANE Transformer.

Compresses PanduANE Core ML package weights from FP16 to W8 (8-bit palette)
using coremltools.optimize, reducing resident model size to ~18 MB.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto


def palettize_pandu_model(
    input_package: Path,
    output_package: Path,
    nbits: int = 8,
    mode: str = "uniform",
    group_size: int = 32,
) -> Path:
    input_package = Path(input_package)
    output_package = Path(output_package)

    if output_package.exists():
        shutil.rmtree(output_package)

    model = ct.models.MLModel(str(input_package), skip_model_load=True)

    config = cto.coreml.OptimizationConfig(
        op_type_configs={
            "conv": cto.coreml.OpPalettizerConfig(
                mode=mode,
                nbits=nbits,
                granularity="per_grouped_channel",
                group_size=group_size,
                weight_threshold=2048,
            )
        }
    )

    t0 = time.perf_counter()
    compressed_model = cto.coreml.palettize_weights(model, config)
    duration = time.perf_counter() - t0

    output_package.parent.mkdir(parents=True, exist_ok=True)
    compressed_model.save(str(output_package))

    # Also copy host weights, metadata, and tokenizer
    source_dir = input_package.parent
    target_dir = output_package.parent

    for item in ("host_weights.pt", "metadata.json", "tokenizer"):
        src = source_dir / item
        dst = target_dir / item
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy(src, dst)

    print(f"Compressed {input_package} -> {output_package} (W{nbits}, {duration:.2f}s)")
    return output_package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Palettize Pandu Core ML package to W8")
    parser.add_argument("--package", type=str, default="checkpoints/pandu_ane/model.mlpackage")
    parser.add_argument("--output", type=str, default="checkpoints/pandu_ane_w8/model.mlpackage")
    parser.add_argument("--bits", type=int, default=8)
    args = parser.parse_args()

    palettize_pandu_model(Path(args.package), Path(args.output), nbits=args.bits)
