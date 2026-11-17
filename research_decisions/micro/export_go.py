"""Export the Pandu-Micro (MicroBERT) encoder into a compact, Go-consumable binary.

Produces TWO artifacts next to this script (paths overridable):

1. ``pandu_encoder.rkb``  — flat binary with the distilled encoder weights and
   the tokenizer vocabulary, consumed by RakitKode's pure-Go embedder
   (``github.com/Haslab-dev/RakitKode/brain`` → ``brain/panduEncoder``).
2. ``pandu_encoder_golden.json`` — exact float32 embedding vectors for 5 fixed
   sample texts, computed here by the PyTorch model. RakitKode's parity test
   (``brain/embedder_pandu_test.go``) re-computes them in pure Go and asserts
   max abs diff < 1e-3 per dimension.

Exported embedding definition
-----------------------------
The exported vector is the model's **final hidden state, masked-mean-pooled
over tokens and L2-normalized** — the standard sentence-embedding recipe. We
deliberately do NOT use the model's decision-head pooling (``pooler(mean+h[0])``
tanh stack): that head is trained for the 6 typed decision tasks, while the
raw mean-pooled hidden state is the general-purpose sentence representation
the distillation corpus shapes. The Go embedder mirrors the trunk exactly:

    x  = token_emb[ids] + pos_emb[positions]     # no dropout at eval
    x  = LayerNorm(x)                            # emb_norm (eps=1e-5)
    h  = 3 × pre-LN encoder layers               # d=192, 4 heads, FFN 512,
                                                 # gelu (erf), norm_first=True
    v  = mean(h over valid tokens)               # masked mean-pool
    v /= ||v||₂                                  # L2 normalize

All math in float32 (checkpoint is FP16; upcast on load). Runtime embeds are
truncated at ``--max-tokens`` (default 128) — fact texts are ≤280 chars, so
truncation rarely binds; the golden vectors use the same limit.

.rkb binary format (little-endian throughout)
---------------------------------------------
    offset  size  field
    0       4     magic "RKBE"
    4       4     uint32 format version (currently 1)
    8       4     int32 dims            (hidden size, 192)
    12      4     int32 num_layers      (3)
    16      4     int32 num_heads       (4)
    20      4     int32 max_positions   (512)
    24      4     int32 max_tokens      (runtime embed truncation, 128)
    28      4     int32 vocab_json_len
    32      N     vocab JSON — array of tokens, id == array index
    ..      4     int32 tensor_count
    ..      ..    per tensor:
                      int32 name_len, name_len bytes UTF-8 name,
                      int32 rank, rank × int32 shape,
                      prod(shape) × float32 LE

Tensor names (row-major, PyTorch Linear convention [out_features, in_features]):
    token_emb [V,d], pos_emb [P,d], emb_norm_w [d], emb_norm_b [d],
    l{i}_attn_inproj_w [3d,d], l{i}_attn_inproj_b [3d]   (rows packed Q;K;V)
    l{i}_attn_outproj_w [d,d], l{i}_attn_outproj_b [d]
    l{i}_ff_w1 [ff,d], l{i}_ff_b1 [ff], l{i}_ff_w2 [d,ff], l{i}_ff_b2 [d]
    l{i}_norm1_w [d], l{i}_norm1_b [d], l{i}_norm2_w [d], l{i}_norm2_b [d]

Checkpoint note
---------------
This exporter loads the micro-tier PanduMicro distillation checkpoint
(``micro/checkpoints/pandu_micro_long_running_fp16.pt``), which is the model
paired with ``micro/model.py`` (the architecture the Go embedder mirrors
exactly) and ``micro/tokenizer.py`` (the 766-token vocabulary embedded in the
.rkb). The tiny-tier sibling ``tiny/checkpoints/microbert_distilled.pt`` has
an identical encoder trunk but a different vocabulary/heads and is NOT what
this script exports.

Usage (research_decisions/ as cwd — the same convention as
``python3 -m micro.distill_jev``, since ``micro/tokenizer.py`` imports the
sibling ``datasets`` package)::

    cd research_decisions
    python3 -m micro.export_go \
        --checkpoint micro/checkpoints/pandu_micro_long_running_fp16.pt \
        --out micro/pandu_encoder.rkb \
        --golden micro/pandu_encoder_golden.json

(the script also inserts research_decisions/ into sys.path, so a direct
``python3 research_decisions/micro/export_go.py`` from the repo root works
too; ``PYTHONPATH=src`` is NOT enough — src/datasets lacks rakitkode_domain).
"""

import argparse
import json
import os
import struct
import sys

# Make `micro.*` / `datasets.*` importable regardless of how we are invoked
# (python -m micro.export_go from research_decisions/, or a direct script
# path from the repo root).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)  # research_decisions/
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import torch  # noqa: E402

from micro.model import PanduMicro, PanduMicroConfig  # noqa: E402
from micro.tokenizer import PanduMicroTokenizer  # noqa: E402

MAGIC = b"RKBE"
FORMAT_VERSION = 1
DIMS = 192
NUM_LAYERS = 3
NUM_HEADS = 4
INTERMEDIATE = 512
MAX_POSITIONS = 512

# Golden sample texts: brain-fact-like sentences covering word tokens, OOV
# char-fallback, punctuation, a [goal] marker and a dunder marker — every
# branch of the tokenizer the Go port must replicate.
GOLDEN_SAMPLES = [
    "Reading the index while rebuilding it deadlocks the worker pool",
    "Keep retry caps at five attempts for upstream requests",
    "Always run go test -race before tagging a release",
    "[goal] prefer subagents for broad search to protect the context window",
    "Pitfall: __has_stacktrace__ found in the failure log; pivot to the stack dump",
]

DEFAULT_CHECKPOINT = os.path.join(_HERE, "checkpoints", "pandu_micro_long_running_fp16.pt")


def load_encoder(checkpoint: str, vocab_size: int) -> PanduMicro:
    """Instantiate the micro/model.py architecture and load the distilled weights."""
    config = PanduMicroConfig(
        vocab_size=vocab_size,
        hidden_size=DIMS,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE,
        max_position_embeddings=MAX_POSITIONS,
    )
    model = PanduMicro(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    rows = state["token_embeddings.weight"].shape[0]
    if rows != vocab_size:
        raise SystemExit(
            f"checkpoint vocab mismatch: token_embeddings has {rows} rows but the "
            f"micro tokenizer builds {vocab_size} tokens — the checkpoint and the "
            "current datasets/tokenizer have drifted"
        )
    model.load_state_dict(state, strict=True)
    model.eval()  # dropout off; deterministic forward
    return model.float()  # FP16 checkpoint → FP32 math, matching the Go embedder


def pooled_embedding(model: PanduMicro, tokenizer: PanduMicroTokenizer, text: str, max_tokens: int):
    """Final hidden states → masked mean-pool → L2 normalize, as float32 list.

    Mirrors model.PanduMicro.forward's trunk (same embedding lookup, same
    LayerNorm, same 3 pre-LN encoder layers with a full attention mask — no
    padding in single-sequence inference) but pools the raw hidden states
    instead of routing through the decision-head pooler.
    """
    ids = tokenizer.encode(text, max_length=max_tokens)
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.float)

    with torch.no_grad():
        b, seq_len = input_ids.shape
        positions = torch.arange(seq_len).unsqueeze(0).expand(b, seq_len)
        x = model.token_embeddings(input_ids) + model.position_embeddings(positions)
        x = model.emb_norm(x)  # emb_drop is identity at eval
        h = model.encoder(x, src_key_padding_mask=None)  # unpadded single sequence

        mask = attention_mask.unsqueeze(-1)
        summed = torch.sum(h * mask, dim=1)
        count = torch.clamp(mask.sum(dim=1), min=1e-9)
        vec = summed / count
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)

    return ids, vec[0].to(torch.float32).tolist()


def write_rkb(out_path: str, vocab, tensors, max_tokens: int):
    """Serialize the .rkb container (format documented in the module docstring)."""
    vocab_json = json.dumps(vocab, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<iiiii", DIMS, NUM_LAYERS, NUM_HEADS, MAX_POSITIONS, max_tokens))
        f.write(struct.pack("<i", len(vocab_json)))
        f.write(vocab_json)
        f.write(struct.pack("<i", len(tensors)))
        for name, shape, data in tensors:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<i", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<i", len(shape)))
            for dim in shape:
                f.write(struct.pack("<i", dim))
            f.write(struct.pack(f"<{len(data)}f", *data))


def collect_tensors(state) -> list:
    """Flatten the encoder state dict into (name, shape, flat-float32-list)."""
    d, ff, heads3 = DIMS, INTERMEDIATE, 3 * DIMS
    spec = [
        ("token_emb", (state["token_embeddings.weight"].shape[0], d), "token_embeddings.weight"),
        ("pos_emb", (MAX_POSITIONS, d), "position_embeddings.weight"),
        ("emb_norm_w", (d,), "emb_norm.weight"),
        ("emb_norm_b", (d,), "emb_norm.bias"),
    ]
    for i in range(NUM_LAYERS):
        p = f"encoder.layers.{i}."
        spec += [
            (f"l{i}_attn_inproj_w", (heads3, d), p + "self_attn.in_proj_weight"),
            (f"l{i}_attn_inproj_b", (heads3,), p + "self_attn.in_proj_bias"),
            (f"l{i}_attn_outproj_w", (d, d), p + "self_attn.out_proj.weight"),
            (f"l{i}_attn_outproj_b", (d,), p + "self_attn.out_proj.bias"),
            (f"l{i}_ff_w1", (ff, d), p + "linear1.weight"),
            (f"l{i}_ff_b1", (ff,), p + "linear1.bias"),
            (f"l{i}_ff_w2", (d, ff), p + "linear2.weight"),
            (f"l{i}_ff_b2", (d,), p + "linear2.bias"),
            (f"l{i}_norm1_w", (d,), p + "norm1.weight"),
            (f"l{i}_norm1_b", (d,), p + "norm1.bias"),
            (f"l{i}_norm2_w", (d,), p + "norm2.weight"),
            (f"l{i}_norm2_b", (d,), p + "norm2.bias"),
        ]
    tensors = []
    for name, shape, key in spec:
        t = state[key].to(torch.float32).contiguous()
        if tuple(t.shape) != tuple(shape):
            raise SystemExit(f"unexpected shape for {key}: {tuple(t.shape)} (want {shape})")
        tensors.append((name, list(shape), t.flatten().tolist()))
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Pandu-Micro encoder for the Go brain embedder")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="distilled .pt checkpoint path")
    parser.add_argument("--out", default=os.path.join(_HERE, "pandu_encoder.rkb"), help=".rkb output path")
    parser.add_argument("--golden", default=os.path.join(_HERE, "pandu_encoder_golden.json"),
                        help="golden vectors JSON output path")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="runtime embed truncation used by the Go embedder and golden vectors")
    args = parser.parse_args()
    max_tokens_arg = args.max_tokens

    tokenizer = PanduMicroTokenizer()
    vocab = list(tokenizer.id_to_token)
    print(f"[export_go] vocab: {len(vocab)} tokens from micro/tokenizer.py")

    model = load_encoder(args.checkpoint, vocab_size=len(vocab))
    state = model.state_dict()
    print(f"[export_go] checkpoint: {args.checkpoint} (encoder trunk {NUM_LAYERS}×{DIMS}, {NUM_HEADS} heads)")

    tensors = collect_tensors(state)
    write_rkb(args.out, vocab, tensors, args.max_tokens)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[export_go] wrote {args.out} ({size_mb:.2f} MB, {len(tensors)} tensors)")

    samples = []
    for text in GOLDEN_SAMPLES:
        ids, vec = pooled_embedding(model, tokenizer, text, args.max_tokens)
        samples.append({"text": text, "tokens": len(ids), "embedding": vec})
        print(f"[export_go] golden: {len(ids):3d} tokens | {text[:60]}")

    golden = {
        "format": "pandu-encoder-golden/1",
        "checkpoint": os.path.basename(args.checkpoint),
        "encoder": "PanduMicro (research_decisions/micro/model.py), 3 pre-LN layers, d=192, 4 heads, FFN 512",
        "pooling": "masked mean over final hidden states, L2-normalized",
        "vocab_size": len(vocab),
        "dims": DIMS,
        "max_tokens": args.max_tokens,
        "rkb": os.path.basename(args.out),
        "tolerance": 1e-3,
        "samples": samples,
    }
    with open(args.golden, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False)
    print(f"[export_go] wrote {args.golden} ({len(samples)} golden vectors)")


if __name__ == "__main__":
    main()
