"""One-shot generator for the committed tiny HuggingFace classification model.

Produces ``fixtures/test_model/`` — a deterministic, tiny
``BertForSequenceClassification`` (vocab 1000, hidden 32, 2 layers,
5 labels) plus a matching BERT wordpiece tokenizer. The host's
``huggingface_classification`` backend loads it from the pulled artifact
directory with pure-Python inference — no GPU, no llama.cpp binary.

Run with a venv that has torch + transformers + safetensors
(solar-host's ``.[huggingface]`` extra):

    env -u PYTHONPATH ../../solar-host/.venv/bin/python \\
        fixtures/generate_test_model.py

Output files (committed to the repo):
    config.json            — BertConfig (5 labels, id2label, pad_token_id=0)
    model.safetensors      — random fixed-seed weights (~300-500 KB)
    tokenizer.json         — fast wordpiece tokenizer
    tokenizer_config.json  — class hint for AutoTokenizer
    special_tokens_map.json
    vocab.txt
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

_OUT_DIR = Path(__file__).resolve().parent / "test_model"

VOCAB_SIZE = 1000
HIDDEN_SIZE = 32
NUM_LAYERS = 2
NUM_HEADS = 4
INTERMEDIATE_SIZE = 64
NUM_LABELS = 5
SEED = 42

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def build_vocab() -> list[str]:
    vocab = list(SPECIAL_TOKENS)
    vocab.extend(f"tok{i}" for i in range(len(vocab), VOCAB_SIZE))
    assert len(vocab) == VOCAB_SIZE
    return vocab


def main() -> int:
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
    )

    out = _OUT_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # ── tokenizer ────────────────────────────────────────────────
    vocab = build_vocab()
    vocab_path = out / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n")

    tokenizer = BertTokenizerFast(
        vocab_file=str(vocab_path),
        do_lower_case=True,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=128,
    )
    tokenizer.save_pretrained(str(out))
    print(f"tokenizer -> {out}")

    # ── model ────────────────────────────────────────────────────
    config = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_position_embeddings=128,
        num_labels=NUM_LABELS,
        id2label={i: f"LABEL_{i}" for i in range(NUM_LABELS)},
        label2id={f"LABEL_{i}": i for i in range(NUM_LABELS)},
        pad_token_id=0,
        torchscript=False,
    )
    torch.manual_seed(SEED)
    model = BertForSequenceClassification(config)
    model.save_pretrained(str(out), safe_serialization=True)
    print(f"model -> {out}")

    # ── report ───────────────────────────────────────────────────
    for f in sorted(out.iterdir()):
        print(f"  {f.name:28} {f.stat().st_size:>9} bytes")
    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"total: {total} bytes")
    assert total < 2_000_000, "fixture too large to commit"
    return 0


if __name__ == "__main__":
    sys.exit(main())
