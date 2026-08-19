"""Build a tiny, self-contained fixture dataset in the challenge shard format.

The published FineWeb export (data/cached_challenge_fineweb.py) needs network access
to Hugging Face and is ~8B tokens. That makes it impossible to smoke-test the training
pipeline on a laptop, in CI, or in a sandbox with restricted egress. This script trains
a small SentencePiece BPE model over a local text corpus and writes
`fineweb_train_*.bin` / `fineweb_val_*.bin` shards that are byte-compatible with
`load_data_shard()` in train_gpt.py.

The result is NOT a substitute for the real benchmark data -- absolute bpb numbers from
a fixture run are meaningless. It exists so you can verify that the pipeline runs,
that the artifact serializes, and that val_bpb accounting is correct.

Example:

    python3 tools/make_fixture_dataset.py --out ./data/fixture
    DATA_PATH=./data/fixture/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/fixture/tokenizers/fineweb_1024_bpe.model \
    torchrun --standalone --nproc_per_node=1 train_gpt.py
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import sentencepiece as spm

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256

# The canonical challenge tokenizer settings. byte_fallback + identity normalization
# keep encode/decode lossless, which is what makes the bits-per-byte denominator
# comparable across submissions. See tools/verify_val_bpb.py.
CANONICAL_SPM_KWARGS = dict(
    model_type="bpe",
    character_coverage=0.9995,
    byte_fallback=True,
    normalization_rule_name="identity",
    add_dummy_prefix=True,
    remove_extra_whitespaces=False,
)


def write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0], header[1], header[2] = SHARD_MAGIC, SHARD_VERSION, tokens.size
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype("<u2", copy=False).tobytes())


def collect_corpus(patterns: list[str]) -> str:
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern, recursive=True)})
    if not paths:
        raise FileNotFoundError(f"No files matched {patterns}")
    chunks = [Path(p).read_text(encoding="utf-8", errors="ignore") for p in paths]
    return "\n".join(chunks)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(repo_root / "data" / "fixture"), help="Output root directory.")
    parser.add_argument(
        "--corpus",
        nargs="+",
        default=[str(repo_root / "records" / "**" / "*.md"), str(repo_root / "README.md")],
        help="Glob patterns for the plain-text corpus. Defaults to the repo's own markdown.",
    )
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--val-frac", type=float, default=0.1, help="Leading fraction of the corpus held out for validation.")
    args = parser.parse_args()

    out = Path(args.out)
    dataset_dir = out / "datasets" / f"fineweb10B_sp{args.vocab_size}"
    tokenizer_dir = out / "tokenizers"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    corpus = collect_corpus(args.corpus)
    split = int(len(corpus) * args.val_frac)
    val_text, train_text = corpus[:split], corpus[split:]
    if not val_text or not train_text:
        raise ValueError("Corpus is too small for the requested --val-frac")

    # The tokenizer is trained on the training split only, never on validation text.
    train_corpus_file = out / "corpus_train.txt"
    train_corpus_file.write_text(train_text, encoding="utf-8")
    prefix = tokenizer_dir / f"fineweb_{args.vocab_size}_bpe"
    if not prefix.with_suffix(".model").exists():
        spm.SentencePieceTrainer.train(
            input=str(train_corpus_file),
            model_prefix=str(prefix),
            vocab_size=args.vocab_size,
            minloglevel=2,
            **CANONICAL_SPM_KWARGS,
        )
    sp = spm.SentencePieceProcessor(model_file=str(prefix.with_suffix(".model")))

    train_ids = np.array(sp.encode(train_text, out_type=int), dtype=np.uint16)
    val_ids = np.array(sp.encode(val_text, out_type=int), dtype=np.uint16)
    write_shard(dataset_dir / "fineweb_train_000000.bin", train_ids)
    write_shard(dataset_dir / "fineweb_val_000000.bin", val_ids)

    # Keep the plaintext validation split next to the shards so verify_val_bpb.py can
    # check the byte denominator against ground truth.
    (dataset_dir / "fineweb_val_reference.txt").write_text(val_text, encoding="utf-8")

    val_bytes = len(val_text.encode("utf-8"))
    print(f"tokenizer: {prefix.with_suffix('.model')} (vocab {sp.vocab_size()})")
    print(f"train shard: {train_ids.size} tokens")
    print(f"val shard:   {val_ids.size} tokens over {val_bytes} utf-8 bytes ({val_bytes / val_ids.size:.3f} bytes/token)")
    print(f"lossless encode/decode roundtrip on validation split: {sp.decode(val_ids.tolist()) == val_text}")
    print(f"\nDATA_PATH={dataset_dir}")
    print(f"TOKENIZER_PATH={prefix.with_suffix('.model')}")


if __name__ == "__main__":
    main()
