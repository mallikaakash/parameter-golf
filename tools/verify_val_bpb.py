"""Verify that a submission's val_bpb denominator is honest.

The challenge is scored in bits per byte so that submitters can bring their own
tokenizer. That only works if the byte count in the denominator equals the number of
bytes the model actually had to predict. Two independent things can break it:

1. A LUT bug. train_gpt.py derives per-token byte counts from SentencePiece piece
   strings (build_sentencepiece_luts). If that accounting drifts from the real UTF-8
   length of the validation text, every score computed with it is wrong.

2. A lossy tokenizer. If the tokenizer normalizes text -- case folding, NFKC,
   whitespace stripping, unknown-token fallback -- then the token stream no longer
   represents the validation bytes. The model is graded on easier text while the
   denominator still claims the original byte count. Measured on a fixture corpus, a
   case-folding tokenizer alone moves bpb by ~0.12 with the LUT byte count staying
   within 0.002% of the true byte count, i.e. check 1 does not catch it.

This script runs both checks. Check 1 always runs. Check 2 requires the canonical
validation plaintext (`--reference`, or a `fineweb_val_reference.txt` sitting next to
the shards) -- without it the script still verifies that the token stream is
self-consistent (decode -> encode is a fixed point), which catches most lossy
tokenizers but cannot prove the tokens represent the official validation bytes.

Usage:

    python3 tools/verify_val_bpb.py \
        --data-path ./data/datasets/fineweb10B_sp1024 \
        --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model

Exit code is 0 if every check that could run passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

HEADER_INTS = 256
SHARD_MAGIC = 20240520
SHARD_VERSION = 1


def load_data_shard(file: Path) -> np.ndarray:
    """Same format contract as load_data_shard() in train_gpt.py."""
    header_bytes = HEADER_INTS * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    if file.stat().st_size != header_bytes + num_tokens * np.dtype("<u2").itemsize:
        raise ValueError(f"Shard size mismatch for {file}")
    return np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)


def build_luts(sp: spm.SentencePieceProcessor, vocab_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy mirror of build_sentencepiece_luts() in train_gpt.py."""
    table_size = max(int(sp.vocab_size()), vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int64)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(int(sp.vocab_size())):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary_token


def lut_byte_count(ids: np.ndarray, luts: tuple[np.ndarray, np.ndarray, np.ndarray]) -> int:
    """Bytes train_gpt.py would attribute to `ids`, counting every token as a target.

    eval_val() scores targets y = ids[1:] conditioned on ids[:-1], so the very first
    token of the stream is never counted there. It is added back here (base bytes only,
    since its "\u2581" is SentencePiece's synthetic add_dummy_prefix marker and decodes
    to nothing) so the total is directly comparable to the full decoded text.
    """
    base_bytes, has_leading_space, is_boundary_token = luts
    tgt, prev = ids[1:], ids[:-1]
    total = int(base_bytes[tgt].sum())
    total += int((has_leading_space[tgt] & ~is_boundary_token[prev]).sum())
    total += int(base_bytes[ids[0]])
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True, help="Directory holding fineweb_val_*.bin shards.")
    parser.add_argument("--tokenizer-path", required=True, help="SentencePiece .model used to produce the shards.")
    parser.add_argument("--reference", default=None, help="Canonical validation plaintext for the strict check.")
    parser.add_argument("--vocab-size", type=int, default=None, help="Expected VOCAB_SIZE; defaults to the tokenizer's.")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    val_files = [Path(p) for p in sorted(glob.glob(str(data_path / "fineweb_val_*.bin")))]
    if not val_files:
        print(f"FAIL: no fineweb_val_*.bin shards under {data_path}")
        return 1
    ids = np.concatenate([load_data_shard(f) for f in val_files]).astype(np.int64)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    vocab_size = args.vocab_size if args.vocab_size is not None else int(sp.vocab_size())

    print(f"val shards      : {len(val_files)} ({ids.size} tokens)")
    print(f"tokenizer       : {args.tokenizer_path} (vocab {sp.vocab_size()})")

    failures: list[str] = []
    if int(sp.vocab_size()) != vocab_size:
        failures.append(f"tokenizer vocab {sp.vocab_size()} != expected {vocab_size}")
    out_of_range = int((ids >= int(sp.vocab_size())).sum())
    if out_of_range:
        failures.append(f"{out_of_range} validation token ids fall outside the tokenizer vocabulary")

    luts = build_luts(sp, vocab_size)
    counted = lut_byte_count(ids, luts)
    decoded = sp.decode(ids.tolist())
    decoded_bytes = len(decoded.encode("utf-8"))

    # Check 1: the LUT accounting must match the bytes the token stream decodes to.
    delta = counted - decoded_bytes
    status = "PASS" if delta == 0 else "FAIL"
    print(f"\n[{status}] LUT byte accounting")
    print(f"       lut bytes {counted}  decoded bytes {decoded_bytes}  delta {delta:+d}")
    if delta != 0:
        failures.append(f"LUT byte count is off by {delta:+d} bytes vs the decoded token stream")

    # Check 2 (strict): the token stream must reproduce the canonical validation bytes.
    reference_path = Path(args.reference) if args.reference else data_path / "fineweb_val_reference.txt"
    if reference_path.is_file():
        reference = reference_path.read_text(encoding="utf-8")
        reference_bytes = len(reference.encode("utf-8"))
        lossless = decoded == reference
        status = "PASS" if lossless else "FAIL"
        print(f"\n[{status}] lossless round-trip against {reference_path}")
        print(f"       reference bytes {reference_bytes}  decoded bytes {decoded_bytes}  delta {decoded_bytes - reference_bytes:+d}")
        if not lossless:
            failures.append("tokenizer is lossy: decoding the validation tokens does not reproduce the reference text")
    else:
        # Weaker fallback: a lossless tokenizer is a fixed point of decode -> encode.
        reencoded = np.array(sp.encode(decoded, out_type=int), dtype=np.int64)
        stable = reencoded.size == ids.size and bool((reencoded == ids).all())
        status = "PASS" if stable else "FAIL"
        print(f"\n[{status}] decode/encode fixed point (no reference text available)")
        if not stable:
            failures.append("re-encoding the decoded validation text does not reproduce the token stream")
        print(f"       NOTE: pass `--reference <canonical val text>` to prove the tokens are the official bytes.")

    print(f"\nbpb denominator : {counted} bytes over {ids.size} tokens ({counted / ids.size:.4f} bytes/token)")
    print(f"                  a val_loss of L nats/token scores L / ln(2) * {ids.size / counted:.6f} bpb")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
