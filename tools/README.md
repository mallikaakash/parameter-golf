# Tools

Small helpers for testing the challenge pipeline. Neither is part of a submission
artifact and neither affects scoring.

## `make_fixture_dataset.py`

Builds a tiny dataset in the challenge shard format from local text, with no network
access and no Hugging Face download. Use it to smoke-test that `train_gpt.py` runs
end to end -- training loop, validation, int8+zlib artifact, round-trip eval -- before
spending money on an 8xH100 box.

```bash
python3 tools/make_fixture_dataset.py --out ./data/fixture
```

Absolute bpb numbers from a fixture run are meaningless: the corpus is a few hundred KB
of local markdown, not FineWeb. Only use it to check that the pipeline works.

## `verify_val_bpb.py`

Checks that a submission's bits-per-byte denominator is honest. The challenge lets you
bring your own tokenizer, which only works if the byte count in the denominator equals
the bytes the model actually had to predict.

```bash
python3 tools/verify_val_bpb.py \
    --data-path ./data/datasets/fineweb10B_sp1024 \
    --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model \
    --reference ./path/to/canonical_val_text.txt
```

Two independent failure modes are checked:

1. **LUT accounting.** `build_sentencepiece_luts()` in `train_gpt.py` derives per-token
   byte counts from SentencePiece piece strings. The tool re-derives them and compares
   the total against the real UTF-8 length of the decoded token stream.

2. **Lossy tokenization.** Case folding, NFKC, whitespace stripping, or unknown-token
   fallback all make the validation text easier to predict while leaving the byte count
   nearly unchanged, so check 1 cannot see them. Passing `--reference` with the
   canonical validation plaintext catches this; without it the tool falls back to a
   weaker decode/encode fixed-point test.

Measured on a fixture corpus, a case-folding tokenizer scored **2.026 bpb vs 2.146 bpb**
for the canonical tokenizer with identical data, model, and step count -- a 0.12 bpb
swing, roughly 70% of the total leaderboard progress to date -- while its LUT byte count
stayed within 0.002% of the true byte count. Check 1 passed it; only check 2 caught it.
