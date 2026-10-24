# Data

Benchmark and frozen-code JSONL files are intentionally excluded from Git.

Expected local layout:

```text
data/
  raw/<dataset>.jsonl
  frozen/<dataset>_initial_codes.json
  splits/<dataset>_dev.jsonl
  splits/<dataset>_test.jsonl
```

Record dataset source, license, preprocessing script, and checksum in
`docs/data_card.md` before publishing results.
