---
license: mit
language:
  - tr
pretty_name: Turkish Speech Bench results
task_categories:
  - automatic-speech-recognition
tags:
  - benchmark
  - leaderboard
  - turkish
  - asr
  - cpu
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: leaderboard.csv
---

# Turkish Speech Bench results

Machine-readable results of [turkish-speech-bench](https://github.com/cyber-emreclskn/turkish-speech-bench):
accuracy (WER/CER), speed (RTFx, latency) and memory of Turkish speech recognition models,
measured on CPUs.

- `leaderboard.csv` — one row per model, dataset and hardware; this is what the dataset viewer shows.
- `results/` — the raw result files, including the hardware fingerprint, library versions and
  per-utterance predictions of every run.

Browse the same numbers on the
[leaderboard Space](https://huggingface.co/spaces/emreclskn/turkish-speech-bench).

```python
from datasets import load_dataset

rows = load_dataset("emreclskn/turkish-speech-bench-results", split="train")
```

Results are only comparable within the same hardware. The measurement protocol is described in the
[methodology](https://github.com/cyber-emreclskn/turkish-speech-bench/blob/main/docs/methodology.md).

This dataset is deployed automatically from the GitHub repository; open issues and pull requests there.

## Licenses

The benchmark code and these result files are MIT. Reference transcripts inside the raw result
files come from the evaluated datasets and keep their own license, which is recorded in each file
(for example FLEURS, CC-BY-4.0).
