# Methodology

The goal is numbers you can trust when choosing a Turkish speech model for a
CPU-only device: a kiosk, a laptop, an edge box. Every rule below exists to make
results **reproducible** and **comparable**.

## What is measured

| Metric | Definition |
|---|---|
| **WER** | Corpus-level word error rate: total edits ÷ total reference words (not the mean of per-utterance WERs). |
| **CER** | Corpus-level character error rate, spaces included. Useful for an agglutinative language where one wrong suffix makes a whole word wrong. |
| **RTF / RTFx** | RTF = transcription time ÷ audio duration. RTFx = 1/RTF = seconds of audio processed per second. |
| **p50 / p90 latency** | Per-utterance wall-clock time of the `transcribe` call. |
| **Load time** | Time to load weights into memory. Downloads are done beforehand and never timed. |
| **Peak RAM** | Highest resident memory of the whole process, sampled every 20 ms. |
| **Model RAM** | Resident memory added by loading the model. |
| **Disk** | Size of the model files. |

## Turkish text normalization

Reference and hypothesis go through the same normalizer
([`tsbench/normalize.py`](../src/tsbench/normalize.py)) before scoring:

1. Unicode NFC.
2. Circumflex vowels folded: `kâğıt` → `kağıt`.
3. **Turkish casing**: `I` → `ı` and `İ` → `i`. Python's `str.lower()` gets both
   wrong (`"I".lower()` is `i`; `"İ".lower()` is `i` + U+0307), which silently
   inflates WER for Turkish.
4. Apostrophes removed: `İstanbul'da` → `istanbulda`.
5. Other punctuation and symbols become spaces; whitespace collapsed.

Digits are kept as digits. Try it: `tsb normalize "İSTANBUL'da Kâğıt!"`.

## Timing protocol

* Audio is decoded and resampled to 16 kHz mono **before** timing, so file I/O
  is excluded.
* Each model runs in its **own fresh process** (`--isolate`, the default). Native
  allocators (CTranslate2, PyTorch) rarely return memory to the OS, so running
  models back-to-back in one process would inflate the next model's peak RAM.
* `prepare()` (downloads) → `load()` (timed) → one untimed warm-up transcription
  → every utterance transcribed and timed once.
* Thread count is pinned (`--threads`, default 4) through `OMP_NUM_THREADS` and
  friends plus each backend's own setting.
* Decoding settings are part of the result (`model.settings`): Whisper runs INT8
  with beam size 5, language forced to Turkish, no VAD, no conditioning on previous
  text; CTC models use greedy decoding.

## Hardware

Each result records a fingerprint: CPU model, physical and logical cores, RAM, OS,
Python and backend library versions. Results are grouped by `hardware_id` (a hash
of CPU, core counts and RAM). **Speed is only comparable within the same hardware.**

Two kinds of hardware are reported:

* **GitHub Actions runners** — anyone can reproduce them with the Benchmark workflow.
* **Contributed hardware** (e.g. kiosk CPUs) — run `tsb run` locally with `--label`
  and open a pull request with the JSON files.

## Datasets

| Name | Content | License |
|---|---|---|
| `fleurs-tr` | Google FLEURS Turkish, read Wikipedia sentences; `test` split by default. The raw transcription is the reference. | CC-BY-4.0 |
| `*.jsonl` | Your own data: one `{"audio": "...", "text": "..."}` per line. | yours |

`--limit N` always takes the first N utterances sorted by id, so a subset is the
same on every machine. The subset is stored in the result (`first-N` or `full`)
and shown on the leaderboard.

## Known limitations

* FLEURS is read speech; spontaneous or noisy speech will score worse.
* A 200-utterance subset gives a usable but noisy WER estimate; full-split runs are
  preferred for close comparisons.
* GitHub-hosted runners are shared machines, so their speed numbers vary by a few
  percent between runs.
