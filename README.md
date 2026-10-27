# 🇹🇷 Turkish Speech Bench

[![CI](https://github.com/cyber-emreclskn/turkish-speech-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/cyber-emreclskn/turkish-speech-bench/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/turkish-speech-bench)](https://pypi.org/project/turkish-speech-bench/)
[![Leaderboard](https://img.shields.io/badge/🤗-Leaderboard-yellow)](https://huggingface.co/spaces/emreclskn/turkish-speech-bench)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**How accurate — and how fast — are Turkish speech recognition models on an ordinary CPU?**

Public ASR leaderboards report accuracy on GPUs for a handful of large languages.
If you are shipping Turkish voice features to a kiosk, a laptop or an edge box,
you need different numbers: WER *and* real-time factor, latency and memory, measured
on the hardware you will actually deploy to. This project measures exactly that,
reproducibly.

- 🇹🇷 **Turkish-aware scoring** — correct `I/ı` and `İ/i` casing, apostrophe suffixes
  (`İstanbul'da`), circumflex (`kâğıt`). Generic normalizers silently inflate Turkish WER.
- ⏱️ **CPU-first metrics** — RTFx, p50/p90 latency, load time, peak RAM, model size.
- 🖥️ **Hardware fingerprinting** — every result records CPU, cores, RAM and library
  versions; results are only compared within the same hardware.
- 🔁 **Reproducible** — deterministic subsets, pinned threads, untimed downloads and
  warm-up; a GitHub Actions workflow anyone can re-run.
- 🔌 **Pluggable** — faster-whisper, Vosk and any Hugging Face CTC model out of the box;
  new backends are ~40 lines.

## Leaderboard

👉 **[huggingface.co/spaces/emreclskn/turkish-speech-bench](https://huggingface.co/spaces/emreclskn/turkish-speech-bench)**

## Quick start

```bash
pip install "turkish-speech-bench[whisper,vosk]"

# 200 FLEURS Turkish test utterances, 4 threads
tsb run -m faster-whisper:small -m vosk:vosk-model-small-tr-0.3 -n 200 -t 4 --label "my laptop"

tsb report            # Markdown leaderboard from ./results
tsb models            # suggested models
tsb normalize "İSTANBUL'da Kâğıt!"   # -> istanbulda kağıt
```

Your own data works too: a JSONL manifest with one `{"audio": "clip.wav", "text": "..."}`
per line, then `tsb run -m ... -d my_data.jsonl`.

### Backends

| Backend | Install extra | Example spec |
|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, INT8) | `whisper` | `faster-whisper:small`, `faster-whisper:large-v3-turbo` |
| [Vosk](https://alphacephei.com/vosk/) (Kaldi) | `vosk` | `vosk:vosk-model-small-tr-0.3` |
| Hugging Face CTC (wav2vec2, XLS-R, HuBERT…) | `ctc` | `hf-ctc:mpoyraz/wav2vec2-xls-r-300m-cv7-turkish` |

## How results are produced

Read [docs/methodology.md](docs/methodology.md) for the details: what each metric means,
the normalization rules, the timing protocol and known limitations.

Reference runs on GitHub-hosted runners come from the
[Benchmark workflow](.github/workflows/benchmark.yml), which opens a pull request with the
result files so every number is reviewed before it reaches the leaderboard.

## Contributing

Results from real devices (kiosks, NUCs, Raspberry Pi-class boards, laptops) are the most
valuable contribution — see [CONTRIBUTING.md](CONTRIBUTING.md). Model requests and new
backends are welcome too.

## License

Code: MIT. Each dataset and model keeps its own license; it is recorded in every
result file.
