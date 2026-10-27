"""Turkish Speech Bench leaderboard (Hugging Face Space).

Reads the result JSON files that live in ``results/`` next to this app; the
GitHub repository syncs them here on every merge to main.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from tsbench.report import load_results, rows

RESULTS_DIR = Path(__file__).parent / "results"
REPO = "https://github.com/cyber-emreclskn/turkish-speech-bench"
ALL = "All hardware"

COLUMNS = [
    ("model", "Model"),
    ("wer", "WER %"),
    ("cer", "CER %"),
    ("rtfx", "RTFx"),
    ("latency_p90_s", "p90 latency (s)"),
    ("peak_rss_mb", "Peak RAM (MB)"),
    ("disk_mb", "Disk (MB)"),
    ("params_m", "Params (M)"),
    ("threads", "Threads"),
    ("hardware", "Hardware"),
    ("dataset", "Dataset"),
    ("date", "Date"),
]

INTRO = f"""
# 🇹🇷 Turkish Speech Bench

How accurate **and how fast** are Turkish speech recognition models on ordinary CPUs?
Every number here was measured with the open-source [`turkish-speech-bench`]({REPO}) tool,
with Turkish-aware text normalization (I/ı, İ/i, apostrophes, circumflex) and the hardware
recorded alongside each result. **Only compare rows from the same hardware.**

**WER/CER** lower is better · **RTFx** = seconds of audio processed per second (higher is faster) ·
**Peak RAM** = whole process.
"""


def load_rows() -> list[dict[str, Any]]:
    return rows(load_results([RESULTS_DIR])) if RESULTS_DIR.exists() else []


def table(hardware: str, dataset: str, max_ram: float | None) -> list[list[Any]]:
    out = []
    for r in load_rows():
        if hardware != ALL and r["hardware"] != hardware:
            continue
        if dataset != ALL and r["dataset"] != dataset:
            continue
        if max_ram and r["peak_rss_mb"] > max_ram:
            continue
        row = dict(r, wer=round(r["wer"] * 100, 2), cer=round(r["cer"] * 100, 2))
        out.append([row[key] for key, _ in COLUMNS])
    return out


def build() -> gr.Blocks:
    data = load_rows()
    hardware = [ALL, *sorted({r["hardware"] for r in data})]
    datasets = [ALL, *sorted({r["dataset"] for r in data})]
    with gr.Blocks(title="Turkish Speech Bench") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            hw = gr.Dropdown(
                hardware, value=hardware[-1] if len(hardware) == 2 else ALL, label="Hardware"
            )
            ds = gr.Dropdown(datasets, value=ALL, label="Dataset")
            ram = gr.Number(label="Max peak RAM (MB, optional)", value=None)
        grid = gr.Dataframe(
            value=table(hw.value, ds.value, None),
            headers=[label for _, label in COLUMNS],
            interactive=False,
            wrap=True,
        )
        for control in (hw, ds, ram):
            control.change(table, [hw, ds, ram], grid)
        gr.Markdown(
            f"Add your hardware: `pip install 'turkish-speech-bench[whisper]'` then "
            f"`tsb run -m faster-whisper:small --label 'my-cpu'` and open a PR with the JSON. "
            f"[Methodology]({REPO}/blob/main/docs/methodology.md)"
        )
    return demo


if __name__ == "__main__":
    build().launch()
