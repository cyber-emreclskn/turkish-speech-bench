"""Hardware fingerprint recorded with every result, so numbers are only ever
compared between runs on the same kind of machine."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import psutil

_TRACKED_PACKAGES = (
    "faster-whisper",
    "ctranslate2",
    "vosk",
    "torch",
    "transformers",
    "onnxruntime",
)


def _cpu_name() -> str:
    try:
        import cpuinfo

        name = cpuinfo.get_cpu_info().get("brand_raw")
        if name:
            return str(name)
    except Exception:  # pragma: no cover - cpuinfo can fail in odd sandboxes
        pass
    return platform.processor() or platform.machine() or "unknown"


def _package_versions() -> dict[str, str]:
    out = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def _environment() -> str:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # "ImageOS" is the variable's real (mixed-case) name on GitHub runners.
        image = os.environ.get("ImageOS") or os.environ.get("RUNNER_OS", "unknown")  # noqa: SIM112
        return f"github-actions:{image}"
    return "local"


def _default_label(cpu: str, logical: int) -> str:
    # GitHub-hosted runners of one label run on several CPU models with very
    # different speeds, so the CPU is always part of the name.
    prefix = "GitHub Actions · " if _environment().startswith("github-actions") else ""
    return f"{prefix}{cpu} ({logical} threads)"


def fingerprint(label: str | None = None) -> dict[str, Any]:
    """Describe this machine. ``label`` is a human name such as ``"kiosk-n100"``."""
    cpu = _cpu_name()
    physical = psutil.cpu_count(logical=False) or 0
    logical = psutil.cpu_count(logical=True) or 0
    ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    hardware_id = hashlib.sha256(f"{cpu}|{physical}|{logical}|{ram_gb:.0f}".encode()).hexdigest()
    return {
        "label": label or _default_label(cpu, logical),
        "hardware_id": hardware_id[:12],
        "cpu": cpu,
        "cores_physical": physical,
        "cores_logical": logical,
        "ram_gb": ram_gb,
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "environment": _environment(),
        "packages": _package_versions(),
    }
