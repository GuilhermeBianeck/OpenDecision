"""Refuse to load a model the machine cannot hold alongside a desktop session.

Each catalog entry may carry ``memory_bytes``, the peak measured while scoring
the longest public benchmark states at the checkpoint's full context. Entries
without a measurement fall back to twice the weight size. Loading is refused
when available memory is below that requirement plus a reserve kept for the
rest of the machine, with an error that names smaller alternatives.

Set ``OPENDECISION_SKIP_MEMORY_CHECK=1`` to load anyway.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Any

from opendecision.errors import BackendError

# Left free for the operating system and the user's other applications.
RESERVE_BYTES = 1 << 30
# Python, tokenizer and framework overhead outside the measured model peak.
PROCESS_OVERHEAD_BYTES = 512 << 20
# MLX keeps freed GPU buffers for reuse; without a limit that cache grew by
# about 1.3 GB per long request and was never returned. 256 MiB keeps reuse
# for short prompts and holds total memory flat on long ones.
MLX_CACHE_LIMIT_BYTES = 256 << 20


def available_bytes() -> int | None:
    """Memory the OS can hand out now without swapping, or None if unknown."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as meminfo:
                for line in meminfo:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        elif system == "Darwin":
            # The kernel's own availability estimate, the figure memory_pressure
            # reports: it counts clean file-cache pages macOS drops on demand.
            level, total = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_level", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.split()
            if 0 < int(level) <= 100:
                return int(total) * int(level) // 100
            output = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5, check=True
            ).stdout
            page = int(re.search(r"page size of (\d+) bytes", output).group(1))
            counts = {
                name.strip().lower(): int(value)
                for name, value in re.findall(r"^(Pages [^:]+):\s+(\d+)\.", output, re.M)
            }
            # Free pages plus those macOS reclaims without writing anything out.
            reclaimable = ("pages free", "pages inactive", "pages speculative", "pages purgeable")
            return page * sum(counts.get(name, 0) for name in reclaimable)
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return None
    return None


def required_bytes(spec: dict[str, Any]) -> int:
    """Measured peak plus process overhead, or an estimate from the weights."""
    measured = spec.get("memory_bytes")
    if measured is not None:
        return int(measured) + PROCESS_OVERHEAD_BYTES
    return 2 * int(spec.get("weights_bytes") or 0) + PROCESS_OVERHEAD_BYTES


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def check_memory(spec: dict[str, Any], *, available: int | None = None) -> None:
    """Raise :class:`BackendError` when ``spec`` does not fit in free memory."""
    if os.environ.get("OPENDECISION_SKIP_MEMORY_CHECK") == "1":
        return
    available = available_bytes() if available is None else available
    if available is None:
        return
    needed = required_bytes(spec) + RESERVE_BYTES
    if available >= needed:
        return
    from opendecision.backends.catalog import MODEL_SPECS

    smaller = sorted(
        (required_bytes(other), name)
        for name, other in MODEL_SPECS.items()
        if required_bytes(other) + RESERVE_BYTES <= available
    )
    hint = (
        f" Models that fit now: {', '.join(name for _, name in smaller[-3:])}."
        if smaller
        else " Close other applications first."
    )
    raise BackendError(
        f"Not enough free memory to load {spec['name']}: it needs about "
        f"{_gb(required_bytes(spec))} plus {_gb(RESERVE_BYTES)} kept free for the rest of "
        f"the system, and {_gb(available)} is available.{hint} "
        "Set OPENDECISION_SKIP_MEMORY_CHECK=1 to load anyway."
    )
