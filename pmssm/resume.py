"""Resume-from-checkpoint support for active learning runs.

At the end of each AL iteration the driver writes `state.pt` at the run's
output directory. On resume, the same output directory is reused: the loop
starts from `saved_iteration + 1` and runs `--n-additional-iterations` more.

A single `state.pt` is overwritten each iteration (atomically via tmp+rename),
so if a SIGKILL lands mid-iteration, the previous iteration's state is intact.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


STATE_FILENAME = "state.pt"


def capture_rng() -> dict:
    """Snapshot all RNG states."""
    return {
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(rng: Mapping[str, Any]) -> None:
    torch.random.set_rng_state(rng["torch_cpu"])
    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


def save_state(output_dir: str | os.PathLike, state: Mapping[str, Any]) -> None:
    """Atomically write state.pt inside output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    final = out / STATE_FILENAME
    tmp = out / (STATE_FILENAME + ".tmp")
    torch.save(dict(state), tmp)
    os.replace(tmp, final)  # atomic on POSIX


def load_state(output_dir: str | os.PathLike) -> dict | None:
    """Load state.pt from output_dir, or None if missing."""
    p = Path(output_dir) / STATE_FILENAME
    if not p.exists():
        return None
    return torch.load(p, weights_only=False, map_location="cpu")
