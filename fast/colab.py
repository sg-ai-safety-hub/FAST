"""Environment setup and reporting for Colab lab notebooks."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

__all__ = ["ci_mode", "gpu_info", "in_colab", "mount_drive", "setup"]


def in_colab() -> bool:
    return "google.colab" in sys.modules


def ci_mode() -> bool:
    """True when running under CI, where there is no GPU and no time budget.

    Labs that train anything should scale themselves down rather than be excluded:

        steps = 10 if ci_mode() else 500

    That keeps the lab genuinely verified on every push. A lab excluded from CI is a lab
    nobody notices is broken until Wednesday morning.
    """
    return os.environ.get("FAST_CI") == "1"


def gpu_info() -> dict | None:
    """Name and memory of the attached GPU, or None if running on CPU."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "memory_gb": round(props.total_memory / 1024**3, 1)}


def setup(require_gpu: bool = True, seed: int = 0) -> None:
    """Seed everything, report the environment, and fail loudly if the GPU is missing.

    Call this in the second cell of every lab. Failing here — with an instruction —
    is much cheaper than failing forty minutes later inside a training loop.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    gpu = gpu_info()
    print(f"python {sys.version.split()[0]}  ·  colab={in_colab()}  ·  seed={seed}")
    if gpu:
        print(f"gpu    {gpu['name']} ({gpu['memory_gb']} GB)")
    else:
        print("gpu    none — running on CPU")
        if require_gpu and not ci_mode():
            raise RuntimeError(
                "This lab needs a GPU. In Colab: Runtime > Change runtime type > "
                "T4 GPU (or better), then re-run this cell."
            )


def mount_drive(subdir: str = "fast") -> Path:
    """Mount Google Drive and return a working directory inside it.

    Use for checkpointing anything expensive. Colab runtimes disconnect, and a
    participant who loses one at minute 40 of a 60-minute lab should not start over.
    """
    if not in_colab():
        path = Path.home() / ".cache" / "fast" / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    from google.colab import drive

    drive.mount("/content/drive")
    path = Path("/content/drive/MyDrive") / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path
