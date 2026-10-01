"""Filesystem locations, resolved independently of the current working directory.

The notebooks run with different cwds (``Artem_notebook/`` for the main notebook,
the repo root for the secondary one), so nothing in this package may rely on cwd.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Repo root. Correct under ``pip install -e .``; override with ``DENGAI_ROOT``.
ROOT = Path(os.environ.get("DENGAI_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "figures"


def check_layout() -> None:
    """Fail loudly and usefully if the package cannot see the repo."""
    missing = [p.name for p in (DATA_DIR, CONFIG_DIR) if not p.is_dir()]
    if missing:
        raise RuntimeError(
            f"DengAI cannot find {missing} under {ROOT}. "
            "Install the project with `pip install -e .` from the repo root, "
            "or set the DENGAI_ROOT environment variable."
        )
