"""Package-level configuration.

Nothing here has side effects at import time.  ``OUTPUT_DIR`` (used only by
the legacy ``save=True`` plot options) defaults to ``./results`` and can be
set through the ``GRS_OUTPUT_DIR`` environment variable or
:func:`set_output_dir`.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from ._explainer import enable_logging, logger  # noqa: F401  (re-export)

OUTPUT_DIR = os.environ.get("GRS_OUTPUT_DIR", "results")


def set_output_dir(path) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = str(path)


def ensure_output_dir() -> str:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
