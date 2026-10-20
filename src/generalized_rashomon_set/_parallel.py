"""Thin joblib layer: chunking, seeding and sequential fallback."""
from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs


def chunk(items: Sequence, n_chunks: int) -> List[list]:
    """Split ``items`` into at most ``n_chunks`` non-empty, contiguous lists."""
    items = list(items)
    if not items:
        return []
    n_chunks = max(1, min(n_chunks, len(items)))
    return [[items[i] for i in idx]
            for idx in np.array_split(np.arange(len(items)), n_chunks) if len(idx)]


def spawn_seeds(random_state, stage: int, n: int):
    """Independent, reproducible seeds for ``n`` workers of pipeline ``stage``."""
    ss = np.random.SeedSequence(random_state, spawn_key=(stage,))
    return ss.spawn(n)


def run_jobs(fn: Callable, job_args: List[tuple], n_jobs=1, backend="loky", verbose=0,
             prefer_chunks_per_worker: int = 4):
    """Run ``fn(*args)`` for every ``args`` in ``job_args``.

    ``job_args`` is already chunked by the caller; this only dispatches.
    """
    if n_jobs in (None, 1) or len(job_args) <= 1:
        return [fn(*a) for a in job_args]
    return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(fn)(*a) for a in job_args)


def n_chunks_for(n_items: int, n_jobs, per_worker: int = 4) -> int:
    workers = effective_n_jobs(n_jobs) if n_jobs not in (None, 1) else 1
    if workers == 1:
        return 1
    return max(1, min(n_items, workers * per_worker))
