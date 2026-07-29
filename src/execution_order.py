"""Deterministic interleaving for model-backed benchmark configurations."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable
from typing import TypeVar


T = TypeVar("T")


def interleaved_product(*dimensions: Iterable[T], seed: int) -> list[tuple[T, ...]]:
    """Return every Cartesian-product task once, in a reproducible mixed order."""

    tasks = list(itertools.product(*dimensions))
    random.Random(seed).shuffle(tasks)
    return tasks
