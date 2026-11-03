"""Shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable however the tests are started (any folder, VS Code, older pytest).
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")          # never open plot windows during tests

import numpy as np
import pytest

from bench import FullFactorial, make_dataset


@pytest.fixture(scope="session")
def ff() -> FullFactorial:
    """The standard 41 x 41 full-factorial dataset used in the experiments."""
    return make_dataset(n_side=41, seed=0)


@pytest.fixture(scope="session")
def ff_small() -> FullFactorial:
    """A coarser 21 x 21 dataset that keeps strategy tests fast."""
    return make_dataset(n_side=21, seed=0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)
