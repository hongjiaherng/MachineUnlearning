"""Smoke tests for the Hydra CLIs.

Invokes the console scripts as subprocesses with the lightest possible config
(MNIST, 1 epoch, tiny batch) to verify config composition and end-to-end wiring.
Skipped unless RUN_CLI_TESTS=1 — these touch disk (dataset download, checkpoint
write) and take longer than the pure-unit tests in test_strategies.py.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CLI_TESTS") != "1",
    reason="set RUN_CLI_TESTS=1 to run CLI smoke tests",
)


def _run(cmd: list[str], tmp_path):
    env = {**os.environ, "HYDRA_FULL_ERROR": "1"}
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=tmp_path, env=env, timeout=600,
    )
    assert result.returncode == 0, f"{' '.join(cmd)} failed:\n{result.stderr}"
    return result


def test_mu_train_mnist_one_epoch(tmp_path):
    _run(
        [
            sys.executable, "-m", "machineunlearning.train",
            "dataset=mnist", "model=mlp", "epochs=1", "batch_size=128",
            "save_model=false",
        ],
        tmp_path,
    )
