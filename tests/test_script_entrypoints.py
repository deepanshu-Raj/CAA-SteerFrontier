import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

ACTIVE_SCRIPTS = [
    "check_model_adapter.py",
    "run_experiment.py",
    "run_cross_behavior.py",
    "plot_behavior_artifacts.py",
    "plot_comparison_artifacts.py",
    "plot_cross_model_artifacts.py",
]


@pytest.mark.parametrize("script_name", ACTIVE_SCRIPTS)
def test_active_script_is_valid_python(script_name):
    path = SCRIPTS / script_name
    assert path.exists(), f"Missing active script: {path}"

    source = path.read_text()
    ast.parse(source, filename=str(path))
    compile(source, str(path), "exec")
    assert 'if __name__ == "__main__":' in source


@pytest.mark.parametrize("script_name", ACTIVE_SCRIPTS)
def test_script_help_does_not_trigger_run(script_name):
    """Exercise imports + CLI construction only; --help exits before execution."""
    path = SCRIPTS / script_name

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MPLBACKEND"] = "Agg"

    # Plotting scripts can cause Matplotlib to build a font/config cache on
    # first import. Redirect that cache to a temporary directory that is
    # deleted immediately after the smoke test.
    with tempfile.TemporaryDirectory(prefix="caa-mpl-smoke-") as mpl_config:
        env["MPLCONFIGDIR"] = mpl_config

        completed = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_expected_active_script_set_is_present():
    actual = {path.name for path in SCRIPTS.glob("*.py")}
    assert set(ACTIVE_SCRIPTS).issubset(actual)
