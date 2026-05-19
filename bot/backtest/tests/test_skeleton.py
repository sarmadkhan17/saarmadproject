import subprocess
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[3]


def _run(module: str, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'bot') + ':' + str(ROOT)
    return subprocess.run(
        ["python3", "-m", module, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
    )


def test_agent_eval_cli_help():
    result = _run("bot.backtest.agent_eval", "--help")
    assert result.returncode == 0
    assert "agent_eval" in result.stdout.lower() or "agent" in result.stdout.lower()


def test_agent_eval_unimplemented_returns_nonzero_with_clear_message():
    result = _run("bot.backtest.agent_eval", "--run")
    assert result.returncode != 0
    assert "not implemented" in result.stderr.lower() or "not implemented" in result.stdout.lower()


def test_pipeline_eval_cli_help():
    result = _run("bot.backtest.pipeline_eval", "--help")
    assert result.returncode == 0


def test_pipeline_eval_run_gracefully_handles_no_symbols(tmp_path):
    # Phase 2: harness is implemented; --run with nonexistent parquet dir
    # exits cleanly with 0 since no symbols match → loop is empty → JSON
    # written with n_evaluations=0. This avoids writing to data/baselines/.
    out_path = tmp_path / "out.json"
    result = _run("bot.backtest.pipeline_eval", "--run",
                  "--parquet", "/tmp/nonexistent_cryptobot_parquet_xyz",
                  "--out", str(out_path))
    assert result.returncode == 0
