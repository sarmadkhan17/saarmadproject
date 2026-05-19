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


def test_pipeline_eval_run_without_components_exits_nonzero():
    # Phase 2: harness is implemented; --run fails only because
    # components.trend_filter (Task 6) is not yet present.
    result = _run("bot.backtest.pipeline_eval", "--run")
    assert result.returncode != 0
