"""Run on a single SWE-Bench instance."""

import traceback
from pathlib import Path

import typer
import yaml

from minisweagent import global_config_dir
from minisweagent.agents.interactive import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger

app = typer.Typer(add_completion=False)

DEFAULT_OUTPUT = global_config_dir / "last_single_run.traj.json"
DEFAULT_TASK = "Say hello world to confirm that the Mini SWE Agent is working."


# fmt: off
@app.command()
def main(
    config_path: Path = typer.Option( builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file", rich_help_panel="Basic"),
    task: str | None = typer.Option(None, "-t", "--task", help="Task/problem statement", rich_help_panel="Basic"),
) -> None:
    # fmt: on
    """Run on a single SWE-Bench instance."""

    config_path = get_config_path(config_path)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    agent = DefaultAgent(
        get_model(None, config.get("model", {})),
        LocalEnvironment(),
        **config.get("agent", {}),
    )

    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run({"task": task or DEFAULT_TASK})  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]


if __name__ == "__main__":
    app()
