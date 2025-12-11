"""
Sidecar Environment for Mini SWE Agent.

This module implements the SidecarEnvironment class, which sets up a dual-container
Docker environment for the agentic framework. It includes:
- An agent container running the Mini SWE Agent framework.
- A dev container for code execution.
- Docker networking to connect the two containers securely.

The agent environment allows the agent to execute code in an isolated dev container
while maintaining its own environment for decision-making and tool usage.
"""

import logging
import os
import shlex
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SidecarEnvironmentConfig:
    dev_host: str = "localhost"
    """Hostname of the dev container (for nc connection)."""
    dev_port: int = 9000
    """Port on the dev container where nc is listening."""
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the containers."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the containers.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the containers."""


class SidecarEnvironment:
    def __init__(self, *, config_class: type = SidecarEnvironmentConfig, logger: logging.Logger | None = None, **kwargs):
        """This class sets up a sidecar environment that executes commands in a dev container via nc.
        Each command execution creates a new nc connection, no persistent shell.
        See `SidecarEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)

        if "cwd" in kwargs:
            self.logger.warning("The 'cwd' parameter is set but not used in SidecarEnvironment.")

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the dev container via nc (non-persistent connection)."""
        timeout = timeout or self.config.timeout

        dev_host = self.config.dev_host
        dev_port = str(self.config.dev_port)

        # Build remote shell command
        exit_prefix = "EXIT_CODE:"
        remote_cmd = (
            f"{command} ; "
            f"echo '{exit_prefix}'$?\n"
            f"exit\n"
        )

        proc = subprocess.Popen(
            ["nc", dev_host, dev_port],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            stdout, _ = proc.communicate(remote_cmd, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            return {
                "output": stdout,
                "returncode": -1,
            }

        # Default return code
        returncode = 0

        # Parse EXIT_CODE from remote output
        if exit_prefix in stdout:
            try:
                exit_line = next(
                    line for line in stdout.splitlines() if exit_prefix in line
                )
                returncode = int(exit_line.split(exit_prefix, 1)[1].strip())
            except (StopIteration, ValueError):
                pass

            # Strip EXIT_CODE line from output
            stdout = stdout.split(exit_prefix, 1)[0].rstrip()

        return {
            "output": stdout,
            "returncode": returncode,
        }

    def cleanup(self):
        """No cleanup needed for non-persistent connections."""
        pass

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup()
