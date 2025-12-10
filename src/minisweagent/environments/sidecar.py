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
        """This class sets up a sidecar environment with two Docker containers:
        one for the agent framework and one for code execution, connected via Docker networking.
        See `SidecarEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.agent_container_id: str | None = None
        self.dev_container_id: str | None = None
        self.network_id: str | None = None
        self.nc_process: subprocess.Popen | None = None
        self.config = config_class(**kwargs)

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        cwd = cwd or self.config.cwd
        timeout = timeout or self.config.timeout

        dev_host = self.config.dev_host      # e.g. "localhost" or "dev"
        dev_port = str(self.config.dev_port) # e.g. "9000"

        # Build remote shell command
        exit_prefix = "EXIT_CODE:"
        remote_cmd = (
            f"cd {shlex.quote(cwd)} && "
            f"{command} ; "
            f"echo '{exit_prefix}'$?\n"
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
                "returncode": -1,  # or some sentinel for timeout
                "timeout": True,
            }

        # Default return code if parsing fails
        returncode = proc.returncode

        # Parse EXIT_CODE from remote output
        if exit_prefix in stdout:
            try:
                exit_line = next(
                    line for line in stdout.splitlines() if exit_prefix in line
                )
                returncode = int(exit_line.split(exit_prefix, 1)[1].strip())
            except (StopIteration, ValueError):
                pass

            # Strip metadata from output
            stdout = stdout.split(exit_prefix, 1)[0].rstrip()

        return {
            "output": stdout,
            "returncode": returncode,
            "timeout": False,
        }

    def cleanup(self):
        """Terminate the nc process."""
        if self.nc_process:
            try:
                self.nc_process.terminate()
                self.nc_process.wait(timeout=5)
                self.logger.info("Terminated nc process")
            except subprocess.TimeoutExpired:
                self.nc_process.kill()
                self.logger.warning("Force killed nc process")
            except Exception as e:
                self.logger.error(f"Error terminating nc process: {e}")
            self.nc_process = None

    def __del__(self):
        """Cleanup socket when object is destroyed."""
        self.cleanup()
