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
    network_name: str = "minisweagent-net"
    dev_container_name: str = "minisweagent-dev"
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
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """


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
        self._setup_connection()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _setup_connection(self):
        """Establish a persistent TCP connection to the dev container using nc."""
        try:
            self.nc_process = subprocess.Popen(
                ['nc', self.config.dev_container_name, '9000'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.logger.info(f"Connected to dev container at {self.config.dev_container_name}:9000 using nc")
        except OSError as e:
            self.logger.error(f"Failed to start nc process: {e}")
            raise

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the dev container via the nc process and return the result as a dict."""
        cwd = cwd or self.config.cwd
        timeout = timeout or self.config.timeout

        if not self.nc_process:
            raise RuntimeError("nc process not started")

        # Send command with cwd change and a marker to indicate completion
        full_command = f"cd {shlex.quote(cwd)}; {command}; echo 'COMMAND_DONE_MARKER'\n"
        full_command_bytes = full_command.encode('utf-8')
        self.nc_process.stdin.write(full_command_bytes)
        self.nc_process.stdin.flush()

        # Read output until the marker is found
        output = b""
        marker = b'COMMAND_DONE_MARKER'
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.nc_process.poll() is not None:
                break  # process ended
            data = self.nc_process.stdout.read(4096)
            if not data:
                break
            output += data
            if marker in output:
                break

        # Extract output before the marker
        output_str = output.decode('utf-8', errors='replace')
        if 'COMMAND_DONE_MARKER' in output_str:
            output_str = output_str.split('COMMAND_DONE_MARKER')[0].strip()

        return {"output": output_str, "returncode": 0}

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
