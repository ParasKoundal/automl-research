"""Execute training commands with timeout enforcement and log capture."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    exit_code: int
    wall_time: float
    crashed: bool
    log_path: Path
    error_tail: str = ""


def execute_training(
    command: str,
    cwd: str,
    log_path: Path,
    time_budget: int,
    env_vars: dict[str, str] | None = None,
) -> RunResult:
    """Run a training command with timeout and log capture.

    Args:
        command: Shell command to run.
        cwd: Working directory.
        log_path: Path to write stdout/stderr.
        time_budget: Max seconds before SIGTERM.
        env_vars: Extra environment variables to inject.

    Returns:
        RunResult with exit code, timing, and crash info.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    start = time.monotonic()

    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,  # create process group for clean kill
            )

            grace_period = 30  # seconds after SIGTERM before SIGKILL
            killed = False

            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if elapsed > time_budget and not killed:
                    # Send SIGTERM to entire process group
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    killed = True

                if killed and elapsed > time_budget + grace_period:
                    # Force kill
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break

                time.sleep(1)

            # Ensure process is done
            proc.wait(timeout=5)

    except Exception as e:
        wall_time = time.monotonic() - start
        return RunResult(
            exit_code=-1,
            wall_time=wall_time,
            crashed=True,
            log_path=log_path,
            error_tail=str(e),
        )

    wall_time = time.monotonic() - start
    exit_code = proc.returncode or 0

    # Read last 50 lines for error context if crashed
    error_tail = ""
    if exit_code != 0:
        try:
            lines = log_path.read_text().splitlines()
            error_tail = "\n".join(lines[-50:])
        except OSError:
            pass

    return RunResult(
        exit_code=exit_code,
        wall_time=wall_time,
        crashed=exit_code != 0,
        log_path=log_path,
        error_tail=error_tail,
    )
