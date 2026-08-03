"""Shared helpers for the D-017 integration suite.

Mirrors the orchestrator suite patterns: poll-with-deadline ``wait_for``,
``_free_port`` + EADDRINUSE retry subprocess spawning, environment
building (with the PYTHONPATH leak scrubbed), and log tailing on failure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # solar-control/
TESTS_INTEGRATION = REPO_ROOT / "tests_integration"

DATA_REPO_ROOT = REPO_ROOT.parent / "data-repository"
SOLAR_HOST_ROOT = REPO_ROOT.parent / "solar-host"

# Per-repo interpreter for subprocesses (each service runs in its own venv
# with its own pinned deps — mirrors the plan's per-service dependencies).
DATA_REPO_PYTHON = DATA_REPO_ROOT / ".venv" / "bin" / "python"
SOLAR_CONTROL_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SOLAR_HOST_PYTHON = SOLAR_HOST_ROOT / ".venv" / "bin" / "python"


def clean_env() -> dict[str, str]:
    """Return a copy of os.environ safe for service subprocesses.

    The Hermes desktop app leaks ``PYTHONPATH`` pointing at its own venv
    site-packages; a service subprocess inheriting it imports mismatched
    binaries (pydantic_core etc.). Strip it (and any Python-home leaks).
    """
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    return env


def free_port() -> int:
    """Return a currently-free loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def wait_for(
    condition: Callable[[], Awaitable[bool]] | Callable[[], bool],
    timeout: float = 30.0,
    interval: float = 0.25,
    description: str = "condition",
) -> None:
    """Poll *condition* until truthy or the deadline passes (assert on timeout)."""
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            result = condition()
            if asyncio.iscoroutine(result):
                result = await result
            if result:
                return
            last = result
        except Exception as exc:  # noqa: BLE001
            last = exc
        await asyncio.sleep(interval)
    raise AssertionError(
        f"Timed out after {timeout}s waiting for {description} (last={last!r})"
    )


def wait_for_sync(
    condition: Callable[[], bool],
    timeout: float = 30.0,
    interval: float = 0.25,
    description: str = "condition",
) -> None:
    """Synchronous variant of ``wait_for`` (for non-async fixture teardown)."""
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            if condition():
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(interval)
    raise AssertionError(
        f"Timed out after {timeout}s waiting for {description} (last={last!r})"
    )


def _url_ready(url: str, timeout: float = 30.0) -> bool:
    """Poll an HTTP(S) URL until it returns 200 (sync; for spawn readiness)."""
    import urllib.request

    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(0.25)
    logger.error("URL %s never became ready (last error: %r)", url, last)
    return False


class ServiceProcess:
    """A managed uvicorn subprocess with log capture."""

    def __init__(
        self,
        name: str,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_dir: Path,
        ready_url: str,
        port: int,
    ) -> None:
        self.name = name
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.log_path = log_dir / f"{name}.log"
        self.ready_url = ready_url
        self.port = port
        self.proc: subprocess.Popen[Any] | None = None

    def start(self, ready_timeout: float = 45.0) -> None:
        log_dir = self.log_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            self.argv,
            cwd=str(self.cwd),
            env=self.env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        logger.info(
            "spawned %s pid=%d: %s (cwd=%s, log=%s)",
            self.name,
            self.proc.pid,
            " ".join(self.argv),
            self.cwd,
            self.log_path,
        )
        if not _url_ready(self.ready_url, timeout=ready_timeout):
            self.terminate()
            raise AssertionError(
                f"{self.name} did not become ready at {self.ready_url}\n"
                f"--- log tail ({self.log_path}) ---\n{self.tail(60)}"
            )

    def tail(self, lines: int = 40) -> str:
        if not self.log_path.exists():
            return "(no log)"
        data = self.log_path.read_bytes()
        return "\n".join(data.decode(errors="replace").splitlines()[-lines:])

    def terminate(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            except Exception:  # noqa: BLE001
                pass
        self.proc = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def spawn_service(
    *,
    name: str,
    python: Path,
    module: str,
    port: int,
    cwd: Path,
    env: dict[str, str],
    log_dir: Path,
    ready_path: str = "/health",
    host: str = "127.0.0.1",
    extra_args: list[str] | None = None,
    ready_timeout: float = 45.0,
) -> tuple[ServiceProcess, int]:
    """Spawn ``python -m uvicorn <module> --host .. --port ..`` with retry.

    Returns (ServiceProcess, actual_port) — the port may differ from the
    requested one if a race forced a retry (EADDRINUSE / readiness failure).
    """
    attempts = 0
    while True:
        attempts += 1
        argv = [
            str(python),
            "-m",
            "uvicorn",
            module,
            "--host",
            host,
            "--port",
            str(port),
            "--log-level",
            "info",
        ]
        if extra_args:
            argv.extend(extra_args)
        proc = ServiceProcess(
            name=name,
            argv=argv,
            cwd=cwd,
            env=env,
            log_dir=log_dir,
            ready_url=f"http://{host}:{port}{ready_path}",
            port=port,
        )
        try:
            proc.start(ready_timeout=ready_timeout)
            return proc, port
        except AssertionError:
            if attempts >= 3:
                raise
            logger.warning("spawn of %s failed on port %d, retrying", name, port)
            port = free_port()


def build_subprocess_env(base_env: dict[str, str] | None = None, **overrides: Any) -> dict[str, str]:
    """Build a subprocess env: clean base + service overrides (non-empty only).

    All values are coerced to str — testcontainers exposes ports as int.
    """
    env = clean_env()
    if base_env:
        env.update({k: str(v) for k, v in base_env.items()})
    for key, value in overrides.items():
        if value is not None:
            env[key] = str(value)
    return env


def tail_service_logs(services: list[ServiceProcess]) -> str:
    """Dump log tails of all service processes (used on fixture failure)."""
    parts = []
    for svc in services:
        if svc is not None:
            parts.append(f"===== {svc.name} (alive={svc.alive}) =====")
            parts.append(svc.tail(60))
    return "\n".join(parts)


def make_certs(cert_dir: Path) -> tuple[Path, Path, Path]:
    """Generate a self-signed CA + 127.0.0.1 server cert via openssl.

    Returns (ca_cert, server_cert, server_key). Point ``SSL_CERT_FILE`` /
    ``REQUESTS_CA_BUNDLE`` at ``ca_cert`` in subprocess envs so the real
    clients (httpx for HarborClient, requests for oras-py) trust the stub.
    """
    cert_dir.mkdir(parents=True, exist_ok=True)
    ca_key = cert_dir / "ca.key"
    ca_crt = cert_dir / "ca.crt"
    srv_key = cert_dir / "server.key"
    srv_csr = cert_dir / "server.csr"
    srv_crt = cert_dir / "server.crt"
    ext = cert_dir / "san.ext"

    def run(*args: str) -> None:
        subprocess.run(args, check=True, capture_output=True)

    run("openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", str(ca_key),
        "-out", str(ca_crt), "-days", "30", "-nodes",
        "-subj", "/CN=StubHarbor Test CA",
        "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign")
    run("openssl", "req", "-newkey", "rsa:2048", "-keyout", str(srv_key),
        "-out", str(srv_csr), "-nodes", "-subj", "/CN=127.0.0.1")
    ext.write_text(
        "subjectAltName=IP:127.0.0.1\n"
        "keyUsage=digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
    )
    run("openssl", "x509", "-req", "-in", str(srv_csr), "-CA", str(ca_crt),
        "-CAkey", str(ca_key), "-CAcreateserial", "-out", str(srv_crt),
        "-days", "30", "-extfile", str(ext))
    return ca_crt, srv_crt, srv_key


def which(binary: str) -> str | None:
    return shutil.which(binary)
