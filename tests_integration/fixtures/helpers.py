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

    def kill(self) -> None:
        """SIGKILL the process group immediately (port closes instantly).

        Unlike ``terminate()`` (graceful SIGTERM), a dying uvicorn keeps its
        port bound during shutdown, so TCP connects succeed while requests
        hang — used by tests that need the service deterministically down.
        """
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
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


def build_subprocess_env(
    base_env: dict[str, str] | None = None, **overrides: Any
) -> dict[str, str]:
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

    run(
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(ca_key),
        "-out",
        str(ca_crt),
        "-days",
        "30",
        "-nodes",
        "-subj",
        "/CN=StubHarbor Test CA",
        "-addext",
        "basicConstraints=critical,CA:TRUE",
        "-addext",
        "keyUsage=critical,keyCertSign,cRLSign",
    )
    run(
        "openssl",
        "req",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(srv_key),
        "-out",
        str(srv_csr),
        "-nodes",
        "-subj",
        "/CN=127.0.0.1",
    )
    ext.write_text(
        "subjectAltName=IP:127.0.0.1\n"
        "keyUsage=digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
    )
    run(
        "openssl",
        "x509",
        "-req",
        "-in",
        str(srv_csr),
        "-CA",
        str(ca_crt),
        "-CAkey",
        str(ca_key),
        "-CAcreateserial",
        "-out",
        str(srv_crt),
        "-days",
        "30",
        "-extfile",
        str(ext),
    )
    return ca_crt, srv_crt, srv_key


def which(binary: str) -> str | None:
    return shutil.which(binary)


# ── Diagnostics / evidence preservation (D-017 flake) ──────────────


async def registry_entries_for_alias(redis_url: str, alias: str) -> list[dict]:
    """Read control's ``solar:registry`` entries for *alias* (Redis HSET).

    Each entry is a RegistryEntry dict with ``host_id``, ``instance_id``,
    ``url``, ``api_key``, ``model_alias``, ``supported_endpoints``,
    ``backend_type``. Returns [] when the alias is not registered or the
    registry is unreachable (a diagnostics helper must never raise).
    """
    import json

    import redis as redis_lib

    try:
        r = redis_lib.from_url(redis_url)
        try:
            raw = r.hget("solar:registry", alias)
        finally:
            r.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("registry read failed for %s: %s", alias, exc)
        return []
    if raw is None:
        return []
    return json.loads(raw)


async def dump_instance_evidence(
    stack: Any,
    alias: str,
    *,
    registry_entries: list[dict] | None = None,
) -> Path:
    """Dump every piece of evidence needed to diagnose a classify failure.

    Writes into ``<stack tmp>/evidence-<alias>/`` next to the pytest log
    dirs: a copy of the instance's server log (per host), sha256 of every
    pulled model file vs the committed fixture, the gateway registry
    entries for the alias (supported_endpoints — disambiguates the
    routing trap from a dead server), a direct upstream probe of each
    candidate instance (bypassing the gateway's fallback entry), and the
    host venv's tokenizer-related package versions (F6 hypothesis 2).

    Returns the evidence dir path.
    """
    import hashlib
    import subprocess

    import httpx

    from fixtures.constants import FIXTURE_MODEL_DIR
    from fixtures.seed import read_test_model_files

    evidence_dir = stack.tmp_dir / f"evidence-{alias}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"evidence dump for alias {alias}"]

    def sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    letters = ("a", "b") + tuple(stack.extra_host_urls.keys())

    # 1. Copy the instance's server log(s) from every host's LOG_DIR.
    for letter in letters:
        log_dir = stack.tmp_dir / f"logs-{letter}"
        if not log_dir.is_dir():
            continue
        matches = sorted(log_dir.glob(f"{alias}_*.log"))
        for log in matches:
            dest = evidence_dir / f"server-{letter}-{log.name}"
            shutil.copy2(log, dest)
            lines.append(f"[host-{letter}] server log copied -> {dest.name}")

    # 2. sha256 of every pulled file vs the committed fixture.
    fixture_files = read_test_model_files(FIXTURE_MODEL_DIR)
    expected = {name: sha256_bytes(data) for name, data in fixture_files.items()}
    for letter in letters:
        models_dir = stack.models_dir(letter)
        if not models_dir.is_dir():
            continue
        for artifact_dir in sorted(models_dir.glob("repo--test-model--*")):
            lines.append(f"[host-{letter}] pulled artifact {artifact_dir.name}:")
            for path in sorted(artifact_dir.iterdir()):
                if not path.is_file():
                    continue
                actual = sha256_bytes(path.read_bytes())
                exp = expected.get(path.name, "(not in committed fixture)")
                verdict = "MATCH" if actual == exp else "DIFF"
                lines.append(f"    {path.name}: {actual} expected={exp} {verdict}")

    # 3. Gateway registry entries for the alias (routing-trap check).
    if registry_entries is None:
        registry_entries = await registry_entries_for_alias(
            stack.db_env["redis"], alias
        )
    lines.append(f"gateway registry entries for {alias}: {len(registry_entries)}")
    for e in registry_entries:
        lines.append(
            "  host_id={host_id} instance_id={instance_id} url={url} "
            "endpoints={supported_endpoints}".format(**e)
        )

    # 4. Direct upstream probe of each candidate instance URL. If the
    #    gateway /v1/models still lists the alias but every direct probe
    #    fails, the listing is the fabricated fallback (gateway.py
    #    get_available_models), not a live server.
    gateway_lists_alias = False
    try:
        resp = await http_control_get_models(stack)
        names = {m.get("name") for m in resp.get("models", [])} | {
            m.get("id") for m in resp.get("data", [])
        }
        gateway_lists_alias = alias in names
    except Exception as exc:  # noqa: BLE001
        lines.append(f"gateway /v1/models probe failed: {type(exc).__name__}: {exc}")
    lines.append(f"gateway /v1/models lists {alias}: {gateway_lists_alias}")
    for e in registry_entries:
        url = f"{e.get('url')}/v1/models"
        api_key = e.get("api_key") or ""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                probe = await client.get(
                    url, headers={"Authorization": f"Bearer {api_key}"}
                )
            lines.append(f"direct probe {url}: HTTP {probe.status_code}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"direct probe {url}: ERROR {type(exc).__name__}: {exc}")
    if registry_entries and gateway_lists_alias:
        dead = all(
            not _probe_alive(str(e.get("url") or ""), str(e.get("api_key") or ""))
            for e in registry_entries
        )
        lines.append(
            "fallback-entry verdict: "
            + (
                "alias listed by gateway but NO upstream alive -> fabricated "
                "fallback entry (gateway.py get_available_models)"
                if dead
                else "at least one upstream alive"
            )
        )

    # 5. Host venv package versions (F6 hypothesis 2).
    env = clean_env()
    proc = subprocess.run(
        [
            str(SOLAR_HOST_PYTHON),
            "-m",
            "pip",
            "freeze",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    wanted = (
        "tokenizers",
        "sentencepiece",
        "tiktoken",
        "transformers",
        "torch==",
        "safetensors",
    )
    if proc.returncode == 0:
        lines.append("host venv versions (pip freeze grep):")
        for line in proc.stdout.splitlines():
            if any(line.startswith(w) for w in wanted):
                lines.append(f"    {line}")
    else:
        lines.append(f"host venv pip freeze failed: {proc.stderr[:200]}")

    (evidence_dir / "evidence.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    logger.warning("classify-failure evidence dumped to %s", evidence_dir)
    return evidence_dir


async def http_control_get_models(stack: Any) -> dict:
    """GET /v1/models through control (used by the evidence dump)."""
    import httpx

    from fixtures.constants import MANAGEMENT_API_KEY

    async with httpx.AsyncClient(
        base_url=stack.control_url,
        headers={"X-API-Key": MANAGEMENT_API_KEY},
        timeout=10.0,
    ) as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200, resp.text
        return resp.json()


def _probe_alive(url: str, api_key: str) -> bool:
    """Synchronous best-effort liveness probe (used by the verdict line)."""
    import urllib.request

    try:
        req = urllib.request.Request(
            f"{url}/v1/models", headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


_SAFETENSORS_SCRIPT = r"""
import json, sys
import safetensors
import safetensors.torch

src, out, meta_json = sys.argv[1], sys.argv[2], sys.argv[3]
meta = json.loads(meta_json)
with open(src, "rb") as f:
    tensors = safetensors.torch.load(f.read())  # bytes -> tensors dict only
safetensors.torch.save_file(tensors, out, metadata=meta)
# Guard: the rewrite must round-trip with identical tensors (bit-identical
# weights -> identical logits -> the fixture's score > 0.0 assertions hold).
tensors2 = safetensors.torch.load_file(out)
assert set(tensors2) == set(tensors), "tensor names changed"
for name in tensors:
    assert tensors2[name].equal(tensors[name]), f"tensor {name} changed"
with safetensors.safe_open(out, framework="pt") as f:
    meta2 = f.metadata()
assert meta2.get("version") == meta.get("version"), f"metadata lost: {meta2}"
print("VERIFIED")
"""


def rewrite_safetensors_with_metadata(
    src_bytes: bytes, metadata: dict[str, str]
) -> bytes:
    """Re-save a safetensors blob with extra header metadata (different bytes).

    Runs in ``solar-host/.venv`` (the only venv with torch+safetensors) as
    a subprocess. The tensors are bit-identical — only the header metadata
    changes — so the artifact identity (sha256) differs while logits stay
    the same. Raises AssertionError if the rewrite does not round-trip.
    """
    import json
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        src_path = Path(td) / "in.safetensors"
        out_path = Path(td) / "out.safetensors"
        src_path.write_bytes(src_bytes)
        proc = subprocess.run(
            [
                str(SOLAR_HOST_PYTHON),
                "-c",
                _SAFETENSORS_SCRIPT,
                str(src_path),
                str(out_path),
                json.dumps(metadata),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=clean_env(),
        )
        if proc.returncode != 0 or "VERIFIED" not in proc.stdout:
            raise AssertionError(
                f"safetensors metadata rewrite failed (rc={proc.returncode}):\n"
                f"stdout: {proc.stdout[:400]}\nstderr: {proc.stderr[:800]}"
            )
        return out_path.read_bytes()
