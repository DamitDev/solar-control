"""D-017 cross-service integration suite harness.

Brings up (per session) ephemeral Postgres + Redis via testcontainers, a
TLS stub Harbor speaking OCI Distribution, and Alembic migrations for both
databases; then (once per session) data-repository / solar-control /
solar-host ×2 uvicorn subprocesses wired to each other over loopback. The
wake test (3600s interval) builds its own stack and runs last.

Fixture scopes
--------------
Session:  postgres_container, redis_container, stub_harbor, db_env,
          alembic_data_repo, alembic_solar_control, stub_model_artifact,
          stack (data-repo + control + host A + host B + model registration)
Module:   wake_stack (its own control with RECONCILE_INTERVAL_S=3600)
Function: clean_state, http_data_repo, http_control, http_host, http_host_b

The WS seam: hosts connect to control's ``/ws/host-channel`` at startup and
push ``host_health`` + ``instances_update``. Control writes these into
Redis ``host_store`` — exactly what the reconciler observes. Every module
gates on both hosts being visible (``hosts_online``) before tests run, and
tests assert through control's API, never by poking host state — except
where the test is about the host itself.

Subprocess envs are built from a *clean* environment: the Hermes desktop
app leaks ``PYTHONPATH`` into its own venv site-packages, which corrupts
service imports (mismatched pydantic_core etc.). See helpers.clean_env().
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Pre-collection env pins. Nothing in this package imports app.* at module
# level, but keep placeholders so accidental imports can't crash collection.
os.environ.setdefault("DATABASE_URL", "postgresql://x:x@127.0.0.1:1/none")
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:1/0")

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

from fixtures.constants import (  # noqa: E402
    DATA_REPOSITORY_API_KEY,
    FIXTURE_MODEL_DIR,
    HARBOR_PASSWORD,
    HARBOR_USERNAME,
    HOST_A_API_KEY,
    HOST_B_API_KEY,
    MANAGEMENT_API_KEY,
    MODEL_NAME,
    MODEL_VERSION,
)
from fixtures.helpers import (  # noqa: E402
    DATA_REPO_PYTHON,
    REPO_ROOT,
    SOLAR_CONTROL_PYTHON,
    SOLAR_HOST_PYTHON,
    build_subprocess_env,
    free_port,
    make_certs,
    spawn_service,
    tail_service_logs,
    wait_for,
)
from fixtures.seed import (  # noqa: E402
    register_host_via_api,
    register_model_in_data_repo,
    read_test_model_files,
    truncate_intents,
)
from fixtures.stub_harbor import StubHarbor  # noqa: E402

logger = logging.getLogger(__name__)

# ── Suite constants ────────────────────────────────────────────────

# Markers applied automatically by folder (orchestrator pattern).
MARKER_FOLDERS = {
    "repo_path": "repo_path",
    "intent_path": "intent_path",
    "migration_path": "migration_path",
    "infrastructure": "infrastructure",
}


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    for item in items:
        for marker, folder in MARKER_FOLDERS.items():
            if folder in item.path.parts:
                item.add_marker(getattr(pytest.mark, marker))
                break
    # The wake-stack module must run LAST: its stack build truncates the
    # shared hosts table and it stops the session control, so any module
    # after it would have no hosts to reconcile against.
    items.sort(key=lambda item: item.path.name == "test_reconciler_wake.py")


# ── Session scope ──────────────────────────────────────────────────


@pytest.fixture(scope="session")
def postgres_container():
    from testcontainers.community.postgres import PostgresContainer

    container = PostgresContainer("postgres:15-alpine")
    container.start()
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        user, password = "test", "test"
        import psycopg2

        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname="test"
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE DATABASE data_repository")
            cur.execute("CREATE DATABASE solar_gateway")
        conn.close()
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def redis_container():
    from testcontainers.community.redis import RedisContainer

    container = RedisContainer("redis:7-alpine")
    container.start()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def stub_harbor(tmp_path_factory: pytest.TempPathFactory):
    """TLS OCI Distribution stub on a random loopback port."""
    cert_dir = tmp_path_factory.mktemp("harbor-certs")
    ca_crt, srv_crt, srv_key = make_certs(cert_dir)
    stub = StubHarbor()
    base_url = stub.start(tls=(str(srv_crt), str(srv_key)))
    stub.ca_cert_path = str(ca_crt)  # type: ignore[attr-defined]
    stub.state.log_file = str(cert_dir / "harbor-requests.log")
    logger.info("stub harbor (TLS) at %s", base_url)
    try:
        yield stub
    finally:
        stub.stop()


@pytest.fixture(scope="session")
def db_env(postgres_container, redis_container):
    """Session-wide environment + connection URLs shared by all services."""
    pg_host = postgres_container.get_container_host_ip()
    pg_port = postgres_container.get_exposed_port(5432)
    redis_host = redis_container.get_container_host_ip()
    redis_port = redis_container.get_exposed_port(6379)

    pg_user, pg_password = "test", "test"
    urls = {
        "pg_host": pg_host,
        "pg_port": pg_port,
        "pg_user": pg_user,
        "pg_password": pg_password,
        "pg_base": f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}",
        "control_db": (
            f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/solar_gateway"
        ),
        "data_repo_db": (
            f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/data_repository"
        ),
        "redis": f"redis://{redis_host}:{redis_port}/0",
    }

    saved = {
        key: os.environ.get(key)
        for key in (
            "DATABASE_URL",
            "REDIS_URL",
            "DATA_REPOSITORY_URL",
            "DATA_REPOSITORY_API_KEY",
            "MANAGEMENT_API_KEY",
            "RECONCILE_INTERVAL_S",
            "RECONCILE_HEALTH_GATE_TIMEOUT_S",
        )
    }
    os.environ["DATABASE_URL"] = urls["control_db"]
    os.environ["REDIS_URL"] = urls["redis"]
    os.environ["DATA_REPOSITORY_API_KEY"] = DATA_REPOSITORY_API_KEY
    os.environ["MANAGEMENT_API_KEY"] = MANAGEMENT_API_KEY
    os.environ["RECONCILE_INTERVAL_S"] = "0.5"
    os.environ["RECONCILE_HEALTH_GATE_TIMEOUT_S"] = "5"
    try:
        yield urls
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_alembic(python: Path, cwd: Path, env: dict[str, str]) -> None:
    import subprocess

    result = subprocess.run(
        [str(python), "-m", "alembic", "upgrade", "head"],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert (
        result.returncode == 0
    ), f"alembic upgrade failed in {cwd}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    logger.info("alembic upgrade head OK in %s", cwd.name)


@pytest.fixture(scope="session")
def alembic_data_repo(db_env: dict[str, str], stub_harbor):
    env = build_subprocess_env(
        POSTGRES_HOST=db_env["pg_host"],
        POSTGRES_PORT=db_env["pg_port"],
        POSTGRES_DB="data_repository",
        POSTGRES_USER=db_env["pg_user"],
        POSTGRES_PASSWORD=db_env["pg_password"],
    )
    _run_alembic(DATA_REPO_PYTHON, Path(DATA_REPO_PYTHON).parent.parent.parent, env)


@pytest.fixture(scope="session")
def alembic_solar_control(db_env: dict[str, str]):
    env = build_subprocess_env(DATABASE_URL=db_env["control_db"])
    _run_alembic(SOLAR_CONTROL_PYTHON, REPO_ROOT, env)


@pytest.fixture(scope="session")
def stub_model_artifact(stub_harbor):
    """Register the committed HF model in stub Harbor once per session.

    The Harbor ref embeds the stub's dynamic port, so it is computed here
    and exposed as ``(harbor_ref, files)``.
    """
    files = read_test_model_files(FIXTURE_MODEL_DIR)
    host_port = stub_harbor.base_url.split(":")[-1]
    harbor_ref = f"127.0.0.1:{host_port}/supernova/{MODEL_NAME}:{MODEL_VERSION}"
    stub_harbor.register_model(harbor_ref, files)
    logger.info("registered fixture artifact %s (%d files)", harbor_ref, len(files))
    return harbor_ref, files


# ── Module scope: the service stack ───────────────────────────────


@dataclass
class Stack:
    """One full service topology (data-repo + control + 2 hosts)."""

    db_env: dict[str, str]
    stub_harbor: StubHarbor
    harbor_ref: str

    data_repo: Any = None
    control: Any = None
    host_a: Any = None
    host_b: Any = None

    data_repo_url: str = ""
    control_url: str = ""
    host_a_url: str = ""
    host_b_url: str = ""

    models_dir_a: Path = field(default_factory=Path)
    models_dir_b: Path = field(default_factory=Path)
    logs_dir: Path = field(default_factory=Path)
    tmp_dir: Path = field(default_factory=Path)

    control_env: dict[str, str] = field(default_factory=dict)
    data_repo_env: dict[str, str] = field(default_factory=dict)
    host_a_env: dict[str, str] = field(default_factory=dict)
    host_b_env: dict[str, str] = field(default_factory=dict)

    extra_hosts: dict[str, Any] = field(default_factory=dict)
    extra_host_urls: dict[str, str] = field(default_factory=dict)

    reconcile_interval_s: float = 0.5

    def host_url(self, letter: str) -> str:
        if letter == "a":
            return self.host_a_url
        if letter == "b":
            return self.host_b_url
        return self.extra_host_urls.get(letter, "")

    def models_dir(self, letter: str) -> Path:
        if letter == "a":
            return self.models_dir_a
        if letter == "b":
            return self.models_dir_b
        return self.tmp_dir / f"models-{letter}"

    def service(self, letter: str) -> Any:
        if letter == "a":
            return self.host_a
        if letter == "b":
            return self.host_b
        return self.extra_hosts.get(letter)

    async def spawn_extra_host(self, letter: str) -> str:
        """Spawn an additional host subprocess and register it in control.

        Used by tests that need more hosts than the default two (e.g. the
        shortfall test: 3 replicas on 2 hosts -> 3rd host fills to ready).
        Returns the host URL.
        """
        if letter in self.extra_hosts:
            return self.extra_host_urls[letter]
        api_key = f"test-host-{letter}-key"
        port = free_port()
        async with _control_client(self.control_url) as client:
            host_id = await register_host_via_api(
                client,
                f"host-{letter}",
                f"http://127.0.0.1:{port}",
                api_key,
                roles=["inference"],
            )
        from fixtures.seed import update_host_roles

        update_host_roles(self.db_env["control_db"], host_id, ["inference"])
        env = build_subprocess_env(
            dict(self.host_b_env),
            API_KEY=api_key,
            PORT=str(port),
            CONFIG_FILE=str(self.tmp_dir / f"config-{letter}.json"),
            LOG_DIR=str(self.tmp_dir / f"logs-{letter}"),
            MODELS_DIR=str(self.tmp_dir / f"models-{letter}"),
            START_PORT=str(35300 + (ord(letter) - ord("a")) * 100),
        )
        svc, actual_port = spawn_service(
            name=f"host-{letter}",
            python=SOLAR_HOST_PYTHON,
            module="solar_host.main:app",
            port=port,
            cwd=Path(SOLAR_HOST_PYTHON).parent.parent.parent,
            env=env,
            log_dir=self.logs_dir,
            ready_path="/health",
        )
        self.extra_hosts[letter] = svc
        url = f"http://127.0.0.1:{actual_port}"
        self.extra_host_urls[letter] = url
        if actual_port != port:
            import psycopg2

            conn = psycopg2.connect(self.db_env["control_db"])
            conn.autocommit = True
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE hosts SET url = %s WHERE name = %s",
                        (url, f"host-{letter}"),
                    )
            finally:
                conn.close()
        await _wait_host_online(self, f"host-{letter}")
        return url

    def remove_extra_host(self, letter: str) -> None:
        """Terminate an extra host and drop its DB row.

        Used to restore the 2-host topology after the shortfall test: the
        migration tests that follow assume exactly two hosts (their
        "no target" displacement scenario must have no third host for the
        MIGRATE target search to find).
        """
        svc = self.extra_hosts.pop(letter, None)
        if svc is not None:
            svc.terminate()
        self.extra_host_urls.pop(letter, None)
        import psycopg2

        conn = psycopg2.connect(self.db_env["control_db"])
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM hosts WHERE name = %s", (f"host-{letter}",))
        finally:
            conn.close()

    async def respawn_data_repo(self) -> str:
        """Kill and re-spawn the data-repository subprocess (same port/env).

        Used by the failed-create-backoff test: with data-repo down, the
        reconciler's CREATE resolve step fails deterministically; respawning
        lets the next tick recover.
        """
        if self.data_repo is not None:
            self.data_repo.terminate()
        svc, actual_port = spawn_service(
            name="data-repo",
            python=DATA_REPO_PYTHON,
            module="app.main:app",
            port=self.data_repo.port if self.data_repo else free_port(),
            cwd=Path(DATA_REPO_PYTHON).parent.parent.parent,
            env=self.data_repo_env,
            log_dir=self.logs_dir,
            ready_path="/health",
        )
        self.data_repo = svc
        self.data_repo_url = f"http://127.0.0.1:{actual_port}"
        return self.data_repo_url

    def tail(self) -> str:
        return tail_service_logs(
            [self.data_repo, self.control, self.host_a, self.host_b]
            + list(self.extra_hosts.values())
        )


async def _build_stack(
    db_env: dict[str, str],
    stub_harbor: StubHarbor,
    harbor_ref: str,
    *,
    tmp_root: Path,
    reconcile_interval_s: float = 0.5,
) -> Stack:
    """Spawn data-repo, control (with 2 pre-registered hosts), and 2 hosts."""
    stack = Stack(
        db_env=db_env,
        stub_harbor=stub_harbor,
        harbor_ref=harbor_ref,
        tmp_dir=tmp_root,
        logs_dir=tmp_root / "logs",
        models_dir_a=tmp_root / "models-a",
        models_dir_b=tmp_root / "models-b",
        reconcile_interval_s=reconcile_interval_s,
    )
    logs_dir = stack.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    base = {
        "SSL_CERT_FILE": stub_harbor.ca_cert_path,
        "REQUESTS_CA_BUNDLE": stub_harbor.ca_cert_path,
        "HF_HOME": str(tmp_root / "hf-cache"),
        "TOKENIZERS_PARALLELISM": "false",
    }

    # ── data-repository ──
    pg_host = db_env["pg_base"].split("@")[1].split(":")[0]
    pg_port = db_env["pg_base"].split(":")[-1]
    pg_user = db_env["pg_base"].split("//")[1].split(":")[0]
    pg_password = db_env["pg_base"].split(":")[2].split("@")[0]
    data_repo_env = build_subprocess_env(
        base,
        POSTGRES_HOST=pg_host,
        POSTGRES_PORT=pg_port,
        POSTGRES_DB="data_repository",
        POSTGRES_USER=pg_user,
        POSTGRES_PASSWORD=pg_password,
        HARBOR_URL=stub_harbor.base_url,
        HARBOR_USERNAME=HARBOR_USERNAME,
        HARBOR_PASSWORD=HARBOR_PASSWORD,
        LOG_LEVEL="INFO",
    )
    data_repo, data_repo_port = spawn_service(
        name="data-repo",
        python=DATA_REPO_PYTHON,
        module="app.main:app",
        port=free_port(),
        cwd=Path(DATA_REPO_PYTHON).parent.parent.parent,
        env=data_repo_env,
        log_dir=logs_dir,
        ready_path="/health",
    )
    stack.data_repo = data_repo
    stack.data_repo_env = data_repo_env
    stack.data_repo_url = f"http://127.0.0.1:{data_repo_port}"

    # ── solar-control ──
    control_env = build_subprocess_env(
        base,
        DATABASE_URL=db_env["control_db"],
        REDIS_URL=db_env["redis"],
        DATA_REPOSITORY_URL=stack.data_repo_url,
        DATA_REPOSITORY_API_KEY=DATA_REPOSITORY_API_KEY,
        MANAGEMENT_API_KEY=MANAGEMENT_API_KEY,
        RECONCILE_INTERVAL_S=str(reconcile_interval_s),
        RECONCILE_HEALTH_GATE_TIMEOUT_S="5",
        # Fast pacing for the test env: the settle/cooldown windows exist to
        # absorb WS-push latency (~ms here), and the registry refresh gates
        # the ready alias — 1s keeps convergence snappy without racing.
        RECONCILE_SETTLE_S="1.0",
        RECONCILE_MIGRATE_SETTLE_S="3.0",
        RECONCILE_DISPLACE_COOLDOWN_S="20.0",
        REGISTRY_REFRESH_INTERVAL_S="1.0",
        LOG_LEVEL=os.environ.get("TEST_LOG_LEVEL", "INFO"),
    )
    stack.control_env = control_env
    control, control_port = spawn_service(
        name="control",
        python=SOLAR_CONTROL_PYTHON,
        module="app.main:sio_asgi_app",
        port=free_port(),
        cwd=REPO_ROOT,
        env=control_env,
        log_dir=logs_dir,
        ready_path="/ready",
    )
    stack.control = control
    stack.control_url = f"http://127.0.0.1:{control_port}"

    # ── hosts ──
    host_common = {
        "HOST": "127.0.0.1",
        "SOLAR_CONTROL_URL": f"ws://127.0.0.1:{control_port}/ws/host-channel",
        "HARBOR_URL": stub_harbor.base_url,
        "HARBOR_USERNAME": HARBOR_USERNAME,
        "HARBOR_PASSWORD": HARBOR_PASSWORD,
        "HF_HOME": str(tmp_root / "hf-cache"),
        "TOKENIZERS_PARALLELISM": "false",
        "LOG_LEVEL": "INFO",
    }
    stack.host_a_env = build_subprocess_env(
        base,
        **host_common,
        API_KEY=HOST_A_API_KEY,
        PORT="8001",
        CONFIG_FILE=str(tmp_root / "config-a.json"),
        LOG_DIR=str(tmp_root / "logs-a"),
        MODELS_DIR=str(stack.models_dir_a),
        JOBS_DIR=str(tmp_root / "jobs-a"),
        START_PORT="35100",
    )
    stack.host_b_env = build_subprocess_env(
        base,
        **host_common,
        API_KEY=HOST_B_API_KEY,
        PORT="8002",
        CONFIG_FILE=str(tmp_root / "config-b.json"),
        LOG_DIR=str(tmp_root / "logs-b"),
        MODELS_DIR=str(stack.models_dir_b),
        JOBS_DIR=str(tmp_root / "jobs-b"),
        START_PORT="35200",
    )

    # Register host rows BEFORE the hosts connect (WS auth matches api_key).
    # The hosts table is session-persistent while stacks are per-module:
    # drop the previous module's rows so name lookups can't resolve to
    # dead hosts (the tests pick host rows by name).
    import psycopg2

    _conn = psycopg2.connect(db_env["control_db"])
    _conn.autocommit = True
    try:
        with _conn.cursor() as cur:
            cur.execute("TRUNCATE hosts, jobs RESTART IDENTITY")
    finally:
        _conn.close()

    host_a_port = free_port()
    host_b_port = free_port()
    async with _control_client(stack.control_url) as http_control:
        host_a_id = await register_host_via_api(
            http_control,
            "host-a",
            f"http://127.0.0.1:{host_a_port}",
            HOST_A_API_KEY,
            roles=["inference"],
        )
        host_b_id = await register_host_via_api(
            http_control,
            "host-b",
            f"http://127.0.0.1:{host_b_port}",
            HOST_B_API_KEY,
            roles=["inference"],
        )

    # The WS registration event does not reliably persist roles in the test
    # environment (see README platform findings), so seed them directly.
    from fixtures.seed import update_host_roles

    update_host_roles(stack.db_env["control_db"], host_a_id, ["inference"])
    update_host_roles(stack.db_env["control_db"], host_b_id, ["inference"])

    stack.host_a, actual_a_port = spawn_service(
        name="host-a",
        python=SOLAR_HOST_PYTHON,
        module="solar_host.main:app",
        port=host_a_port,
        cwd=Path(SOLAR_HOST_PYTHON).parent.parent.parent,
        env=stack.host_a_env,
        log_dir=logs_dir,
        ready_path="/health",
    )
    stack.host_a_url = f"http://127.0.0.1:{actual_a_port}"

    stack.host_b, actual_b_port = spawn_service(
        name="host-b",
        python=SOLAR_HOST_PYTHON,
        module="solar_host.main:app",
        port=host_b_port,
        cwd=Path(SOLAR_HOST_PYTHON).parent.parent.parent,
        env=stack.host_b_env,
        log_dir=logs_dir,
        ready_path="/health",
    )
    stack.host_b_url = f"http://127.0.0.1:{actual_b_port}"

    # If a spawn retry moved a host to a different port, fix the DB row.
    if actual_a_port != host_a_port or actual_b_port != host_b_port:
        await _update_host_urls(stack)

    # Gate: reconciler is blind until both hosts are connected + registered.
    await _wait_hosts_online(stack)
    logger.info(
        "stack ready: %s / %s / %s / %s",
        stack.data_repo_url,
        stack.control_url,
        stack.host_a_url,
        stack.host_b_url,
    )
    return stack


def _placeholder_url() -> str:
    return f"http://127.0.0.1:{free_port()}"


async def _update_host_urls(stack: Stack) -> None:
    """Patch the registered host rows with the real host URLs (direct SQL)."""
    import psycopg2

    conn = psycopg2.connect(stack.db_env["control_db"])
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE hosts SET url = %s WHERE name = %s",
                (stack.host_a_url, "host-a"),
            )
            cur.execute(
                "UPDATE hosts SET url = %s WHERE name = %s",
                (stack.host_b_url, "host-b"),
            )
    finally:
        conn.close()


async def _wait_host_online(stack: Stack, name: str, timeout: float = 40.0) -> None:
    import httpx

    async with httpx.AsyncClient(
        base_url=stack.control_url, headers={"X-API-Key": MANAGEMENT_API_KEY}
    ) as client:

        async def online() -> bool:
            resp = await client.get("/api/hosts")
            if resp.status_code != 200:
                return False
            rows = {row["name"]: row for row in resp.json()}
            return rows.get(name, {}).get("status") == "online"

        try:
            await wait_for(
                online,
                timeout=timeout,
                interval=0.25,
                description=f"host {name} online in control",
            )
        except AssertionError:
            raise AssertionError(
                f"host {name} never came online in control\n" + stack.tail()
            ) from None


async def _wait_hosts_online(stack: Stack, timeout: float = 40.0) -> None:
    await _wait_host_online(stack, "host-a", timeout=timeout)
    await _wait_host_online(stack, "host-b", timeout=timeout)


async def _ensure_model_registered(stack: Stack) -> None:
    """Idempotently register the fixture model in data-repo (via its API)."""
    import httpx

    async with httpx.AsyncClient(base_url=stack.data_repo_url, timeout=10.0) as client:
        resp = await client.get(f"/api/models/{MODEL_NAME}/versions")
        if resp.status_code == 200:
            versions = resp.json().get("versions", [])
            if any(v["version"] == MODEL_VERSION for v in versions):
                return
        await register_model_in_data_repo(
            client,
            name=MODEL_NAME,
            harbor_ref=stack.harbor_ref,
            version=MODEL_VERSION,
        )


def _control_client(base_url: str):
    """Return an httpx.AsyncClient for control's management API."""
    import httpx

    return httpx.AsyncClient(
        base_url=base_url,
        headers={"X-API-Key": MANAGEMENT_API_KEY},
        timeout=15.0,
    )


@pytest.fixture(scope="session")
def stack(
    db_env: dict[str, str],
    stub_harbor,
    stub_model_artifact,
    alembic_data_repo,
    alembic_solar_control,
    tmp_path_factory: pytest.TempPathFactory,
) -> Any:
    """One full stack for the whole session (previously per module).

    Per-module stacks cost ~10-13s each (16 sequential 4-process spawns with
    readiness gates) — the dominant cost of the suite. A single session
    stack is safe because ``clean_state`` already resets per-test state
    (instances, intents, volatile Redis) and no test kills a host; the only
    module needing its own stack is the wake test (3600s interval), which is
    reordered to run last and stops this stack's control first.

    Deliberately a *sync* fixture: pytest-asyncio's loop-scope machinery
    re-creates async generator fixtures per test event loop (and then tears
    them down on a closed loop), which spawned one full stack per test.
    ``asyncio.run`` wraps the one-shot async setup instead.
    """
    tmp_root = tmp_path_factory.mktemp("stack")
    harbor_ref, _files = stub_model_artifact
    stack = asyncio.run(
        _build_stack(db_env, stub_harbor, harbor_ref, tmp_root=tmp_root)
    )
    try:
        asyncio.run(_ensure_model_registered(stack))
        yield stack
    finally:
        for svc in list(stack.extra_hosts.values()) + [
            stack.host_b,
            stack.host_a,
            stack.control,
            stack.data_repo,
        ]:
            if svc is not None:
                svc.terminate()


# ── Function scope ─────────────────────────────────────────────────


@pytest_asyncio.fixture
async def http_data_repo(stack: Stack):
    import httpx

    async with httpx.AsyncClient(base_url=stack.data_repo_url, timeout=10.0) as client:
        yield client


@pytest_asyncio.fixture
async def http_control(stack: Stack):
    async with _control_client(stack.control_url) as client:
        yield client


@pytest_asyncio.fixture
async def http_host(stack: Stack):
    """Client for host A (direct host API — only where the test is about the host)."""
    import httpx

    async with httpx.AsyncClient(
        base_url=stack.host_a_url,
        headers={"X-API-Key": HOST_A_API_KEY},
        timeout=15.0,
    ) as client:
        yield client


@pytest_asyncio.fixture
async def http_host_b(stack: Stack):
    import httpx

    async with httpx.AsyncClient(
        base_url=stack.host_b_url,
        headers={"X-API-Key": HOST_B_API_KEY},
        timeout=15.0,
    ) as client:
        yield client


async def _delete_all_instances(stack: Stack) -> None:
    """Stop + delete every instance on all hosts (direct host API)."""
    import httpx

    targets: list[tuple[str, str]] = [
        (stack.host_a_url, HOST_A_API_KEY),
        (stack.host_b_url, HOST_B_API_KEY),
    ]
    # Extra hosts (e.g. host-c from the shortfall test) persist for the rest
    # of the session — clean them too, or their instances leak across tests.
    for letter, url in stack.extra_host_urls.items():
        targets.append((url, f"test-host-{letter}-key"))
    for url, api_key in targets:
        async with httpx.AsyncClient(
            base_url=url, headers={"X-API-Key": api_key}, timeout=15.0
        ) as client:
            resp = await client.get("/instances")
            if resp.status_code != 200:
                continue
            for inst in resp.json():
                iid = inst["id"]
                await client.post(f"/instances/{iid}/stop")
                await client.delete(f"/instances/{iid}")


async def _flush_volatile_redis(redis_url: str) -> None:
    """Flush registry/health/routing/reconcile keys — keep solar:hosts:*."""
    import redis.asyncio as aioredis

    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        prefixes = (
            "solar:registry",
            "solar:health:",
            "solar:active:",
            "solar:weight:",
            "solar:rr:",
            "solar:reconcile:lock:",
            "solar:endpoint_cache:",
        )
        for prefix in prefixes:
            async for key in r.scan_iter(match=f"{prefix}*"):
                await r.delete(key)
    finally:
        await r.aclose()


def _wipe_model_caches(stack: Stack) -> None:
    """Wipe every host's model cache (MODELS_DIR/manifest.json + artifacts).

    Hosts are session-scoped now, so their caches persist across tests;
    several tests assert cold-cache behavior (first pull / pull counts).
    Instances must already be deleted first — running model servers may
    hold the files open.
    """
    import shutil

    for letter in ("a", "b") + tuple(stack.extra_host_urls.keys()):
        models_dir = stack.models_dir(letter)
        if not models_dir.exists():
            continue
        for child in models_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


@pytest_asyncio.fixture
async def clean_state(stack: Stack):
    """Per-test slate: no intents, no instances, no model caches, Redis flushed.

    Intents are truncated BEFORE instances are deleted: the session-scoped
    reconciler reacts to instance deletions within one tick (0.5s) and would
    otherwise see a shortfall for the just-deleted intent and start a CREATE
    pull that races the cache wipe (re-creating the model dir after it).
    """
    truncate_intents(stack.db_env["control_db"])
    await _delete_all_instances(stack)
    _wipe_model_caches(stack)
    await _flush_volatile_redis(stack.db_env["redis"])
    stack.stub_harbor.reset()
    yield
    # Teardown: leave nothing behind for the next test.
    truncate_intents(stack.db_env["control_db"])
    await _delete_all_instances(stack)
    _wipe_model_caches(stack)
    await _flush_volatile_redis(stack.db_env["redis"])
    stack.stub_harbor.reset()
