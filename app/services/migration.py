"""Instance migration orchestrator (S-037).

Migrates an inference instance from a source host to a target host by:
1. Validating both hosts and checking for active training jobs
2. Capturing the instance configuration
3. Checking placement constraints (roles, GPU type, VRAM, disk)
4. Enforcing one-replica-per-host on the target
5. Ensuring the model is on the target via S-019 distribution
6. Stopping the source instance
7. Creating the target instance from the captured configuration

If stop or create fails, the partially-completed MigrationResult is returned
with status="failed" and per-step status so callers can inspect progress.

Active training jobs from S-032/S-033 are non-migratable workloads;
migration is rejected if the source host has any active job steps.

The stop-before-create ordering is explicit and documented. For rolling /
zero-downtime migrations, the S-042 strategy layer will orchestrate
create-then-stop on top of this primitive.
"""

import asyncio
import logging
import uuid
from typing import Any

import aiohttp
from fastapi import HTTPException

from app.database.hosts import host_db
from app.models import Host
from app.models.migration import MigrationResult, MigrationStep
from app.model_resolvers import resolve
from app.redis_state import host_store
from app.validation import validate_priority

logger = logging.getLogger(__name__)

# ── Shared: create an instance on a host (Option B refactor) ────


async def create_instance_on_host(
    host: Host, instance_data: dict[str, Any]
) -> dict[str, Any]:
    """Create an inference instance on *host* with the given config.

    Validates priority (S-036), resolves ``model_source`` (S-019), sets
    the derived ``model``/``model_id`` while preserving the original URI
    for cross-host operations (S-037), and POSTs to the host.
    """
    # Validate priority if present (S-036)
    validate_priority(instance_data)

    # Resolve model_source and set model/model_id while preserving the
    # original URI.  Support both flat and {config: {...}} payload shapes.
    config = instance_data.get("config", instance_data)
    model_source = config.get("model_source")
    if model_source:
        resolved = await resolve(model_source, host.url, host.api_key)
        # Extract filesystem path from local:// URI (scheme is 8 chars)
        if resolved.startswith("local://"):
            model_path = resolved[8:]
        else:
            model_path = resolved
        backend_type = config.get("backend_type", "llamacpp")
        if backend_type.startswith("huggingface"):
            config["model_id"] = model_path
        else:
            config["model"] = model_path
        if "config" in instance_data:
            instance_data["config"] = config

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{host.url}/instances"
            headers = {
                "X-API-Key": host.api_key,
                "Content-Type": "application/json",
            }
            async with session.post(
                url,
                json=instance_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    return await response.json()
                text = await response.text()
                raise HTTPException(status_code=response.status, detail=text)
    except HTTPException:
        raise
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ClientConnectorError,
        asyncio.TimeoutError,
    ):
        raise HTTPException(
            status_code=502,
            detail=f"Host '{host.name}' is unreachable at {host.url}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach host '{host.name}': {e}",
        )


# ── Migration validation helpers ────────────────────────────────


async def capture_instance_config(
    source_host: Host, instance_id: str
) -> dict[str, Any]:
    """Retrieve the full instance configuration from *source_host*.

    Tries Redis cache first (fast path), then falls back to an HTTP
    ``GET /instances`` call to the host.  The Redis cache only
    short-circuits when the entry includes a ``config`` key (i.e. a
    full dump from a prior HTTP call); the flat WebSocket notification
    format omits most config fields and is not used as a shortcut.
    """
    # Fast path: full config in Redis cache?
    instances = await host_store.get_host_instances(source_host.id)
    for inst in instances:
        iid = inst.get("instance_id") or inst.get("id")
        if iid == instance_id and "config" in inst:
            logger.debug(
                "Instance config for %s/%s found in Redis cache",
                source_host.name,
                instance_id,
            )
            return inst

    # Fallback / direct: fetch from source host via HTTP
    logger.info(
        "Instance %s not in Redis cache for host %s, falling back to HTTP",
        instance_id,
        source_host.name,
    )
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"X-API-Key": source_host.api_key}
            url = f"{source_host.url}/instances"
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Failed to fetch instances from source host: {text}",
                    )
                all_instances = await response.json()
                for inst in all_instances:
                    iid = inst.get("instance_id") or inst.get("id")
                    if iid == instance_id:
                        return inst
    except HTTPException:
        raise
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ClientConnectorError,
        asyncio.TimeoutError,
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Source host '{source_host.name}' is unreachable "
                f"at {source_host.url}"
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach source host '{source_host.name}': {e}",
        )

    raise HTTPException(
        status_code=404,
        detail=(
            f"Instance '{instance_id}' not found on source host "
            f"'{source_host.name}'"
        ),
    )


async def check_one_replica_per_host(target_host: Host, alias: str) -> None:
    """Ensure no instance with the same *alias* exists on *target_host*.

    Raises ``HTTPException(409)`` if a conflict is found.
    """
    instances = await host_store.get_host_instances(target_host.id)
    for inst in instances:
        config = inst.get("config", inst)
        inst_alias = config.get("alias") or inst.get("alias")
        if inst_alias == alias:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Instance with alias '{alias}' already exists on "
                    f"target host '{target_host.name}'. One replica per "
                    f"host per alias is enforced."
                ),
            )


async def validate_target_fitness(
    target_host: Host,
    instance_config: dict[str, Any],
    *,
    allow_production: bool = False,
    source_gpu_type: str | None = None,
) -> None:
    """Validate that *target_host* is suitable for *instance_config*.

    Checks:
    - Target has ``"inference"`` role
    - Production safeguard (requires explicit ``allow_production``)
    - GPU type difference is logged but not blocked (GGUF models are
      portable across architectures and pulled fresh via S-019)\n    - Sufficient VRAM (best-effort, ≥ 2 GB threshold)
    - Sufficient disk space (best-effort, ≥ 5 GB threshold)
    """
    # Role check
    roles = target_host.roles or []
    if "inference" not in roles:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Target host '{target_host.name}' does not have the "
                f"'inference' role. Roles: {roles}"
            ),
        )

    # Production safeguard
    config = instance_config.get("config", instance_config)
    priority = config.get("priority") or instance_config.get("priority")
    if priority == "production" and not allow_production:
        raise HTTPException(
            status_code=422,
            detail=(
                "Cannot migrate a 'production' instance without "
                "explicit allow_production=true. Production instances "
                "require an explicit policy decision to migrate."
            ),
        )

    # GPU type difference is logged but not rejected. GGUF models are
    # portable across GPU architectures and the model is pulled fresh
    # on the target host via S-019 distribution.  Placement constraints
    # (deployment-intent §4.5) default gpu_type to null (any).
    if source_gpu_type and target_host.gpu_type:
        if source_gpu_type != target_host.gpu_type:
            logger.info(
                "GPU type differs — source '%s' (%s) → target '%s' (%s)",
                source_gpu_type,
                source_gpu_type,
                target_host.name,
                target_host.gpu_type,
            )

    # Resource check via /health (disk + VRAM)
    MIN_VRAM_GB = 2.0
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{target_host.url.rstrip('/')}/health"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    # Disk check
                    available = data.get("disk", {}).get("available_gb")
                    if available is not None and available < 5.0:
                        raise HTTPException(
                            status_code=507,
                            detail=(
                                f"Insufficient disk on target host "
                                f"'{target_host.name}': "
                                f"{available:.2f} GB available"
                            ),
                        )

                    # VRAM check
                    memory_list = data.get("memory", [])
                    for mem in memory_list:
                        if mem.get("memory_type") == "VRAM":
                            vram_available = mem.get("available_gb")
                            if (
                                vram_available is not None
                                and vram_available < MIN_VRAM_GB
                            ):
                                raise HTTPException(
                                    status_code=507,
                                    detail=(
                                        f"Insufficient VRAM on target host "
                                        f"'{target_host.name}': "
                                        f"{vram_available:.2f} GB available "
                                        f"(minimum {MIN_VRAM_GB} GB required)"
                                    ),
                                )
                            break
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(
            "Failed to check resources on target host %s: %s",
            target_host.id,
            e,
        )


async def ensure_model_on_target(
    target_host: Host, model_source: str
) -> tuple[str, bool]:
    """Ensure *model_source* is pulled on *target_host* via S-019.

    Returns ``(local_path, cached)`` on success.
    Raises ``HTTPException`` on failure.
    """
    from app.model_resolvers.parser import parse
    from app.routes.management.models import _pull_on_host

    parsed = parse(model_source)
    result = await _pull_on_host(parsed, model_source, target_host)

    from app.routes.management.models import _StructuredPullError

    if isinstance(result, _StructuredPullError):
        raise HTTPException(
            status_code=result.status_code,
            detail=(
                f"Failed to pull model '{result.source_uri}' on target "
                f"host '{target_host.name}': [{result.error}] "
                f"{result.detail}"
            ),
        )

    return result  # (path, cached)


async def check_no_active_training(host: Host) -> None:
    """Verify *host* has no active training job steps.

    Queries the host's ``GET /jobs`` endpoint. Raises ``HTTPException(409)``
    if any job step is in an active (non-terminal) state.

    This implements the S-037 requirement that active training jobs from
    S-032/S-033 are non-migratable workloads.
    """
    TERMINAL_STATES = {"completed", "failed", "cancelled", "error"}

    try:
        async with aiohttp.ClientSession() as session:
            url = f"{host.url.rstrip('/')}/jobs"
            headers = {"X-API-Key": host.api_key}
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status == 200:
                    jobs = await response.json()
                elif response.status == 404:
                    logger.debug(
                        "Host '%s' returned 404 for GET /jobs, "
                        "assuming no training jobs",
                        host.name,
                    )
                    return
                else:
                    text = await response.text()
                    logger.warning(
                        "Host '%s' returned %d for GET /jobs: %s",
                        host.name,
                        response.status,
                        text,
                    )
                    return

        active_ids: list[str] = []
        for job in (jobs if isinstance(jobs, list) else []):
            status = job.get("status") or job.get("state", "")
            if status not in TERMINAL_STATES:
                job_id = job.get("job_id") or job.get("id", "unknown")
                active_ids.append(job_id)

        if active_ids:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Source host '{host.name}' has {len(active_ids)} "
                    f"active training job(s): {', '.join(active_ids[:5])}"
                    f"{'...' if len(active_ids) > 5 else ''}. "
                    f"Training jobs are non-migratable workloads. "
                    f"Stop or wait for training jobs to complete before "
                    f"migrating instances from this host."
                ),
            )
    except HTTPException:
        raise
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ClientConnectorError,
        asyncio.TimeoutError,
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Source host '{host.name}' is unreachable for training job "
                f"check at {host.url}. Cannot verify no active training jobs "
                f"– migration rejected."
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Cannot reach source host '{host.name}' for training job "
                f"check: {e}. Migration rejected."
            ),
        )


async def stop_source_instance(source_host: Host, instance_id: str) -> dict[str, Any]:
    """Stop *instance_id* on *source_host*.

    Returns the host response on success. Raises ``HTTPException`` on
    failure.
    """
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{source_host.url}/instances/{instance_id}/stop"
            headers = {
                "X-API-Key": source_host.api_key,
                "Content-Type": "application/json",
            }
            async with session.post(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    return await response.json()
                text = await response.text()
                raise HTTPException(status_code=response.status, detail=text)
    except HTTPException:
        raise
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ClientConnectorError,
        asyncio.TimeoutError,
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                f"Source host '{source_host.name}' is unreachable "
                f"at {source_host.url}"
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach source host '{source_host.name}': {e}",
        )


# ── Orchestrator ────────────────────────────────────────────────


def _config_field(instance: dict[str, Any], key: str) -> Any:
    """Read a field from the instance dict, checking nested config first."""
    config = instance.get("config", {})
    if isinstance(config, dict) and key in config:
        return config[key]
    return instance.get(key)


def _build_result(
    migration_id: str,
    source_host: Host,
    target_host: Host,
    source_instance_id: str,
    alias: str,
    model_source: str,
    priority: str,
    target_instance_id: str | None,
    steps: list[MigrationStep],
    *,
    status: str = "completed",
    error: str | None = None,
) -> MigrationResult:
    """Build a MigrationResult with consistent field population."""
    return MigrationResult(
        migration_id=migration_id,
        status=status,
        source_host_id=source_host.id,
        source_host_name=source_host.name,
        target_host_id=target_host.id,
        target_host_name=target_host.name,
        source_instance_id=source_instance_id,
        target_instance_id=target_instance_id,
        alias=alias,
        model_source=model_source,
        priority=priority,
        steps=steps,
        error=error,
    )


async def execute_migration(
    *,
    instance_id: str,
    source_host_id: str,
    target_host_id: str,
    allow_production: bool = False,
) -> MigrationResult:
    """Execute a full migration of *instance_id* from source to target host.

    Orchestrates all validation, model distribution, stop, and create
    steps. Returns a ``MigrationResult`` with per-step status on success.
    Raises ``HTTPException`` for fatal errors.
    """
    migration_id = str(uuid.uuid4())
    steps: list[MigrationStep] = []

    # ── 1. Validate hosts ───────────────────────────────────────
    source_host = await host_db.get_host(source_host_id)
    if not source_host:
        raise HTTPException(
            status_code=404,
            detail=f"Source host '{source_host_id}' not found",
        )

    target_host = await host_db.get_host(target_host_id)
    if not target_host:
        raise HTTPException(
            status_code=404,
            detail=f"Target host '{target_host_id}' not found",
        )

    if source_host_id == target_host_id:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Source and target host are the same ('{source_host.name}'). "
                f"Migration requires two distinct hosts."
            ),
        )

    steps.append(MigrationStep(step="validate_hosts", status="ok"))

    # ── 1.5. Check source host has no active training jobs ──────
    await check_no_active_training(source_host)
    steps.append(MigrationStep(step="check_training_jobs", status="ok"))

    # ── 2. Capture instance configuration ───────────────────────
    instance_config = await capture_instance_config(source_host, instance_id)

    alias = _config_field(instance_config, "alias")
    model_source = _config_field(instance_config, "model_source")
    priority = _config_field(instance_config, "priority") or "production"
    source_gpu_type = source_host.gpu_type

    if not alias:
        raise HTTPException(
            status_code=422,
            detail="Instance configuration is missing required 'alias' field",
        )
    if not model_source:
        raise HTTPException(
            status_code=422,
            detail="Instance configuration is missing required 'model_source' field",
        )

    # Validate captured priority before any destructive operations (S-036/S-037).
    # Legacy instances may have invalid priorities that would fail at create_target
    # step (step 7) after the source instance has already been stopped.
    from app.validation import VALID_PRIORITIES

    if priority not in VALID_PRIORITIES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Captured instance has invalid priority '{priority}'. "
                f"Must be one of: {', '.join(sorted(VALID_PRIORITIES))}"
            ),
        )

    steps.append(
        MigrationStep(
            step="capture_config",
            status="ok",
            detail={
                "alias": alias,
                "model_source": model_source,
                "priority": priority,
            },
        )
    )

    # ── 3. Validate target fitness ──────────────────────────────
    await validate_target_fitness(
        target_host,
        instance_config,
        allow_production=allow_production,
        source_gpu_type=source_gpu_type,
    )
    steps.append(MigrationStep(step="validate_target", status="ok"))

    # ── 4. Check one-replica-per-host ───────────────────────────
    await check_one_replica_per_host(target_host, alias)
    steps.append(MigrationStep(step="check_anti_affinity", status="ok"))

    # ── 5. Ensure model on target ───────────────────────────────
    try:
        path, cached = await ensure_model_on_target(target_host, model_source)
        steps.append(
            MigrationStep(
                step="ensure_model",
                status="ok",
                detail={"path": path, "cached": cached},
            )
        )
    except HTTPException as e:
        steps.append(
            MigrationStep(
                step="ensure_model",
                status="failed",
                detail={"error": str(e.detail), "status_code": e.status_code},
            )
        )
        return _build_result(
            migration_id,
            source_host,
            target_host,
            instance_id,
            alias,
            model_source,
            priority,
            None,
            steps,
            status="failed",
            error=f"Ensure model failed: {e.detail}",
        )

    # ── 6. Stop source instance ─────────────────────────────────
    try:
        await stop_source_instance(source_host, instance_id)
        steps.append(MigrationStep(step="stop_source", status="ok"))
    except HTTPException as e:
        steps.append(
            MigrationStep(
                step="stop_source",
                status="failed",
                detail={"error": str(e.detail), "status_code": e.status_code},
            )
        )
        return _build_result(
            migration_id,
            source_host,
            target_host,
            instance_id,
            alias,
            model_source,
            priority,
            None,
            steps,
            status="failed",
            error=f"Stop source failed: {e.detail}",
        )

    # ── 7. Create target instance ───────────────────────────────
    # Build the instance config for the target host.
    config = instance_config.get("config", instance_config)

    # Remove host-assigned and instance-level fields from the config dict.
    _INSTANCE_FIELDS = frozenset(
        {
            "id",
            "status",
            "port",
            "pid",
            "api_key",
            "supported_endpoints",
            "managed_by",
            "intent_id",
            "created_at",
            "started_at",
            "error_message",
            "retry_count",
            "busy",
            "prefill_progress",
            "active_slots",
        }
    )
    create_payload: dict[str, Any] = {
        k: v for k, v in config.items() if k not in _INSTANCE_FIELDS
    }
    # Set model to the path resolved by ensure_model_on_target so the
    # host does not need to resolve model_source itself (which would
    # reject repo:// URIs without the companion host-side fix).
    create_payload["model"] = path
    create_payload.pop("model_source", None)
    # Ensure key fields from captured config are present.
    for key in ("alias", "priority", "backend_type"):
        if key not in create_payload:
            val = _config_field(instance_config, key)
            if val is not None:
                create_payload[key] = val

    target_instance: dict[str, Any]
    try:
        target_instance = await create_instance_on_host(
            target_host, {"config": create_payload}
        )
    except HTTPException as e:
        steps.append(
            MigrationStep(
                step="create_target",
                status="failed",
                detail={"error": str(e.detail), "status_code": e.status_code},
            )
        )
        return _build_result(
            migration_id,
            source_host,
            target_host,
            instance_id,
            alias,
            model_source,
            priority,
            None,
            steps,
            status="failed",
            error=f"Create target failed: {e.detail}",
        )

    target_instance_id = target_instance.get("instance_id") or target_instance.get(
        "id", ""
    )
    steps.append(
        MigrationStep(
            step="create_target",
            status="ok",
            detail={"target_instance_id": target_instance_id},
        )
    )

    logger.info(
        "Migration %s completed: %s/%s (%s) → %s/%s",
        migration_id,
        source_host.name,
        instance_id,
        alias,
        target_host.name,
        target_instance_id,
    )

    return _build_result(
        migration_id,
        source_host,
        target_host,
        instance_id,
        alias,
        model_source,
        priority,
        target_instance_id,
        steps,
    )
