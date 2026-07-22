"""Job step execution API routes (under /api/jobs).

Provides:
- ``POST /api/jobs`` — Submit a job step (select host, proxy, return result)
- ``GET /api/jobs`` — List jobs (paginated)
- ``GET /api/jobs/{id}`` — Get job status (DB + optional host proxy)
- ``DELETE /api/jobs/{id}`` — Cancel a job (proxy to host + DB update)

Solar Control translates the SuperNova-level job intent into a host-level
``JobDefinition`` (per S-021 workspace spec and the S-027 host job API),
proxies it to the selected host's ``POST /jobs``, and tracks the job in the
``jobs`` table. The Solar Control ``job_id`` is sent to the host as the
``JobDefinition.job_id`` so both sides agree on the identifier — this keeps
``GET``/``DELETE`` proxying and S-025/S-026 event correlation consistent.
"""

import logging
import uuid
from typing import Any

import aiohttp
from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.database.hosts import host_db
from app.database.jobs import job_db
from app.jobs.client import JobHostClientError, job_client
from app.jobs.host_selector import select_host
from app.models import Host
from app.models.job import (
    Job,
    JobCreate,
    JobResponse,
    JobStatus,
    JobStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Steps that only prepare inputs (download/populate the workspace) and never
# require a GPU. Used to default ``is_preparation_step`` when the intent does
# not specify it.
_PREPARATION_STEPS = frozenset({"download_model", "download_dataset"})


# ── Helpers ──────────────────────────────────────────────────


async def _get_host_or_404(host_id: str) -> Host:
    """Look up a host by ID or raise 404."""
    host = await host_db.get_host(host_id)
    if not host:
        raise HTTPException(
            status_code=404,
            detail=f"Host '{host_id}' not found",
        )
    return host


async def _get_job_or_404(job_id: str) -> Job:
    """Look up a job by ID or raise 404."""
    job = await job_db.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found",
        )
    return job


def _resolve_train_input(
    steps_config: dict[str, Any],
    step_name: str,
    field: str,
    fallback: str,
) -> str:
    """Resolve a downstream step input from an upstream step's config.

    Looks up ``steps_config[step_name][field]`` and returns it if present,
    otherwise returns ``fallback``. This lets downstream steps (e.g. ``train``)
    derive their input paths from upstream steps (e.g. ``download_model``)
    without hardcoding.
    """
    step = steps_config.get(step_name, {})
    value = step.get(field, fallback)
    return str(value)


def _resolve_step_image(step_name: str, step_cfg: dict[str, Any]) -> str:
    """Resolve the container image for a pipeline step.

    Precedence:
    1. Explicit ``image`` in the step config (from the SuperNova intent).
    2. Fallback built from ``settings.job_step_image_registry`` +
       ``settings.job_step_image_tag`` (step name hyphenated).

    Raises
    ------
    ValueError
        If no image is given and no fallback registry is configured.
    """
    image = step_cfg.get("image")
    if image:
        return str(image)

    registry = settings.job_step_image_registry.rstrip("/")
    if registry:
        return f"{registry}/{step_name.replace('_', '-')}:{settings.job_step_image_tag}"

    raise ValueError(
        f"No container image specified for step '{step_name}' and no "
        f"'job_step_image_registry' fallback is configured"
    )


def _derive_step_environment(
    step_name: str,
    step_cfg: dict[str, Any],
    steps_config: dict[str, Any],
    payload: dict[str, Any],
    job_name: str,
) -> dict[str, str]:
    """Derive the S-021 Section 4.3 step-specific environment for a step.

    Solar Host injects the workspace paths (§4.1) and credentials (§4.2) and
    merges this ``environment`` on top, so Solar Control only supplies the
    step-specific variables here. All values are coerced to ``str`` because the
    host's ``StepDefinition.environment`` is typed ``dict[str, str]``.

    Any explicit ``environment`` (or ``env``) mapping in the step config is
    merged last so callers can override derived values.
    """
    env: dict[str, str] = {}

    if step_name == "download_model":
        env["MODEL_URI"] = str(
            step_cfg.get("model_uri", payload.get("base_model_uri", ""))
        )
        env["MODEL_OUTPUT_DIR"] = str(
            step_cfg.get("output_dir", "/workspace/models/model")
        )

    elif step_name == "download_dataset":
        env["DATASET_URI"] = str(
            step_cfg.get("dataset_uri", payload.get("training_data_uri", ""))
        )
        env["DATASET_OUTPUT_DIR"] = str(
            step_cfg.get("output_dir", "/workspace/data/dataset")
        )

    elif step_name == "train":
        run_name = str(step_cfg.get("run_name", job_name))
        output_dir = str(step_cfg.get("output_dir", f"/workspace/output/{run_name}"))
        env["TRAINING_CONFIG"] = "/workspace/config/training.json"
        env["MODEL_DIR"] = _resolve_train_input(
            steps_config, "download_model", "output_dir", "/workspace/models/model"
        )
        env["DATASET_DIR"] = _resolve_train_input(
            steps_config, "download_dataset", "output_dir", "/workspace/data/dataset"
        )
        env["OUTPUT_DIR"] = output_dir
        env["WANDB"] = str(step_cfg.get("wandb", False)).lower()
        if step_cfg.get("resume"):
            env["RESUME"] = str(step_cfg["resume"])

    elif step_name == "convert_model":
        train_output = _resolve_train_input(
            steps_config, "train", "output_dir", "/workspace/output/run"
        )
        env["MODEL_INPUT"] = str(step_cfg.get("model_input", f"{train_output}/best"))
        env["MODEL_OUTPUT"] = str(step_cfg.get("model_output", f"{train_output}.gguf"))
        env["QUANTIZATION"] = str(step_cfg.get("quantization", "Q4_K_M"))

    elif step_name == "upload_model":
        train_output = _resolve_train_input(
            steps_config, "train", "output_dir", "/workspace/output/run"
        )
        default_source = _resolve_train_input(
            steps_config, "convert_model", "model_output", f"{train_output}/best"
        )
        env["MODEL_SOURCE_PATH"] = str(
            step_cfg.get("model_source_path", default_source)
        )
        env["HARBOR_TARGET_REF"] = str(step_cfg.get("harbor_target_ref", ""))
        env["ARTIFACT_NAME"] = str(step_cfg.get("artifact_name", ""))
        env["VERSION"] = str(step_cfg.get("version", ""))
        env["ARTIFACT_CATEGORY"] = str(step_cfg.get("artifact_category", "model"))
        env["METADATA_PATH"] = "/workspace/config/upload-metadata.json"

    # Merge any explicit environment overrides (stringified) last.
    explicit_env = step_cfg.get("environment") or step_cfg.get("env") or {}
    for key, value in explicit_env.items():
        env[str(key)] = str(value)

    return env


def _build_training_config(
    steps_config: dict[str, Any],
    payload: dict[str, Any],
    job_name: str,
) -> dict[str, Any]:
    """Build the ``training.json`` payload (S-021 Section 7.3).

    Solar Host writes this verbatim to ``/workspace/config/training.json`` when
    a ``train`` step is present. Derived workspace paths are provided as
    sensible defaults and any explicit ``training_config`` from SuperNova is
    merged on top.
    """
    train_cfg = steps_config.get("train", {})
    run_name = str(train_cfg.get("run_name", job_name))
    model_dir = _resolve_train_input(
        steps_config, "download_model", "output_dir", "/workspace/models/model"
    )
    training_config: dict[str, Any] = {
        "name": run_name,
        "model": model_dir,
        "tokenizer": model_dir,
        "output_dir": str(train_cfg.get("output_dir", f"/workspace/output/{run_name}")),
        "train_dataset": _resolve_train_input(
            steps_config, "download_dataset", "output_dir", "/workspace/data/dataset"
        ),
    }
    explicit_config = payload.get("training_config") or {}
    training_config.update(explicit_config)
    return training_config


def _translate_payload(
    payload: dict[str, Any],
    *,
    job_id: str,
    correlation_id: str | None = None,
    submission_id: str | None = None,
) -> dict[str, Any]:
    """Translate a SuperNova-level job intent into a host ``JobDefinition``.

    Produces the exact request body expected by Solar Host's ``POST /jobs``
    (S-027 :class:`JobDefinition`): a top-level ``job_id``/``name``, an ordered
    ``steps`` list (each with ``image``, ``environment``, and optional ``gpu``),
    plus the job-level fields the host writes into ``job.json`` and
    ``training.json``.

    Parameters
    ----------
    payload:
        The raw SuperNova job intent.
    job_id:
        The Solar Control job ID, sent as ``JobDefinition.job_id`` so both
        sides share the identifier.
    correlation_id, submission_id:
        Optional correlation/submission identifiers forwarded to the host.

    Returns
    -------
    dict
        A ``JobDefinition``-shaped dict ready to POST to the host.

    Raises
    ------
    ValueError
        If the pipeline is empty or a step is missing a container image.
    """
    pipeline: list[str] = list(payload.get("pipeline", []))
    if not pipeline:
        raise ValueError("Job intent must include a non-empty 'pipeline'")

    steps_config: dict[str, Any] = payload.get("steps", {})
    job_name: str = str(payload.get("name", "unnamed-job"))

    # --- Build the ordered step list ---
    steps: list[dict[str, Any]] = []
    for step_index, step_name in enumerate(pipeline):
        step_cfg = steps_config.get(step_name, {})

        step_def: dict[str, Any] = {
            "name": step_name,
            "image": _resolve_step_image(step_name, step_cfg),
            "environment": _derive_step_environment(
                step_name, step_cfg, steps_config, payload, job_name
            ),
            "is_preparation_step": bool(
                step_cfg.get("is_preparation_step", step_name in _PREPARATION_STEPS)
            ),
        }

        # GPU: explicit config wins; the train step defaults to a single GPU.
        if "gpu" in step_cfg:
            if step_cfg["gpu"] is not None:
                step_def["gpu"] = step_cfg["gpu"]
        elif step_name == "train":
            step_def["gpu"] = {"count": 1}

        steps.append(step_def)
        logger.debug("Translated step %d: %s -> %s", step_index, step_name, step_def)

    # --- Assemble the JobDefinition ---
    job_definition: dict[str, Any] = {
        "job_id": job_id,
        "name": job_name,
        "steps": steps,
        "retention_hours": payload.get("retention_hours", 24.0),
    }

    # Optional job-level fields (only include when present/relevant).
    if payload.get("min_free_disk_gb") is not None:
        job_definition["min_free_disk_gb"] = payload["min_free_disk_gb"]
    for key in ("base_model_uri", "training_data_uri", "model_selection", "deployment"):
        if payload.get(key) is not None:
            job_definition[key] = payload[key]

    if "train" in pipeline:
        job_definition["training_config"] = _build_training_config(
            steps_config, payload, job_name
        )

    if correlation_id is not None:
        job_definition["correlation_id"] = correlation_id
    if submission_id is not None:
        job_definition["submission_id"] = submission_id

    return job_definition


# ── Endpoints ─────────────────────────────────────────────────


@router.post("", response_model=JobResponse)
async def create_job(data: JobCreate):
    """Submit a job step for execution.

    Flow:
    1. Translate the SuperNova intent into a host ``JobDefinition`` (400 on
       invalid payload).
    2. Select an eligible host (role=training, online, sufficient disk).
    3. Create a ``pending`` job record in the database.
    4. Proxy ``POST /jobs`` to the selected host.
    5. Update job status from the host response.
    6. Return the job record with the result.
    """
    # 1. Translate/validate the payload first (independent of host availability).
    job_id = str(uuid.uuid4())
    try:
        job_definition = _translate_payload(
            data.payload,
            job_id=job_id,
            correlation_id=data.correlation_id,
            submission_id=data.submission_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 2. Select host
    try:
        host = await select_host()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # 3. Create pending job record
    job = Job(
        id=job_id,
        host_id=host.id,
        status=JobStatus.PENDING,
        payload=job_definition,
        correlation_id=data.correlation_id,
        submission_id=data.submission_id,
    )
    await job_db.add_job(job)

    # 4. Proxy to host
    try:
        result = await job_client.submit_job(host, job_definition)
    except JobHostClientError as exc:
        await job_db.update_job_status(job_id, JobStatus.FAILED, error_message=str(exc))
        job.status = JobStatus.FAILED
        job.error_message = str(exc)
        raise HTTPException(
            status_code=502,
            detail={
                "message": f"Host '{host.name}' rejected the job",
                "host_id": host.id,
                "host_name": host.name,
                "host_status_code": exc.status_code,
                "host_body": exc.body,
                "job_id": job_id,
            },
        )
    except (aiohttp.ClientError, TimeoutError) as exc:
        await job_db.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=f"Connection to host failed: {exc}",
        )
        job.status = JobStatus.FAILED
        job.error_message = f"Connection to host failed: {exc}"
        raise HTTPException(
            status_code=502,
            detail={
                "message": f"Could not reach host '{host.name}'",
                "host_id": host.id,
                "host_name": host.name,
                "job_id": job_id,
            },
        )

    # 5. Update status from the host response (host accepts and starts running).
    new_status = JobStatus.from_host(result.get("status")) or JobStatus.RUNNING
    await job_db.update_job_status(job_id, new_status, result=result)
    job.status = new_status
    job.result = result

    logger.info(
        "Job '%s' submitted to host '%s' (%s) — status=%s",
        job_id,
        host.name,
        host.id,
        new_status.value,
    )

    return JobResponse(job=job, message=f"Job submitted to host '{host.name}'")


@router.get("", response_model=list[Job])
async def list_jobs(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List jobs (most recent first)."""
    return await job_db.get_all_jobs(limit=limit, offset=offset)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    """Get job status from the database.

    For non-terminal jobs, also fetches real-time status from the host (the
    host is keyed by the same ``job_id``). Falls back gracefully if the host is
    unreachable.
    """
    job = await _get_job_or_404(job_id)

    host_status: dict[str, Any] | None = None
    if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        try:
            host = await _get_host_or_404(job.host_id)
            host_status = await job_client.get_job_status(host, job_id)
        except (JobHostClientError, aiohttp.ClientError, HTTPException) as exc:
            logger.debug(
                "Could not fetch real-time status for job '%s': %s", job_id, exc
            )

    return JobStatusResponse(job=job, host_status=host_status)


@router.delete("/{job_id}", response_model=JobResponse)
async def cancel_job(job_id: str):
    """Cancel a running or pending job.

    Proxies ``DELETE /jobs/{job_id}`` to the host, then updates the database.
    Idempotent — cancelling an already-terminal job returns success without
    calling the host. A ``404``/``409`` from the host (unknown or already
    terminal) is treated as "already done" and the local state is reconciled.
    """
    job = await _get_job_or_404(job_id)

    if job.status in (JobStatus.CANCELLED, JobStatus.COMPLETED):
        return JobResponse(
            job=job,
            message=f"Job '{job_id}' is already {job.status.value}",
        )

    # Proxy cancel to host
    try:
        host = await _get_host_or_404(job.host_id)
        host_result = await job_client.cancel_job(host, job_id)
    except JobHostClientError as exc:
        if exc.status_code in (404, 409):
            # Host doesn't know about this job, or it's already terminal —
            # reconcile local state without failing the request.
            logger.warning(
                "Host returned HTTP %d for cancel of job '%s' (host=%s): %s",
                exc.status_code,
                job_id,
                job.host_id,
                exc,
            )
        else:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": (
                        f"Host '{exc.host_name}' returned HTTP "
                        f"{exc.status_code} during cancel"
                    ),
                    "host_id": exc.host_id,
                    "host_name": exc.host_name,
                    "job_id": job_id,
                },
            )
    except (aiohttp.ClientError, TimeoutError) as exc:
        logger.warning(
            "Could not reach host '%s' during cancel of job '%s': %s",
            job.host_id,
            job_id,
            exc,
        )
        # Still mark as cancelled locally.
    else:
        logger.info(
            "Job '%s' cancelled on host '%s' (%s): %s",
            job_id,
            host.name,
            host.id,
            host_result,
        )

    await job_db.update_job_status(job_id, JobStatus.CANCELLED)
    job.status = JobStatus.CANCELLED

    return JobResponse(job=job, message=f"Job '{job_id}' cancelled")
