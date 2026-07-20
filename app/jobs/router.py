"""Job step execution API routes (under /api/jobs).

Provides:
- ``POST /api/jobs`` — Submit a job step (select host, proxy, return result)
- ``GET /api/jobs/{id}`` — Get job status (DB + optional host proxy)
- ``DELETE /api/jobs/{id}`` — Cancel a job (proxy to host + DB update)
"""

import logging
import uuid
from typing import Any

import aiohttp
from fastapi import APIRouter, HTTPException

from app.database.hosts import host_db
from app.database.jobs import job_db
from app.jobs.client import JobHostClientError, job_client
from app.jobs.host_selector import select_host
from app.models import Host
from app.models.job import Job, JobCreate, JobResponse, JobStatus, JobStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


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


def _translate_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Translate SuperNova-level job intent to host-level job config.

    Per S-021 workspace spec (Sections 4.3, 5.2, 7.3):
    - Derives per-step environment variables from the high-level job description
    - Constructs the training config JSON if a train step is present
    - Sets workspace paths (models, data, output, config) for each step
    - Passes through infrastructure credentials (Harbor, HF, W&B)

    Parameters
    ----------
    payload:
        The raw job step configuration from SuperNova.

    Returns
    -------
    dict
        The host-level job config to send to ``POST /jobs``.
        Includes the job manifest (``job.json`` content), per-step
        environment variables, and the training config if applicable.
    """
    pipeline: list[str] = payload.get("pipeline", [])
    steps_config: dict[str, Any] = payload.get("steps", {})
    job_name: str = payload.get("name", "unnamed-job")

    # --- Build job manifest (job.json) ---
    job_manifest: dict[str, Any] = {
        "name": job_name,
        "pipeline": pipeline,
        "base_model_uri": payload.get("base_model_uri"),
        "training_data_uri": payload.get("training_data_uri"),
        "model_selection": payload.get("model_selection"),
        "deployment": payload.get("deployment"),
        "retention_hours": payload.get("retention_hours", 24),
        "steps": {name: steps_config.get(name, {}) for name in pipeline},
    }

    # --- Derive per-step environment variables (S-021 Section 4.3) ---
    step_envs: dict[str, dict[str, Any]] = {}

    for step_name in pipeline:
        step_cfg = steps_config.get(step_name, {})
        env: dict[str, Any] = {}

        if step_name == "download_model":
            env["MODEL_URI"] = step_cfg.get(
                "model_uri", payload.get("base_model_uri", "")
            )
            env["MODEL_OUTPUT_DIR"] = step_cfg.get(
                "output_dir", "/workspace/models/model"
            )

        elif step_name == "download_dataset":
            env["DATASET_URI"] = step_cfg.get(
                "dataset_uri", payload.get("training_data_uri", "")
            )
            env["DATASET_OUTPUT_DIR"] = step_cfg.get(
                "output_dir", "/workspace/data/dataset"
            )

        elif step_name == "train":
            run_name = step_cfg.get("run_name", job_name)
            output_dir = step_cfg.get("output_dir", f"/workspace/output/{run_name}")
            env["TRAINING_CONFIG"] = "/workspace/config/training.json"
            env["MODEL_DIR"] = _resolve_train_input(
                steps_config,
                "download_model",
                "output_dir",
                "/workspace/models/model",
            )
            env["DATASET_DIR"] = _resolve_train_input(
                steps_config,
                "download_dataset",
                "output_dir",
                "/workspace/data/dataset",
            )
            env["OUTPUT_DIR"] = output_dir
            env["WANDB"] = str(step_cfg.get("wandb", False)).lower()

        elif step_name == "convert_model":
            train_output = _resolve_train_input(
                steps_config,
                "train",
                "output_dir",
                "/workspace/output/run",
            )
            env["MODEL_INPUT"] = step_cfg.get("model_input", f"{train_output}/best")
            env["MODEL_OUTPUT"] = step_cfg.get("model_output", f"{train_output}.gguf")
            env["QUANTIZATION"] = step_cfg.get("quantization", "Q4_K_M")

        elif step_name == "upload_model":
            env["MODEL_SOURCE_PATH"] = step_cfg.get(
                "model_source_path",
                _resolve_train_input(
                    steps_config,
                    "convert_model",
                    "model_output",
                    _resolve_train_input(
                        steps_config,
                        "train",
                        "output_dir",
                        "/workspace/output/run",
                    )
                    + "/best",
                ),
            )
            env["HARBOR_TARGET_REF"] = step_cfg.get("harbor_target_ref", "")
            env["ARTIFACT_NAME"] = step_cfg.get("artifact_name", "")
            env["VERSION"] = step_cfg.get("version", "")
            env["ARTIFACT_CATEGORY"] = step_cfg.get("artifact_category", "model")
            env["METADATA_PATH"] = "/workspace/config/upload-metadata.json"

        step_envs[step_name] = env

    # --- Build training config if train step is present (S-021 Section 7.3) ---
    training_config: dict[str, Any] | None = None
    if "train" in pipeline:
        train_cfg = steps_config.get("train", {})
        run_name = train_cfg.get("run_name", job_name)
        training_config = {
            "name": run_name,
            "model": _resolve_train_input(
                steps_config,
                "download_model",
                "output_dir",
                "/workspace/models/model",
            ),
            "tokenizer": _resolve_train_input(
                steps_config,
                "download_model",
                "output_dir",
                "/workspace/models/model",
            ),
            "output_dir": train_cfg.get("output_dir", f"/workspace/output/{run_name}"),
            "train_dataset": _resolve_train_input(
                steps_config,
                "download_dataset",
                "output_dir",
                "/workspace/data/dataset",
            ),
        }
        # Merge in any explicit training config from SuperNova
        explicit_config = payload.get("training_config") or {}
        training_config.update(explicit_config)

    # --- Assemble host-level payload ---
    host_payload: dict[str, Any] = {
        "job_manifest": job_manifest,
        "step_envs": step_envs,
    }

    if training_config is not None:
        host_payload["training_config"] = training_config

    # Pass through any additional fields the host may need
    for key in ("retention_hours",):
        if key in payload:
            host_payload[key] = payload[key]

    return host_payload


def _resolve_train_input(
    steps_config: dict[str, Any],
    step_name: str,
    field: str,
    fallback: str,
) -> str:
    """Resolve a downstream step input from an upstream step's config.

    Looks up ``steps_config[step_name][field]`` and returns it if
    present, otherwise returns ``fallback``. This lets downstream
    steps (e.g. ``train``) derive their input paths from upstream
    steps (e.g. ``download_model``) without hardcoding.
    """
    step = steps_config.get(step_name, {})
    return step.get(field, fallback)


# ── Endpoints ─────────────────────────────────────────────────


@router.post("", response_model=JobResponse)
async def create_job(data: JobCreate):
    """Submit a job step for execution.

    Flow:
    1. Select an eligible host (role=training, online, sufficient disk)
    2. Translate the payload to host-level config
    3. Create a ``pending`` job record in the database
    4. Proxy ``POST /jobs`` to the selected host
    5. Update job status to ``running`` or ``failed``
    6. Return the job record with result
    """
    # 1. Select host
    try:
        host = await select_host()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    # 2. Translate payload
    host_payload = _translate_payload(data.payload)

    # 3. Create pending job record
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        host_id=host.id,
        status=JobStatus.PENDING,
        payload=host_payload,
        correlation_id=data.correlation_id,
    )
    await job_db.add_job(job)

    # 4. Proxy to host
    try:
        result = await job_client.submit_job(host, host_payload)
    except JobHostClientError as exc:
        await job_db.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=str(exc),
        )
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

    # 5. Mark running
    await job_db.update_job_status(job_id, JobStatus.RUNNING, result=result)
    job.status = JobStatus.RUNNING
    job.result = result

    logger.info(
        "Job '%s' submitted to host '%s' (%s) — running",
        job_id,
        host.name,
        host.id,
    )

    return JobResponse(
        job=job,
        message=f"Job submitted to host '{host.name}'",
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    """Get job status from the database.

    Optionally proxies to the host for real-time status.
    Falls back gracefully if the host is unreachable.
    """
    job = await _get_job_or_404(job_id)

    # Try to get real-time status from the host
    host_status: dict[str, Any] | None = None
    if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        try:
            host = await _get_host_or_404(job.host_id)
            host_status = await job_client.get_job_status(host, job_id)
        except (JobHostClientError, aiohttp.ClientError, HTTPException) as exc:
            logger.debug(
                "Could not fetch real-time status for job '%s': %s",
                job_id,
                exc,
            )

    return JobStatusResponse(job=job, host_status=host_status)


@router.delete("/{job_id}", response_model=JobResponse)
async def cancel_job(job_id: str):
    """Cancel a running or pending job.

    Proxies ``DELETE /jobs/{job_id}`` to the host, then updates
    the database. Idempotent — cancelling an already-cancelled or
    completed job returns success without calling the host.
    """
    job = await _get_job_or_404(job_id)

    if job.status in (JobStatus.CANCELLED, JobStatus.COMPLETED):
        return JobResponse(
            job=job,
            message=f"Job '{job_id}' is already {job.status.value}",
        )

    if job.status == JobStatus.FAILED:
        # Mark as cancelled in our DB even though the host may have
        # already cleaned up
        await job_db.update_job_status(job_id, JobStatus.CANCELLED)
        job.status = JobStatus.CANCELLED
        return JobResponse(
            job=job,
            message=f"Job '{job_id}' was already failed; marked as cancelled",
        )

    # Proxy cancel to host
    try:
        host = await _get_host_or_404(job.host_id)
        host_result = await job_client.cancel_job(host, job_id)
    except JobHostClientError as exc:
        if exc.status_code == 404:
            # Host doesn't know about this job — just update local state
            logger.warning(
                "Job '%s' not found on host '%s' during cancel: %s",
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
        # Still mark as cancelled locally
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

    return JobResponse(
        job=job,
        message=f"Job '{job_id}' cancelled",
    )
