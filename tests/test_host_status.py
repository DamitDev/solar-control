"""Tests for shared host status aggregation (S-033).

Covers the ``active_jobs`` summary built from a translated host
``JobDefinition``, and the three emit paths that must all carry it: the
Socket.IO ``host_status`` broadcast, the WebUI ``initial_status`` snapshot, and
the gateway's HTTP-polling host-online notification.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.database.jobs import job_db
from app.jobs.router import _translate_payload
from app.models import Host, HostStatus
from app.models.job import Job, JobStatus
from app.services.host_status import (
    build_active_job_summary,
    build_host_status_payload,
    get_host_active_jobs,
)

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def training_host():
    return Host(
        id="host-1",
        name="Training Box",
        url="http://training-box:8000",
        api_key="host-key-1",
        status=HostStatus.ONLINE,
        roles=["training"],
        disk_available_gb=500.0,
    )


@pytest.fixture
def job_definition():
    """A real translated ``JobDefinition``, as stored in ``Job.payload``.

    Built through ``_translate_payload`` rather than hand-written so the tests
    break if the host contract's ``steps`` shape ever changes.
    """
    return _translate_payload(
        {
            "name": "iris-osl-retrain",
            "pipeline": ["download_model", "train", "upload_model"],
            "min_free_disk_gb": 200.0,
            "training_config": {
                "batch_size": 8,
                "max_steps": 1000,
                "weight_decay": 0.01,
            },
            "steps": {
                "download_model": {"image": "repo/download-model:v1"},
                "train": {"image": "repo/train:v1", "run_name": "run-1"},
                "upload_model": {"image": "repo/upload-model:v1"},
            },
        },
        job_id="job-1",
    )


def _job(job_definition, **overrides) -> Job:
    defaults = dict(
        id="job-1",
        host_id="host-1",
        status=JobStatus.RUNNING,
        payload=job_definition,
        current_step_name="train",
        current_step_index=1,
    )
    defaults.update(overrides)
    return Job(**defaults)


# ── build_active_job_summary ──────────────────────────────────


def test_summary_reads_pipeline_from_translated_definition(job_definition):
    """``steps`` is an ordered list of dicts on the host side, not a mapping."""
    summary = build_active_job_summary(
        _job(job_definition, submission_id="submission-9")
    )

    assert summary.pipeline == ["download_model", "train", "upload_model"]
    assert summary.name == "iris-osl-retrain"
    assert summary.job_id == "job-1"
    assert summary.submission_id == "submission-9"
    assert summary.status == "running"


def test_summary_exposes_current_step_while_running(job_definition):
    summary = build_active_job_summary(_job(job_definition))

    assert summary.current_step_name == "train"
    assert summary.current_step_index == 1
    assert summary.last_step_name == "train"
    assert summary.last_step_index == 1


def test_summary_step_index_zero_is_preserved(job_definition):
    """Index 0 is falsy — it must not be dropped."""
    summary = build_active_job_summary(
        _job(job_definition, current_step_name="download_model", current_step_index=0)
    )

    assert summary.current_step_index == 0
    assert summary.last_step_index == 0


@pytest.mark.parametrize(
    "status", [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
)
def test_summary_clears_current_step_when_terminal(job_definition, status):
    """A finished job is not executing anything, but we keep the step it reached."""
    completed_at = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
    summary = build_active_job_summary(
        _job(
            job_definition,
            status=status,
            completed_at=completed_at,
            error_message="Out of memory",
        )
    )

    assert summary.current_step_name is None
    assert summary.current_step_index is None
    assert summary.last_step_name == "train"
    assert summary.last_step_index == 1
    assert summary.completed_at == completed_at.isoformat()
    assert summary.error_message == "Out of memory"


def test_summary_resource_hints(job_definition):
    """Hints carry real resource requirements, not just hyperparameters."""
    hints = build_active_job_summary(_job(job_definition)).resource_hints

    # The train step defaults to one GPU in _translate_payload.
    assert hints["peak_gpu_count"] == 1
    assert hints["min_free_disk_gb"] == 200.0
    assert hints["training_config"] == {"batch_size": 8, "max_steps": 1000}


def test_summary_resource_hints_empty_when_nothing_to_report():
    job = _job({"name": "tiny", "steps": [{"name": "noop"}]})

    assert build_active_job_summary(job).resource_hints == {}


@pytest.mark.parametrize(
    "gpu,expected",
    [
        ({"count": 4}, 4),
        ({}, None),
        (2, 2),
        ([0, 1, 2], 3),
        (None, None),
    ],
)
def test_summary_peak_gpu_count_tolerates_gpu_shapes(gpu, expected):
    """The gpu block comes straight from the intent, so it may be any shape."""
    payload = {"steps": [{"name": "a"}, {"name": "b", "gpu": gpu}]}

    hints = build_active_job_summary(_job(payload)).resource_hints

    assert hints.get("peak_gpu_count") == expected


def test_summary_tolerates_malformed_steps():
    payload = {"steps": [{"name": "ok"}, {"image": "no-name"}, "garbage", None]}

    assert build_active_job_summary(_job(payload)).pipeline == ["ok"]


def test_summary_tolerates_empty_payload():
    summary = build_active_job_summary(_job({}, current_step_name=None))

    assert summary.pipeline == []
    assert summary.name is None
    assert summary.resource_hints == {}


# ── get_host_active_jobs ──────────────────────────────────────


@pytest.mark.anyio
async def test_get_host_active_jobs_maps_rows(job_definition):
    with patch.object(
        job_db, "get_active_by_host", AsyncMock(return_value=[_job(job_definition)])
    ):
        summaries = await get_host_active_jobs("host-1")

    assert [s.job_id for s in summaries] == ["job-1"]


@pytest.mark.anyio
async def test_get_host_active_jobs_swallows_db_errors():
    """Host status is emitted from the connect handler — it must never raise."""
    with patch.object(
        job_db, "get_active_by_host", AsyncMock(side_effect=RuntimeError("db down"))
    ):
        assert await get_host_active_jobs("host-1") == []


@pytest.mark.anyio
async def test_build_host_status_payload_includes_active_jobs(
    training_host, job_definition
):
    with patch.object(
        job_db, "get_active_by_host", AsyncMock(return_value=[_job(job_definition)])
    ):
        payload = await build_host_status_payload(training_host, connected=True)

    assert payload.host_id == "host-1"
    assert payload.connected is True
    assert [j.job_id for j in payload.active_jobs] == ["job-1"]


# ── Emit paths ────────────────────────────────────────────────


@pytest.mark.anyio
async def test_emit_host_status_survives_db_failure(training_host):
    """A database outage must not stop hosts from connecting."""
    from app.socketio_app import host_handlers

    with (
        patch.object(
            job_db, "get_active_by_host", AsyncMock(side_effect=RuntimeError("db down"))
        ),
        patch.object(host_handlers.sio, "emit", AsyncMock()) as mock_emit,
    ):
        await host_handlers._emit_host_status(training_host, connected=True)

    assert mock_emit.await_args.args[1]["active_jobs"] == []


@pytest.mark.anyio
async def test_webui_initial_status_includes_active_jobs(training_host, job_definition):
    """The first snapshot a WebUI client receives must carry active jobs."""
    from app.socketio_app import webui_handlers

    with (
        patch.object(
            webui_handlers, "settings", SimpleNamespace(management_api_key="mgmt-key")
        ),
        patch.object(
            webui_handlers.host_db,
            "get_all_hosts",
            AsyncMock(return_value=[training_host]),
        ),
        patch.object(webui_handlers, "is_host_connected", AsyncMock(return_value=True)),
        patch.object(
            webui_handlers, "get_connected_host_ids", AsyncMock(return_value=[])
        ),
        patch.object(webui_handlers, "get_pending_hosts", AsyncMock(return_value=[])),
        patch.object(
            job_db, "get_active_by_host", AsyncMock(return_value=[_job(job_definition)])
        ),
        patch.object(webui_handlers.sio, "emit", AsyncMock()) as mock_emit,
    ):
        await webui_handlers.webui_connect(
            "sid-1", {"headers": []}, {"api_key": "mgmt-key"}
        )

    initial = next(
        c.args[1] for c in mock_emit.call_args_list if c.args[0] == "initial_status"
    )
    assert [j["job_id"] for j in initial[0]["active_jobs"]] == ["job-1"]


@pytest.mark.anyio
async def test_gateway_host_online_notification_includes_active_jobs(
    training_host, job_definition
):
    """HTTP polling must not broadcast an empty list over live job state."""
    from app.gateway import gateway
    from app.socketio_app import server

    with (
        patch("app.gateway.host_db.get_host", AsyncMock(return_value=training_host)),
        patch.object(
            job_db, "get_active_by_host", AsyncMock(return_value=[_job(job_definition)])
        ),
        patch.object(server.sio, "emit", AsyncMock()) as mock_emit,
    ):
        await gateway._notify_host_online(training_host)

    event, payload = mock_emit.await_args.args
    assert event == "host_status"
    assert [j["job_id"] for j in payload["active_jobs"]] == ["job-1"]
