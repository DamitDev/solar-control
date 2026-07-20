"""Tests for job step execution: host selection, submission, status, cancel, and event forwarding."""

from unittest.mock import AsyncMock, patch

import pytest

from app.database.jobs import JobDB
from app.jobs.host_selector import select_host
from app.jobs.client import JobHostClient, JobHostClientError
from app.jobs.router import (
    _translate_payload,
    _resolve_train_input,
)
from app.models import Host, HostStatus
from app.models.job import Job, JobStatus

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def online_training_host():
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
def offline_training_host():
    return Host(
        id="host-2",
        name="Offline Box",
        url="http://offline-box:8000",
        api_key="host-key-2",
        status=HostStatus.OFFLINE,
        roles=["training"],
        disk_available_gb=500.0,
    )


@pytest.fixture
def low_disk_host():
    return Host(
        id="host-3",
        name="Low Disk Box",
        url="http://low-disk:8000",
        api_key="host-key-3",
        status=HostStatus.ONLINE,
        roles=["training"],
        disk_available_gb=5.0,
    )


@pytest.fixture
def inference_only_host():
    return Host(
        id="host-4",
        name="Inference Box",
        url="http://inference:8000",
        api_key="host-key-4",
        status=HostStatus.ONLINE,
        roles=["gpu"],
        disk_available_gb=500.0,
    )


@pytest.fixture
def sample_supernova_payload():
    return {
        "name": "iris-osl-retrain-2026-03",
        "pipeline": [
            "download_model",
            "download_dataset",
            "train",
            "upload_model",
        ],
        "base_model_uri": "repo://IRIS-BERT-base:v1",
        "training_data_uri": "repo://iris-tickets:2026-03",
        "model_selection": {
            "strategy": "best_metric",
            "metric": "f1",
            "direction": "max",
        },
        "deployment": {
            "target": "iris-osl:110m",
            "replicas": 2,
            "strategy": "rolling",
        },
        "retention_hours": 24,
        "steps": {
            "download_model": {
                "model_uri": "repo://IRIS-BERT-base:v1",
                "output_dir": "/workspace/models/IRIS-BERT-base",
            },
            "download_dataset": {
                "dataset_uri": "repo://iris-tickets:2026-03",
                "output_dir": "/workspace/data/tickets-dataset",
            },
            "train": {
                "run_name": "base-osl-2026-05",
                "output_dir": "/workspace/output/base_osl",
                "wandb": False,
            },
            "upload_model": {
                "harbor_target_ref": "imgrepo.damit.hu/supernova/iris-osl:v4",
                "artifact_name": "iris-osl",
                "version": "v4",
                "artifact_category": "model",
            },
        },
    }


# ── Host Selection Tests ─────────────────────────────────────


@pytest.mark.anyio
async def test_select_host_filters_by_role(online_training_host, inference_only_host):
    """Only hosts with role='training' should be considered."""
    with patch(
        "app.jobs.host_selector.host_db.get_all_hosts",
        AsyncMock(return_value=[online_training_host]),
    ):
        host = await select_host(role="training", min_disk_gb=10.0)
        assert "training" in host.roles


@pytest.mark.anyio
async def test_select_host_filters_offline(online_training_host, offline_training_host):
    """Offline hosts should be excluded."""
    with patch(
        "app.jobs.host_selector.host_db.get_all_hosts",
        AsyncMock(return_value=[online_training_host, offline_training_host]),
    ):
        host = await select_host(role="training", min_disk_gb=10.0)
        assert host.id == "host-1"


@pytest.mark.anyio
async def test_select_host_filters_by_disk(online_training_host, low_disk_host):
    """Hosts with insufficient disk should be excluded."""
    with patch(
        "app.jobs.host_selector.host_db.get_all_hosts",
        AsyncMock(return_value=[online_training_host, low_disk_host]),
    ):
        host = await select_host(role="training", min_disk_gb=50.0)
        assert host.id == "host-1"


@pytest.mark.anyio
async def test_select_host_no_host_available(online_training_host):
    """Should raise RuntimeError when no host qualifies."""
    with patch(
        "app.jobs.host_selector.host_db.get_all_hosts",
        AsyncMock(return_value=[online_training_host]),
    ):
        with pytest.raises(RuntimeError, match="No training-capable host"):
            await select_host(role="training", min_disk_gb=999999.0)


@pytest.mark.anyio
async def test_select_host_empty_list():
    """Should raise RuntimeError when no hosts exist at all."""
    with patch(
        "app.jobs.host_selector.host_db.get_all_hosts",
        AsyncMock(return_value=[]),
    ):
        with pytest.raises(RuntimeError, match="No training-capable host"):
            await select_host(role="training")


@pytest.mark.anyio
async def test_select_host_picks_most_disk(online_training_host, low_disk_host):
    """Should pick the host with the most available disk."""
    big_disk_host = Host(
        id="host-big",
        name="Big Disk",
        url="http://big:8000",
        api_key="key",
        status=HostStatus.ONLINE,
        roles=["training"],
        disk_available_gb=2000.0,
    )
    with patch(
        "app.jobs.host_selector.host_db.get_all_hosts",
        AsyncMock(return_value=[online_training_host, low_disk_host, big_disk_host]),
    ):
        host = await select_host(role="training", min_disk_gb=10.0)
        assert host.id == "host-big"


# ── Payload Translation Tests ────────────────────────────────


def test_translate_payload_basic(sample_supernova_payload):
    """Should produce a host-level payload with job_manifest and step_envs."""
    result = _translate_payload(sample_supernova_payload)

    assert "job_manifest" in result
    assert "step_envs" in result
    assert result["job_manifest"]["name"] == "iris-osl-retrain-2026-03"
    assert result["job_manifest"]["pipeline"] == [
        "download_model",
        "download_dataset",
        "train",
        "upload_model",
    ]
    assert result["retention_hours"] == 24


def test_translate_payload_step_envs(sample_supernova_payload):
    """Each pipeline step should get its derived environment variables."""
    result = _translate_payload(sample_supernova_payload)
    step_envs = result["step_envs"]

    # download_model
    assert step_envs["download_model"]["MODEL_URI"] == "repo://IRIS-BERT-base:v1"
    assert (
        step_envs["download_model"]["MODEL_OUTPUT_DIR"]
        == "/workspace/models/IRIS-BERT-base"
    )

    # download_dataset
    assert step_envs["download_dataset"]["DATASET_URI"] == "repo://iris-tickets:2026-03"
    assert (
        step_envs["download_dataset"]["DATASET_OUTPUT_DIR"]
        == "/workspace/data/tickets-dataset"
    )

    # train
    assert step_envs["train"]["TRAINING_CONFIG"] == "/workspace/config/training.json"
    assert step_envs["train"]["MODEL_DIR"] == "/workspace/models/IRIS-BERT-base"
    assert step_envs["train"]["DATASET_DIR"] == "/workspace/data/tickets-dataset"
    assert step_envs["train"]["OUTPUT_DIR"] == "/workspace/output/base_osl"
    assert step_envs["train"]["WANDB"] == "false"

    # upload_model
    assert (
        step_envs["upload_model"]["HARBOR_TARGET_REF"]
        == "imgrepo.damit.hu/supernova/iris-osl:v4"
    )
    assert step_envs["upload_model"]["ARTIFACT_NAME"] == "iris-osl"


def test_translate_payload_training_config(sample_supernova_payload):
    """Should generate training.json when train step is present."""
    result = _translate_payload(sample_supernova_payload)
    tc = result["training_config"]

    assert tc["name"] == "base-osl-2026-05"
    assert tc["model"] == "/workspace/models/IRIS-BERT-base"
    assert tc["output_dir"] == "/workspace/output/base_osl"
    assert tc["train_dataset"] == "/workspace/data/tickets-dataset"


def test_translate_payload_no_train():
    """Should not include training_config when no train step."""
    payload = {
        "name": "simple-job",
        "pipeline": ["download_model"],
        "steps": {
            "download_model": {
                "model_uri": "repo://some-model:v1",
                "output_dir": "/workspace/models/some-model",
            }
        },
    }
    result = _translate_payload(payload)
    assert "training_config" not in result


def test_resolve_train_input():
    """Should resolve upstream step output paths correctly."""
    steps = {
        "download_model": {"output_dir": "/workspace/models/foo"},
    }
    result = _resolve_train_input(steps, "download_model", "output_dir", "/fallback")
    assert result == "/workspace/models/foo"

    # Fallback when step config is missing
    result = _resolve_train_input(steps, "missing_step", "output_dir", "/fallback")
    assert result == "/fallback"


# ── JobHostClient Tests ──────────────────────────────────────


class _MockResponse:
    def __init__(self, status: int, json_data=None, text_data=None):
        self.status = status
        self._json_data = json_data
        self._text_data = text_data

    async def json(self):
        if self._json_data is not None:
            return self._json_data
        raise ValueError("No JSON data")

    async def text(self):
        return self._text_data or ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _MockSession:
    closed = False

    def __init__(self, response: _MockResponse):
        self._response = response

    def post(self, url, **kwargs):
        self.last_url = url
        self.last_kwargs = kwargs
        return self._response

    def get(self, url, **kwargs):
        self.last_url = url
        self.last_kwargs = kwargs
        return self._response

    def delete(self, url, **kwargs):
        self.last_url = url
        self.last_kwargs = kwargs
        return self._response

    async def close(self):
        self.closed = True


@pytest.mark.anyio
async def test_client_submit_job_success(online_training_host):
    """Successful job submission should return the host response."""
    client = JobHostClient()
    host_response = {"job_id": "host-job-1", "status": "accepted"}
    client._session = _MockSession(_MockResponse(200, host_response))

    result = await client.submit_job(online_training_host, {"task": "train"})
    assert result == host_response


@pytest.mark.anyio
async def test_client_submit_job_failure(online_training_host):
    """Host rejection should raise JobHostClientError."""
    client = JobHostClient()
    client._session = _MockSession(_MockResponse(400, {"error": "bad request"}))

    with pytest.raises(JobHostClientError) as exc:
        await client.submit_job(online_training_host, {"task": "train"})
    assert exc.value.status_code == 400
    assert exc.value.host_id == "host-1"


@pytest.mark.anyio
async def test_client_get_job_status(online_training_host):
    """GET should return the host's job status response."""
    client = JobHostClient()
    status_response = {"status": "running", "progress": 0.5}
    client._session = _MockSession(_MockResponse(200, status_response))

    result = await client.get_job_status(online_training_host, "job-1")
    assert result == status_response


@pytest.mark.anyio
async def test_client_cancel_job(online_training_host):
    """DELETE should return the host's cancel response."""
    client = JobHostClient()
    cancel_response = {"status": "cancelled"}
    client._session = _MockSession(_MockResponse(200, cancel_response))

    result = await client.cancel_job(online_training_host, "job-1")
    assert result == cancel_response


@pytest.mark.anyio
async def test_client_cancel_job_not_found(online_training_host):
    """Host 404 on cancel should raise JobHostClientError."""
    client = JobHostClient()
    client._session = _MockSession(_MockResponse(404, {"error": "not found"}))

    with pytest.raises(JobHostClientError) as exc:
        await client.cancel_job(online_training_host, "job-unknown")
    assert exc.value.status_code == 404


# ── Job DB Tests ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_job_db_add_and_get():
    """Should persist a job and retrieve it by ID."""
    db = JobDB()
    job = Job(id="test-job-1", host_id="host-1", payload={"key": "value"})

    with patch.object(db, "_session") as mock_session_ctx:
        mock_session = AsyncMock()
        mock_session_ctx.return_value.__aenter__.return_value = mock_session

        await db.add_job(job)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()


@pytest.mark.anyio
async def test_job_db_update_status():
    """Should update job status and set completed_at for terminal states."""
    db = JobDB()

    with patch.object(db, "_session") as mock_session_ctx:
        mock_session = AsyncMock()
        mock_session_ctx.return_value.__aenter__.return_value = mock_session

        result = await db.update_job_status("job-1", JobStatus.COMPLETED)
        assert result is True
        mock_session.commit.assert_called_once()


# ── Event Forwarding Tests ───────────────────────────────────


def test_job_log_payload_shape():
    """JobLogPayload should contain all required fields."""
    from app.models.socketio import JobLogPayload

    payload = JobLogPayload(
        job_id="job-1",
        host_id="host-1",
        host_name="Test Host",
        seq=42,
        line="Epoch 3/10 loss=0.42",
        level="info",
        correlation_id="corr-123",
        timestamp="2026-07-20T12:00:00Z",
    )
    data = payload.model_dump()
    assert data["job_id"] == "job-1"
    assert data["host_id"] == "host-1"
    assert data["seq"] == 42
    assert data["line"] == "Epoch 3/10 loss=0.42"
    assert data["correlation_id"] == "corr-123"


def test_job_lifecycle_payload_shape():
    """JobLifecyclePayload should contain all required fields."""
    from app.models.socketio import JobLifecyclePayload

    payload = JobLifecyclePayload(
        job_id="job-1",
        host_id="host-1",
        host_name="Test Host",
        event="step_completed",
        step_name="train",
        correlation_id="corr-123",
        data={"exit_code": 0},
        timestamp="2026-07-20T12:00:00Z",
    )
    data = payload.model_dump()
    assert data["event"] == "step_completed"
    assert data["step_name"] == "train"
    assert data["data"]["exit_code"] == 0


# ── WebUI Filter Tests ───────────────────────────────────────


@pytest.mark.anyio
async def test_should_emit_to_client_no_filter():
    """Without a filter, all events should pass through."""
    from app.socketio_app.webui_handlers import _should_emit_to_client

    assert await _should_emit_to_client(None, "job_log", {"job_id": "job-1"}) is True


@pytest.mark.anyio
async def test_should_emit_to_client_job_id_filter():
    """Should filter by job_id when filter is set."""
    from app.socketio_app.webui_handlers import _should_emit_to_client

    filt = {"job_ids": ["job-1", "job-2"]}
    assert await _should_emit_to_client(filt, "job_log", {"job_id": "job-1"}) is True
    assert await _should_emit_to_client(filt, "job_log", {"job_id": "job-3"}) is False


@pytest.mark.anyio
async def test_should_emit_to_client_host_id_filter():
    """Should filter by host_id when filter is set."""
    from app.socketio_app.webui_handlers import _should_emit_to_client

    filt = {"host_ids": ["host-1"]}
    assert (
        await _should_emit_to_client(filt, "host_status", {"host_id": "host-1"}) is True
    )
    assert (
        await _should_emit_to_client(filt, "host_status", {"host_id": "host-2"})
        is False
    )
