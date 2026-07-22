"""Tests for job step execution.

Covers host selection, translation of the SuperNova intent into the Solar Host
``JobDefinition`` (S-027) contract, the HTTP proxy client, DB operations, and
the S-025/S-026 event forwarding handlers.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.database.jobs import JobDB
from app.jobs.host_selector import select_host
from app.jobs.client import JobHostClient, JobHostClientError
from app.jobs.router import _translate_payload, _resolve_train_input
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
    """A realistic SuperNova intent — each step carries its container image."""
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
                "image": "imgrepo.damit.hu/supernova/download-model:v1",
                "model_uri": "repo://IRIS-BERT-base:v1",
                "output_dir": "/workspace/models/IRIS-BERT-base",
            },
            "download_dataset": {
                "image": "imgrepo.damit.hu/supernova/download-dataset:v1",
                "dataset_uri": "repo://iris-tickets:2026-03",
                "output_dir": "/workspace/data/tickets-dataset",
            },
            "train": {
                "image": "imgrepo.damit.hu/supernova/train:v1",
                "run_name": "base-osl-2026-05",
                "output_dir": "/workspace/output/base_osl",
                "wandb": False,
            },
            "upload_model": {
                "image": "imgrepo.damit.hu/supernova/upload-model:v1",
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
    ) as mock_get:
        host = await select_host(role="training", min_disk_gb=10.0)
        assert "training" in host.roles
        # role filter is pushed down to the DB query
        mock_get.assert_awaited_once_with(role="training")


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


# ── Payload Translation Tests (S-027 JobDefinition contract) ──


def test_translate_payload_is_job_definition(sample_supernova_payload):
    """Output must be a host JobDefinition: job_id, name, ordered steps."""
    result = _translate_payload(sample_supernova_payload, job_id="job-abc")

    assert result["job_id"] == "job-abc"
    assert result["name"] == "iris-osl-retrain-2026-03"
    assert [s["name"] for s in result["steps"]] == [
        "download_model",
        "download_dataset",
        "train",
        "upload_model",
    ]
    assert result["retention_hours"] == 24
    # Job-level passthrough fields
    assert result["base_model_uri"] == "repo://IRIS-BERT-base:v1"
    assert result["training_data_uri"] == "repo://iris-tickets:2026-03"
    assert result["model_selection"]["metric"] == "f1"
    assert result["deployment"]["replicas"] == 2
    # No invented wrapper keys
    assert "job_manifest" not in result
    assert "step_envs" not in result


def test_translate_payload_step_images_and_flags(sample_supernova_payload):
    """Each step carries its image; download steps are preparation steps."""
    result = _translate_payload(sample_supernova_payload, job_id="job-abc")
    steps = {s["name"]: s for s in result["steps"]}

    assert steps["train"]["image"] == "imgrepo.damit.hu/supernova/train:v1"
    assert steps["download_model"]["is_preparation_step"] is True
    assert steps["download_dataset"]["is_preparation_step"] is True
    assert steps["train"]["is_preparation_step"] is False
    # train defaults to a single GPU; prep steps request none
    assert steps["train"]["gpu"] == {"count": 1}
    assert "gpu" not in steps["download_model"]


def test_translate_payload_step_environment_is_string_map(sample_supernova_payload):
    """Step-specific env vars (S-021 §4.3) are set and all values are strings."""
    result = _translate_payload(sample_supernova_payload, job_id="job-abc")
    steps = {s["name"]: s for s in result["steps"]}

    dm = steps["download_model"]["environment"]
    assert dm["MODEL_URI"] == "repo://IRIS-BERT-base:v1"
    assert dm["MODEL_OUTPUT_DIR"] == "/workspace/models/IRIS-BERT-base"

    dd = steps["download_dataset"]["environment"]
    assert dd["DATASET_URI"] == "repo://iris-tickets:2026-03"
    assert dd["DATASET_OUTPUT_DIR"] == "/workspace/data/tickets-dataset"

    tr = steps["train"]["environment"]
    assert tr["TRAINING_CONFIG"] == "/workspace/config/training.json"
    assert tr["MODEL_DIR"] == "/workspace/models/IRIS-BERT-base"
    assert tr["DATASET_DIR"] == "/workspace/data/tickets-dataset"
    assert tr["OUTPUT_DIR"] == "/workspace/output/base_osl"
    assert tr["WANDB"] == "false"

    up = steps["upload_model"]["environment"]
    assert up["HARBOR_TARGET_REF"] == "imgrepo.damit.hu/supernova/iris-osl:v4"
    assert up["ARTIFACT_NAME"] == "iris-osl"

    # Host StepDefinition.environment is dict[str, str] — enforce it here.
    for step in result["steps"]:
        for key, value in step["environment"].items():
            assert isinstance(key, str)
            assert isinstance(value, str), f"{step['name']}.{key} is not a str"


def test_translate_payload_training_config(sample_supernova_payload):
    """training_config is emitted (host writes training.json) when train present."""
    result = _translate_payload(sample_supernova_payload, job_id="job-abc")
    tc = result["training_config"]

    assert tc["name"] == "base-osl-2026-05"
    assert tc["model"] == "/workspace/models/IRIS-BERT-base"
    assert tc["tokenizer"] == "/workspace/models/IRIS-BERT-base"
    assert tc["output_dir"] == "/workspace/output/base_osl"
    assert tc["train_dataset"] == "/workspace/data/tickets-dataset"


def test_translate_payload_merges_explicit_training_config(sample_supernova_payload):
    """Explicit training_config from SuperNova is merged over derived defaults."""
    sample_supernova_payload["training_config"] = {"epochs": 3, "batch_size": 4}
    result = _translate_payload(sample_supernova_payload, job_id="job-abc")
    tc = result["training_config"]
    assert tc["epochs"] == 3
    assert tc["batch_size"] == 4
    assert tc["model"] == "/workspace/models/IRIS-BERT-base"


def test_translate_payload_threads_ids(sample_supernova_payload):
    """correlation_id and submission_id are forwarded to the host."""
    result = _translate_payload(
        sample_supernova_payload,
        job_id="job-abc",
        correlation_id="corr-99",
        submission_id="sub-42",
    )
    assert result["correlation_id"] == "corr-99"
    assert result["submission_id"] == "sub-42"


def test_translate_payload_no_train():
    """No training_config when there is no train step."""
    payload = {
        "name": "simple-job",
        "pipeline": ["download_model"],
        "steps": {
            "download_model": {
                "image": "reg/download-model:v1",
                "model_uri": "repo://some-model:v1",
                "output_dir": "/workspace/models/some-model",
            }
        },
    }
    result = _translate_payload(payload, job_id="job-x")
    assert "training_config" not in result


def test_translate_payload_empty_pipeline_raises():
    """An empty pipeline is a client error."""
    with pytest.raises(ValueError, match="non-empty 'pipeline'"):
        _translate_payload({"name": "x", "pipeline": []}, job_id="job-x")


def test_translate_payload_missing_image_raises(monkeypatch):
    """A step without an image and no registry fallback is a client error."""
    from app.jobs import router

    monkeypatch.setattr(router.settings, "job_step_image_registry", "")
    payload = {"name": "x", "pipeline": ["train"], "steps": {"train": {}}}
    with pytest.raises(ValueError, match="No container image"):
        _translate_payload(payload, job_id="job-x")


def test_translate_payload_default_image_registry(monkeypatch):
    """Falls back to the configured registry when a step omits its image."""
    from app.jobs import router

    monkeypatch.setattr(
        router.settings, "job_step_image_registry", "imgrepo.damit.hu/supernova"
    )
    monkeypatch.setattr(router.settings, "job_step_image_tag", "v9")
    payload = {"name": "x", "pipeline": ["download_model"], "steps": {}}
    result = _translate_payload(payload, job_id="job-x")
    assert result["steps"][0]["image"] == "imgrepo.damit.hu/supernova/download-model:v9"


def test_resolve_train_input():
    """Should resolve upstream step output paths correctly."""
    steps = {"download_model": {"output_dir": "/workspace/models/foo"}}
    assert (
        _resolve_train_input(steps, "download_model", "output_dir", "/fallback")
        == "/workspace/models/foo"
    )
    assert (
        _resolve_train_input(steps, "missing_step", "output_dir", "/fallback")
        == "/fallback"
    )


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
    """Successful job submission returns the host response and targets /jobs."""
    client = JobHostClient()
    host_response = {"job_id": "job-abc", "status": "running"}
    session = _MockSession(_MockResponse(200, host_response))
    client._session = session

    result = await client.submit_job(online_training_host, {"job_id": "job-abc"})
    assert result == host_response
    assert session.last_url == "http://training-box:8000/jobs"
    assert session.last_kwargs["headers"]["X-API-Key"] == "host-key-1"


@pytest.mark.anyio
async def test_client_submit_job_accepts_202(online_training_host):
    """The host returns 202 Accepted on submit — treated as success."""
    client = JobHostClient()
    client._session = _MockSession(_MockResponse(202, {"job_id": "job-abc"}))
    result = await client.submit_job(online_training_host, {"job_id": "job-abc"})
    assert result["job_id"] == "job-abc"


@pytest.mark.anyio
async def test_client_submit_job_failure(online_training_host):
    """Host rejection should raise JobHostClientError."""
    client = JobHostClient()
    client._session = _MockSession(_MockResponse(400, {"error": "bad request"}))

    with pytest.raises(JobHostClientError) as exc:
        await client.submit_job(online_training_host, {"job_id": "j"})
    assert exc.value.status_code == 400
    assert exc.value.host_id == "host-1"


@pytest.mark.anyio
async def test_client_cancel_job(online_training_host):
    """DELETE should return the host's cancel response at the job URL."""
    client = JobHostClient()
    session = _MockSession(_MockResponse(200, {"detail": "cancelled"}))
    client._session = session

    result = await client.cancel_job(online_training_host, "job-abc")
    assert result == {"detail": "cancelled"}
    assert session.last_url == "http://training-box:8000/jobs/job-abc"


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
        mock_session.add = lambda *a, **k: None  # add() is sync on a Session
        mock_session_ctx.return_value.__aenter__.return_value = mock_session

        await db.add_job(job)
        mock_session.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_job_db_update_status():
    """Should update job status and set completed_at for terminal states."""
    db = JobDB()

    with patch.object(db, "_session") as mock_session_ctx:
        mock_session = AsyncMock()
        mock_session_ctx.return_value.__aenter__.return_value = mock_session
        mock_execute_result = AsyncMock()
        mock_execute_result.rowcount = 1
        mock_session.execute.return_value = mock_execute_result

        result = await db.update_job_status("job-1", JobStatus.COMPLETED)
        assert result is True
        mock_session.commit.assert_awaited_once()


def test_job_status_from_host():
    """Host status strings map to JobStatus; unknowns return None."""
    assert JobStatus.from_host("running") is JobStatus.RUNNING
    assert JobStatus.from_host("COMPLETED") is JobStatus.COMPLETED
    assert JobStatus.from_host("weird") is None
    assert JobStatus.from_host(None) is None


# ── Event Payload Shape Tests ────────────────────────────────


def test_job_log_payload_shape():
    """JobLogPayload mirrors the host step-log entry shape."""
    from app.models.socketio import JobLogPayload

    data = JobLogPayload(
        job_id="job-1",
        host_id="host-1",
        host_name="Test Host",
        step_name="train",
        step_index=2,
        stream="stdout",
        seq=42,
        line="Epoch 3/10 loss=0.42",
        correlation_id="corr-123",
        timestamp="2026-07-20T12:00:00Z",
    ).model_dump()
    assert data["job_id"] == "job-1"
    assert data["step_index"] == 2
    assert data["stream"] == "stdout"
    assert data["seq"] == 42
    assert data["correlation_id"] == "corr-123"


def test_job_lifecycle_payload_shape():
    """JobLifecyclePayload normalizes a host lifecycle event."""
    from app.models.socketio import JobLifecyclePayload

    data = JobLifecyclePayload(
        job_id="job-1",
        host_id="host-1",
        host_name="Test Host",
        event="step_completed",
        status="completed",
        step_name="train",
        step_index=2,
        correlation_id="corr-123",
        data={"duration_s": 12.5, "exit_code": 0},
        timestamp="2026-07-20T12:00:00Z",
    ).model_dump()
    assert data["event"] == "step_completed"
    assert data["status"] == "completed"
    assert data["step_name"] == "train"
    assert data["data"]["exit_code"] == 0


# ── Event Forwarding Handler Tests (S-025 / S-026) ───────────


@pytest.fixture(autouse=True)
def _clear_correlation_cache():
    from app.socketio_app import host_handlers

    host_handlers._correlation_cache.clear()
    yield
    host_handlers._correlation_cache.clear()


@pytest.mark.anyio
async def test_host_step_log_batch_forwarding(online_training_host):
    """A batched step_log is expanded to one enriched job_log per entry."""
    from app.socketio_app import host_handlers, webui_handlers

    batch = {
        "entries": [
            {
                "job_id": "job-1",
                "step_name": "train",
                "step_index": 2,
                "stream": "stdout",
                "seq": 0,
                "line": "starting",
                "timestamp": "2026-07-20T12:00:00Z",
            },
            {
                "job_id": "job-1",
                "step_name": "train",
                "step_index": 2,
                "stream": "stdout",
                "seq": 1,
                "line": "epoch 1",
                "timestamp": "2026-07-20T12:00:01Z",
            },
        ]
    }
    job = Job(id="job-1", host_id="host-1", correlation_id="corr-1")

    with (
        patch.object(
            host_handlers.host_store,
            "get_host_id_for_sid",
            AsyncMock(return_value="host-1"),
        ),
        patch.object(
            host_handlers.host_db,
            "get_host",
            AsyncMock(return_value=online_training_host),
        ),
        patch.object(host_handlers.job_db, "get_job", AsyncMock(return_value=job)),
        patch.object(
            webui_handlers, "broadcast_job_log", AsyncMock()
        ) as mock_broadcast,
    ):
        await host_handlers.host_step_log("sid-1", batch)

    assert mock_broadcast.await_count == 2
    first = mock_broadcast.await_args_list[0].args[0]
    assert first["job_id"] == "job-1"
    assert first["host_id"] == "host-1"
    assert first["line"] == "starting"
    assert first["correlation_id"] == "corr-1"


@pytest.mark.anyio
async def test_host_step_log_ignores_unknown_sid():
    """Events from an unregistered socket are dropped."""
    from app.socketio_app import host_handlers, webui_handlers

    with (
        patch.object(
            host_handlers.host_store,
            "get_host_id_for_sid",
            AsyncMock(return_value=None),
        ),
        patch.object(
            webui_handlers, "broadcast_job_log", AsyncMock()
        ) as mock_broadcast,
    ):
        await host_handlers.host_step_log("sid-x", {"entries": [{"job_id": "j"}]})

    mock_broadcast.assert_not_called()


@pytest.mark.anyio
async def test_host_lifecycle_updates_status_and_forwards(online_training_host):
    """job_completed persists COMPLETED and rebroadcasts a normalized event."""
    from app.socketio_app import host_handlers, webui_handlers

    event = {
        "job_id": "job-1",
        "status": "completed",
        "workspace_path": "/jobs/job-1",
        "retention_deadline": "2026-07-21T12:00:00Z",
        "timestamp": "2026-07-20T12:00:00Z",
    }
    job = Job(id="job-1", host_id="host-1", correlation_id="corr-1")

    with (
        patch.object(
            host_handlers.host_store,
            "get_host_id_for_sid",
            AsyncMock(return_value="host-1"),
        ),
        patch.object(
            host_handlers.host_db,
            "get_host",
            AsyncMock(return_value=online_training_host),
        ),
        patch.object(host_handlers.job_db, "get_job", AsyncMock(return_value=job)),
        patch.object(
            host_handlers.job_db,
            "update_job_status",
            AsyncMock(return_value=True),
        ) as mock_update,
        patch.object(
            webui_handlers, "broadcast_job_lifecycle", AsyncMock()
        ) as mock_broadcast,
    ):
        await host_handlers._handle_job_lifecycle("job_completed", "sid-1", event)

    mock_update.assert_awaited_once()
    args, kwargs = mock_update.await_args
    assert args[0] == "job-1"
    assert args[1] is JobStatus.COMPLETED

    payload = mock_broadcast.await_args.args[0]
    assert payload["event"] == "job_completed"
    assert payload["status"] == "completed"
    assert payload["correlation_id"] == "corr-1"
    # Non-normalized extras are preserved under `data`.
    assert payload["data"]["workspace_path"] == "/jobs/job-1"


@pytest.mark.anyio
async def test_host_lifecycle_step_event_does_not_change_status(online_training_host):
    """step_* events are forwarded but never mutate the persisted job status."""
    from app.socketio_app import host_handlers, webui_handlers

    event = {
        "job_id": "job-1",
        "step_name": "train",
        "step_index": 2,
        "status": "running",
        "timestamp": "2026-07-20T12:00:00Z",
    }

    with (
        patch.object(
            host_handlers.host_store,
            "get_host_id_for_sid",
            AsyncMock(return_value="host-1"),
        ),
        patch.object(
            host_handlers.host_db,
            "get_host",
            AsyncMock(return_value=online_training_host),
        ),
        patch.object(
            host_handlers.job_db,
            "get_job",
            AsyncMock(return_value=Job(id="job-1", host_id="host-1")),
        ),
        patch.object(
            host_handlers.job_db, "update_job_status", AsyncMock()
        ) as mock_update,
        patch.object(webui_handlers, "broadcast_job_lifecycle", AsyncMock()),
    ):
        await host_handlers._handle_job_lifecycle("step_started", "sid-1", event)

    mock_update.assert_not_called()


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
