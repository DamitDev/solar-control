"""Tests for intent API (S-040)."""

import pytest
from unittest.mock import AsyncMock, patch


from app.models.intent import (
    IntentCreate,
    IntentPhase,
    IntentResponse,
    IntentStatus,
    PlacementConstraints,
    ReconcileState,
    ResourceRequirements,
)
from app.validation import validate_intent_create

# ── Validation unit tests ──────────────────────────────────────


def test_validate_intent_create_valid_minimal():
    """Minimal valid intent passes validation."""
    data = {
        "alias": "test-model",
        "model_source": "repo://test:v1",
        "backend": {"backend_type": "huggingface_classification"},
    }
    errors = validate_intent_create(data)
    assert errors == []


def test_validate_intent_create_valid_full():
    """Full intent with all optional fields passes."""
    data = {
        "alias": "test-model",
        "model_source": "huggingface://org/model",
        "replicas": 3,
        "priority": "staging",
        "strategy": "immediate",
        "backend": {
            "backend_type": "llamacpp",
            "dtype": "float16",
            "max_length": 512,
        },
        "placement": {
            "roles": ["inference"],
            "gpu_type": "nvidia_cuda",
            "host_allow": ["h1"],
            "host_deny": ["h2"],
        },
        "resources": {"vram_gb": 8.0, "ram_gb": 16.0},
        "metadata": {"source": "supernova"},
    }
    errors = validate_intent_create(data)
    assert errors == []


def test_validate_intent_missing_alias():
    errors = validate_intent_create(
        {
            "model_source": "repo://x:v1",
            "backend": {"backend_type": "llamacpp"},
        }
    )
    assert any(e["field"] == "alias" for e in errors)


def test_validate_intent_missing_model_source():
    errors = validate_intent_create(
        {
            "alias": "x",
            "backend": {"backend_type": "llamacpp"},
        }
    )
    assert any(e["field"] == "model_source" for e in errors)


def test_validate_intent_invalid_scheme():
    errors = validate_intent_create(
        {
            "alias": "x",
            "model_source": "http://example.com/model",
            "backend": {"backend_type": "llamacpp"},
        }
    )
    assert any(e["field"] == "model_source" for e in errors)


def test_validate_intent_local_scheme():
    """local:// is a valid scheme."""
    errors = validate_intent_create(
        {
            "alias": "x",
            "model_source": "local:///opt/models/model.gguf",
            "backend": {"backend_type": "llamacpp"},
        }
    )
    assert errors == []


def test_validate_intent_huggingface_scheme():
    """huggingface:// is a valid scheme."""
    errors = validate_intent_create(
        {
            "alias": "x",
            "model_source": "huggingface://meta-llama/Llama-2-7b-hf",
            "backend": {"backend_type": "llamacpp"},
        }
    )
    assert errors == []


def test_validate_intent_negative_replicas():
    errors = validate_intent_create(
        {
            "alias": "x",
            "model_source": "repo://x:v1",
            "replicas": -1,
            "backend": {"backend_type": "llamacpp"},
        }
    )
    assert any(e["field"] == "replicas" for e in errors)


def test_validate_intent_zero_replicas():
    """replicas=0 is valid (pre-create then scale up)."""
    errors = validate_intent_create(
        {
            "alias": "x",
            "model_source": "repo://x:v1",
            "replicas": 0,
            "backend": {"backend_type": "llamacpp"},
        }
    )
    assert errors == []


def test_validate_intent_invalid_priority():
    errors = validate_intent_create(
        {
            "alias": "x",
            "model_source": "repo://x:v1",
            "priority": "critical",
            "backend": {"backend_type": "llamacpp"},
        }
    )
    assert any(e["field"] == "priority" for e in errors)


def test_validate_intent_all_valid_priorities():
    """All three valid priorities pass."""
    for p in ["production", "staging", "ephemeral"]:
        errors = validate_intent_create(
            {
                "alias": "x",
                "model_source": "repo://x:v1",
                "priority": p,
                "backend": {"backend_type": "llamacpp"},
            }
        )
        assert errors == [], f"Priority '{p}' should be valid"


def test_validate_intent_invalid_strategy():
    errors = validate_intent_create(
        {
            "alias": "x",
            "model_source": "repo://x:v1",
            "strategy": "blue-green",
            "backend": {"backend_type": "llamacpp"},
        }
    )
    assert any(e["field"] == "strategy" for e in errors)


def test_validate_intent_all_valid_strategies():
    """Both valid strategies pass."""
    for s in ["rolling", "immediate"]:
        errors = validate_intent_create(
            {
                "alias": "x",
                "model_source": "repo://x:v1",
                "strategy": s,
                "backend": {"backend_type": "llamacpp"},
            }
        )
        assert errors == [], f"Strategy '{s}' should be valid"


def test_validate_intent_missing_backend_type():
    errors = validate_intent_create(
        {
            "alias": "x",
            "model_source": "repo://x:v1",
            "backend": {},
        }
    )
    assert any(e["field"] == "backend.backend_type" for e in errors)


def test_validate_intent_invalid_backend_type():
    errors = validate_intent_create(
        {
            "alias": "x",
            "model_source": "repo://x:v1",
            "backend": {"backend_type": "unknown_type"},
        }
    )
    assert any(e["field"] == "backend.backend_type" for e in errors)


def test_validate_intent_all_valid_backend_types():
    """All five valid backend types pass."""
    for bt in [
        "llamacpp",
        "huggingface_causal",
        "huggingface_classification",
        "huggingface_embedding",
        "huggingface_vision",
    ]:
        errors = validate_intent_create(
            {
                "alias": "x",
                "model_source": "repo://x:v1",
                "backend": {"backend_type": bt},
            }
        )
        assert errors == [], f"backend_type '{bt}' should be valid"


def test_validate_intent_forbidden_backend_fields():
    """Server-derived fields must not appear in backend."""
    for forbidden in ["alias", "model_source", "host", "port", "api_key"]:
        errors = validate_intent_create(
            {
                "alias": "x",
                "model_source": "repo://x:v1",
                "backend": {"backend_type": "llamacpp", forbidden: "value"},
            }
        )
        assert any(
            e["field"] == f"backend.{forbidden}" for e in errors
        ), f"Expected error for backend.{forbidden}"


def test_validate_intent_empty_placement_roles():
    errors = validate_intent_create(
        {
            "alias": "x",
            "model_source": "repo://x:v1",
            "backend": {"backend_type": "llamacpp"},
            "placement": {"roles": []},
        }
    )
    assert any(e["field"] == "placement.roles" for e in errors)


# ── Model unit tests ────────────────────────────────────────────


def test_intent_create_defaults():
    """IntentCreate applies correct defaults."""
    intent = IntentCreate(
        alias="m",
        model_source="repo://m:v1",
        backend={"backend_type": "llamacpp"},
    )
    assert intent.replicas == 1
    assert intent.priority == "production"
    assert intent.strategy == "rolling"
    assert intent.placement.roles == ["inference"]
    assert intent.resources.vram_gb is None


def test_intent_status_defaults():
    """IntentStatus has correct new-intent defaults."""
    status = IntentStatus()
    assert status.phase == IntentPhase.PENDING
    assert status.reconcile == ReconcileState.IDLE
    assert status.observed_replicas == 0
    assert status.ready_replicas == 0
    assert status.available is False


# ── Route integration tests (mock IntentDB) ────────────────────


@pytest.fixture
def valid_intent_create() -> IntentCreate:
    return IntentCreate(
        alias="test-model",
        model_source="repo://test:v1",
        replicas=2,
        priority="production",
        strategy="rolling",
        backend={"backend_type": "huggingface_classification", "max_length": 512},
        placement=PlacementConstraints(roles=["inference"], gpu_type="nvidia_cuda"),
        resources=ResourceRequirements(vram_gb=6.0),
    )


@pytest.fixture
def mock_intent_response() -> IntentResponse:
    return IntentResponse(
        id="550e8400-e29b-41d4-a716-446655440000",
        alias="test-model",
        model_source="repo://test:v1",
        replicas=2,
        priority="production",
        strategy="rolling",
        backend={"backend_type": "huggingface_classification", "max_length": 512},
        placement=PlacementConstraints(roles=["inference"], gpu_type="nvidia_cuda"),
        resources=ResourceRequirements(vram_gb=6.0),
        metadata={},
        status=IntentStatus(
            phase=IntentPhase.PENDING,
            reconcile=ReconcileState.IDLE,
            desired_replicas=2,
            observed_replicas=0,
            ready_replicas=0,
            updated_replicas=0,
            available=False,
            shortfall=0,
            created_at="2026-07-24T00:00:00Z",
            updated_at="2026-07-24T00:00:00Z",
        ),
    )


@pytest.mark.anyio
async def test_create_intent_success(valid_intent_create, mock_intent_response):
    """POST /api/intents returns 201 with pending status."""
    from fastapi.testclient import TestClient

    with (
        patch(
            "app.routes.management.intents.intent_db.create_intent",
            new=AsyncMock(return_value=mock_intent_response),
        ),
        patch(
            "app.routes.management.intents.intent_db.check_alias_conflict",
            new=AsyncMock(return_value=False),
        ),
    ):
        from app.main import app

        client = TestClient(app)
        response = client.post(
            "/api/intents",
            json=valid_intent_create.model_dump(),
            headers={"X-API-Key": "change-me-management"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == mock_intent_response.id
        assert data["alias"] == "test-model"
        assert data["status"]["phase"] == "pending"
        assert data["status"]["reconcile"] == "idle"


@pytest.mark.anyio
async def test_create_intent_alias_conflict(valid_intent_create):
    """POST with duplicate alias returns 409."""
    with patch(
        "app.routes.management.intents.intent_db.check_alias_conflict",
        new=AsyncMock(return_value=True),
    ):
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/api/intents",
            json=valid_intent_create.model_dump(),
            headers={"X-API-Key": "change-me-management"},
        )

        assert response.status_code == 409


@pytest.mark.anyio
async def test_create_intent_validation_error():
    """POST with invalid data returns 422."""
    with patch(
        "app.routes.management.intents.intent_db.check_alias_conflict",
        new=AsyncMock(return_value=False),
    ):
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post(
            "/api/intents",
            json={
                "alias": "valid-alias",
                "model_source": "http://bad",
                "backend": {"backend_type": "invalid_type"},
            },
            headers={"X-API-Key": "change-me-management"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "errors" in data["detail"]
        assert len(data["detail"]["errors"]) >= 2  # bad scheme + bad backend_type


@pytest.mark.anyio
async def test_create_intent_unauthorized(valid_intent_create):
    """POST without API key returns 401."""
    from app.main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post(
        "/api/intents",
        json=valid_intent_create.model_dump(),
    )

    assert response.status_code == 401


@pytest.mark.anyio
async def test_list_intents(mock_intent_response):
    """GET /api/intents returns list."""
    with patch(
        "app.routes.management.intents.intent_db.list_intents",
        new=AsyncMock(return_value=[mock_intent_response]),
    ):
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get(
            "/api/intents",
            headers={"X-API-Key": "change-me-management"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == mock_intent_response.id


@pytest.mark.anyio
async def test_get_intent_found(mock_intent_response):
    """GET /api/intents/{id} returns the intent."""
    with patch(
        "app.routes.management.intents.intent_db.get_intent",
        new=AsyncMock(return_value=mock_intent_response),
    ):
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get(
            "/api/intents/550e8400-e29b-41d4-a716-446655440000",
            headers={"X-API-Key": "change-me-management"},
        )

        assert response.status_code == 200
        assert response.json()["id"] == mock_intent_response.id


@pytest.mark.anyio
async def test_get_intent_not_found():
    """GET /api/intents/{id} with unknown ID returns 404."""
    with patch(
        "app.routes.management.intents.intent_db.get_intent",
        new=AsyncMock(return_value=None),
    ):
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get(
            "/api/intents/nonexistent",
            headers={"X-API-Key": "change-me-management"},
        )

        assert response.status_code == 404


@pytest.mark.anyio
async def test_delete_intent_success(mock_intent_response):
    """DELETE /api/intents/{id} returns 202 with deleting phase."""
    mock_intent_response.status.phase = IntentPhase.DELETING
    with patch(
        "app.routes.management.intents.intent_db.soft_delete_intent",
        new=AsyncMock(return_value=mock_intent_response),
    ):
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.delete(
            "/api/intents/550e8400-e29b-41d4-a716-446655440000",
            headers={"X-API-Key": "change-me-management"},
        )

        assert response.status_code == 202
        data = response.json()
        assert data["phase"] == "deleting"


@pytest.mark.anyio
async def test_delete_intent_not_found():
    """DELETE with unknown ID returns 404."""
    with patch(
        "app.routes.management.intents.intent_db.soft_delete_intent",
        new=AsyncMock(return_value=None),
    ):
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.delete(
            "/api/intents/nonexistent",
            headers={"X-API-Key": "change-me-management"},
        )

        assert response.status_code == 404


@pytest.mark.anyio
async def test_delete_intent_with_orphan(mock_intent_response):
    """DELETE with ?orphan=true returns 202 with orphan message."""
    mock_intent_response.status.phase = IntentPhase.DELETING
    with patch(
        "app.routes.management.intents.intent_db.soft_delete_intent",
        new=AsyncMock(return_value=mock_intent_response),
    ):
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.delete(
            "/api/intents/550e8400-e29b-41d4-a716-446655440000?orphan=true",
            headers={"X-API-Key": "change-me-management"},
        )

        assert response.status_code == 202
        data = response.json()
        assert "orphaned" in data["message"].lower()


@pytest.mark.anyio
async def test_list_intents_with_filters(mock_intent_response):
    """GET /api/intents passes query params to list_intents."""
    mock_list = AsyncMock(return_value=[mock_intent_response])
    with patch(
        "app.routes.management.intents.intent_db.list_intents",
        new=mock_list,
    ):
        from app.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get(
            "/api/intents?priority=production&phase=pending&limit=10&offset=0",
            headers={"X-API-Key": "change-me-management"},
        )

        assert response.status_code == 200
        mock_list.assert_called_once_with(
            alias=None,
            priority="production",
            phase="pending",
            limit=10,
            offset=0,
        )
