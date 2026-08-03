"""Shared suite constants for the D-017 integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MANAGEMENT_API_KEY = "test-mgmt-key"
DATA_REPOSITORY_API_KEY = "repo-key"
# Distinct per-host keys: control looks hosts up by api_key
# (get_host_by_api_key → scalar_one_or_none) — a shared key between the
# two registered host rows crashes the WS connect handler.
HOST_A_API_KEY = "test-host-a-key"
HOST_B_API_KEY = "test-host-b-key"
HARBOR_USERNAME = "robot$test"
HARBOR_PASSWORD = "test"

MODEL_NAME = "test-model"
MODEL_VERSION = "v1"
MODEL_SOURCE_URI = f"repo://{MODEL_NAME}:{MODEL_VERSION}"
MODEL_ALIAS = "test-classifier"

# Backend payload for the tiny HF classification model. Used both for
# intent ``backend`` dicts and imperative instance creation.
BACKEND_CLASSIFICATION: dict[str, Any] = {
    "backend_type": "huggingface_classification",
    "device": "cpu",
    "dtype": "float32",
    "max_length": 128,
    "labels": [f"LABEL_{i}" for i in range(5)],
}

FIXTURE_MODEL_DIR = Path(__file__).resolve().parent / "test_model"


def harbor_port(harbor_ref: str) -> str:
    """Extract the port from a ``host:port/repo:tag`` Harbor ref.

    NB: do NOT split the whole ref on ':' — the tag after the last colon
    would win. Split the host part first.
    """
    return harbor_ref.split("/")[0].split(":")[-1]
