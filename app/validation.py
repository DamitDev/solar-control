"""Shared validation helpers (S-036, S-040).

Centralized validators used by route handlers and services so
priorities, constraints, and error formats stay consistent across
the codebase.
"""

from typing import Any
from urllib.parse import urlparse

from fastapi import HTTPException

VALID_PRIORITIES: frozenset[str] = frozenset({"production", "staging", "ephemeral"})
VALID_STRATEGIES: frozenset[str] = frozenset({"rolling", "immediate"})
VALID_BACKEND_TYPES: frozenset[str] = frozenset(
    {
        "llamacpp",
        "huggingface_causal",
        "huggingface_classification",
        "huggingface_embedding",
        "huggingface_vision",
    }
)
VALID_MODEL_SOURCE_SCHEMES: frozenset[str] = frozenset({"repo", "huggingface", "local"})
FORBIDDEN_BACKEND_FIELDS: frozenset[str] = frozenset(
    {
        "alias",
        "model_source",
        "host",
        "port",
        "api_key",
    }
)


def validate_priority(instance_data: dict[str, Any]) -> None:
    """Validate the priority field if present (S-036)."""
    priority = instance_data.get("priority")
    if priority is not None and priority not in VALID_PRIORITIES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid priority '{priority}'. "
                f"Must be one of: {', '.join(sorted(VALID_PRIORITIES))}"
            ),
        )


def validate_intent_create(data: dict[str, Any]) -> list[dict[str, str]]:
    """Validate an intent creation request (S-039 §4.7).

    Returns a list of {field, message} errors. Empty list means valid.
    Does NOT raise — the route handler decides the HTTP status.
    """
    errors: list[dict[str, str]] = []

    # alias
    alias = data.get("alias")
    if not alias or not isinstance(alias, str) or not alias.strip():
        errors.append(
            {"field": "alias", "message": "alias is required and must be non-empty"}
        )

    # model_source
    model_source = data.get("model_source", "")
    if not model_source:
        errors.append({"field": "model_source", "message": "model_source is required"})
    else:
        parsed = urlparse(model_source)
        if parsed.scheme not in VALID_MODEL_SOURCE_SCHEMES:
            errors.append(
                {
                    "field": "model_source",
                    "message": (
                        f"unsupported scheme '{parsed.scheme}'. "
                        f"Must be one of: {', '.join(sorted(VALID_MODEL_SOURCE_SCHEMES))}"
                    ),
                }
            )

    # replicas
    replicas = data.get("replicas", 1)
    if not isinstance(replicas, int) or replicas < 0:
        errors.append({"field": "replicas", "message": "replicas must be >= 0"})

    # priority
    priority = data.get("priority", "production")
    if priority not in VALID_PRIORITIES:
        errors.append(
            {
                "field": "priority",
                "message": (
                    f"'{priority}' is not a valid priority. "
                    f"Must be one of: {', '.join(sorted(VALID_PRIORITIES))}"
                ),
            }
        )

    # strategy
    strategy = data.get("strategy", "rolling")
    if strategy not in VALID_STRATEGIES:
        errors.append(
            {
                "field": "strategy",
                "message": (
                    f"'{strategy}' is not a valid strategy. "
                    f"Must be one of: {', '.join(sorted(VALID_STRATEGIES))}"
                ),
            }
        )

    # backend
    backend = data.get("backend", {})
    if not isinstance(backend, dict):
        errors.append({"field": "backend", "message": "backend must be an object"})
    else:
        backend_type = backend.get("backend_type")
        if not backend_type:
            errors.append(
                {"field": "backend.backend_type", "message": "backend_type is required"}
            )
        elif backend_type not in VALID_BACKEND_TYPES:
            errors.append(
                {
                    "field": "backend.backend_type",
                    "message": (
                        f"'{backend_type}' is not a supported backend_type. "
                        f"Must be one of: {', '.join(sorted(VALID_BACKEND_TYPES))}"
                    ),
                }
            )

        # Forbidden fields
        for forbidden in FORBIDDEN_BACKEND_FIELDS:
            if forbidden in backend:
                errors.append(
                    {
                        "field": f"backend.{forbidden}",
                        "message": f"'{forbidden}' is server-derived and must not be set by the client",
                    }
                )

    # placement
    placement = data.get("placement", {})
    if isinstance(placement, dict):
        roles = placement.get("roles", ["inference"])
        if not isinstance(roles, list) or len(roles) == 0:
            errors.append(
                {
                    "field": "placement.roles",
                    "message": "placement.roles must be a non-empty list",
                }
            )

    return errors
