"""Shared pytest fixtures for the solar-control test suite."""

from unittest.mock import patch

import pytest


@pytest.fixture
def repo_settings():
    """Patch ``app.model_resolvers.repo.settings`` with sensible defaults.

    Yields the mock so tests can override individual fields (e.g. unset
    ``data_repository_url`` to exercise the unconfigured-config path).
    """
    with patch("app.model_resolvers.repo.settings") as mock_settings:
        mock_settings.data_repository_url = "http://data-repo:8000"
        mock_settings.data_repository_api_key = ""
        mock_settings.data_repository_timeout_s = 10.0
        yield mock_settings
