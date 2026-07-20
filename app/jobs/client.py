"""HTTP client for proxying job requests to Solar Hosts.

Thin wrapper around ``aiohttp`` that calls the host's ``POST /jobs``,
``GET /jobs/{id}``, and ``DELETE /jobs/{id}`` endpoints.
"""

import asyncio
import logging
from typing import Any

import aiohttp

from app.config import settings
from app.models import Host

logger = logging.getLogger(__name__)


class JobHostClientError(Exception):
    """Raised when the host returns a non-success response."""

    def __init__(
        self,
        message: str,
        status_code: int,
        host_id: str,
        host_name: str,
        body: Any = None,
    ) -> None:
        self.status_code = status_code
        self.host_id = host_id
        self.host_name = host_name
        self.body = body
        super().__init__(message)


class JobHostClient:
    """HTTP client for Solar Host job endpoints.

    Each method takes a :class:`Host` and forwards the request to
    ``{host.url}/jobs`` with the host's API key for authentication.
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is not None and not self._session.closed:
            return self._session
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def submit_job(self, host: Host, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit a job to ``POST {host.url}/jobs``.

        Returns the JSON response body on success.

        Raises
        ------
        JobHostClientError
            If the host returns a non-2xx status.
        aiohttp.ClientError
            If a connection-level error occurs.
        """
        session = await self._ensure_session()
        url = f"{host.url.rstrip('/')}/jobs"
        headers = {
            "X-API-Key": host.api_key,
            "Content-Type": "application/json",
        }

        logger.info(
            "Submitting job to host '%s' (%s) — POST %s",
            host.name,
            host.id,
            url,
        )

        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=settings.job_submission_timeout_s),
        ) as response:
            body = await _read_body(response)

            if response.status < 200 or response.status >= 300:
                logger.warning(
                    "Host '%s' (%s) returned HTTP %d: %s",
                    host.name,
                    host.id,
                    response.status,
                    body,
                )
                raise JobHostClientError(
                    message=(f"Host '{host.name}' returned HTTP {response.status}"),
                    status_code=response.status,
                    host_id=host.id,
                    host_name=host.name,
                    body=body,
                )

            logger.info(
                "Job submitted to host '%s' (%s) — HTTP %d",
                host.name,
                host.id,
                response.status,
            )
            return body

    async def get_job_status(self, host: Host, job_id: str) -> dict[str, Any]:
        """Get job status from ``GET {host.url}/jobs/{job_id}``."""
        session = await self._ensure_session()
        url = f"{host.url.rstrip('/')}/jobs/{job_id}"
        headers = {"X-API-Key": host.api_key}

        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10.0),
        ) as response:
            body = await _read_body(response)

            if response.status == 404:
                raise JobHostClientError(
                    message=f"Job '{job_id}' not found on host '{host.name}'",
                    status_code=404,
                    host_id=host.id,
                    host_name=host.name,
                    body=body,
                )
            if response.status < 200 or response.status >= 300:
                raise JobHostClientError(
                    message=(
                        f"Host '{host.name}' returned HTTP {response.status}"
                        f" for job '{job_id}'"
                    ),
                    status_code=response.status,
                    host_id=host.id,
                    host_name=host.name,
                    body=body,
                )

            return body

    async def cancel_job(self, host: Host, job_id: str) -> dict[str, Any]:
        """Cancel a job via ``DELETE {host.url}/jobs/{job_id}``."""
        session = await self._ensure_session()
        url = f"{host.url.rstrip('/')}/jobs/{job_id}"
        headers = {"X-API-Key": host.api_key}

        logger.info(
            "Cancelling job '%s' on host '%s' (%s) — DELETE %s",
            job_id,
            host.name,
            host.id,
            url,
        )

        async with session.delete(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10.0),
        ) as response:
            body = await _read_body(response)

            if response.status < 200 or response.status >= 300:
                logger.warning(
                    "Cancel job '%s' on host '%s' returned HTTP %d: %s",
                    job_id,
                    host.name,
                    response.status,
                    body,
                )
                raise JobHostClientError(
                    message=(
                        f"Cancel job '{job_id}' on host '{host.name}' "
                        f"returned HTTP {response.status}"
                    ),
                    status_code=response.status,
                    host_id=host.id,
                    host_name=host.name,
                    body=body,
                )

            return body


async def _read_body(response: aiohttp.ClientResponse) -> Any:
    """Try to read response as JSON, fall back to text."""
    try:
        return await response.json()
    except Exception:
        return await response.text()


# Module-level singleton
job_client = JobHostClient()
