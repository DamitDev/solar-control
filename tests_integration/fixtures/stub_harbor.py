"""Stub Harbor — OCI Distribution v2 HTTP server for integration tests.

Speaks just enough of the OCI Distribution spec for BOTH real clients used
by the platform:

- ``harbor_oci_client.HarborClient.verify_artifact`` (data-repository):
  ``GET /service/token`` (Basic auth) -> bearer token, then
  ``HEAD /v2/{repo}/manifests/{ref}`` with ``Accept`` + ``Docker-Content-Digest``.
- ``harbor_oci_client.OrasHelper`` (solar-host, wraps oras-py with the
  ``token`` auth backend): unauthenticated request -> ``401`` with
  ``Www-Authenticate: Bearer realm=...`` -> token fetch (Basic auth) ->
  retry with bearer token. Then ``GET /v2/{repo}/manifests/{ref}`` and
  ``GET /v2/{repo}/blobs/{digest}``.

The stub runs in a background thread (``ThreadingHTTPServer``) on a random
loopback port. It records every request it receives so tests can assert
*who* called Harbor and how often (``received_requests()``).

Artifacts are registered with ``register_model(ref, files)`` where ``files``
is ``{filename: bytes}``. Each file becomes a flat OCI layer with
``org.opencontainers.image.title`` set to the filename and a real
``sha256:`` digest of the file bytes — exactly what ``OrasHelper.pull``
writes to disk (one file per layer, no tarball).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_MEDIA_TYPE = "application/vnd.oci.image.manifest.v1+json"
LAYER_MEDIA_TYPE = "application/vnd.oci.image.layer.v1.tar"  # NOT +gzip (direct write)
CONFIG_MEDIA_TYPE = "application/vnd.unknown.config.v1+json"
TITLE_ANNOTATION = "org.opencontainers.image.title"

# Blank config blob ("{}"): digest + size known from the OCI spec.
_BLANK_CONFIG_DIGEST = (
    "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
)
_BLANK_CONFIG_SIZE = 2

_TOKEN = "stub-harbor-test-token"
_SERVICE = "harbor-registry"

# repo ref path pattern: /v2/{repository}/manifests|blobs/{reference}
_PATH_RE = re.compile(
    r"^/v2/(?P<repo>[^/]+(?:/[^/]+)*)/(?P<kind>manifests|blobs)/(?P<ref>[^/]+)$"
)


def sha256_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class _StubHarborState:
    """Mutable state shared with the request handler."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # repo -> {reference: manifest_dict}
        self.manifests: dict[str, dict[str, dict[str, Any]]] = {}
        # digest -> bytes
        self.blobs: dict[str, bytes] = {}
        # (method, path, headers-dict) log
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.base_url = ""
        # Optional on-disk mirror of the request log (for debugging hangs).
        self.log_file: str = ""

    def record(self, method: str, path: str, headers: dict[str, str]) -> None:
        with self._lock:
            self.requests.append((method, path, dict(headers)))
        if self.log_file:
            try:
                auth = headers.get("Authorization", "")[:12]
                with open(self.log_file, "a") as f:
                    f.write(f"{method} {path} auth={auth}\n")
            except Exception:
                pass

    def received_requests(self) -> list[tuple[str, str, dict[str, str]]]:
        with self._lock:
            return list(self.requests)

    def reset(self) -> None:
        with self._lock:
            self.requests = []

    def register_model(
        self, harbor_ref: str, files: dict[str, bytes]
    ) -> dict[str, Any]:
        """Register an artifact. ``harbor_ref`` e.g. ``127.0.0.1:PORT/supernova/test-model:v1``.

        Returns the manifest dict (with its digest under ``_digest``).
        """
        repo, reference = split_ref(harbor_ref)
        layers = []
        for filename, data in files.items():
            digest = sha256_digest(data)
            with self._lock:
                self.blobs[digest] = data
            layers.append(
                {
                    "mediaType": LAYER_MEDIA_TYPE,
                    "digest": digest,
                    "size": len(data),
                    "annotations": {TITLE_ANNOTATION: filename},
                }
            )
        manifest = {
            "schemaVersion": 2,
            "mediaType": MANIFEST_MEDIA_TYPE,
            "config": {
                "mediaType": CONFIG_MEDIA_TYPE,
                "digest": _BLANK_CONFIG_DIGEST,
                "size": _BLANK_CONFIG_SIZE,
            },
            "layers": layers,
        }
        with self._lock:
            self.blobs[_BLANK_CONFIG_DIGEST] = b"{}"
            self.manifests.setdefault(repo, {})[reference] = manifest
        return manifest

    def get_manifest(self, repo: str, reference: str) -> dict[str, Any] | None:
        with self._lock:
            return self.manifests.get(repo, {}).get(reference)

    def get_blob(self, digest: str) -> bytes | None:
        with self._lock:
            return self.blobs.get(digest)


def split_ref(harbor_ref: str) -> tuple[str, str]:
    """Split ``host/repo:tag`` (or ``host/repo@digest``) into (repo, reference).

    The host part is stripped; the stub serves any host as long as the
    repository+reference path matches.
    """
    # Strip scheme if present
    if "://" in harbor_ref:
        harbor_ref = harbor_ref.split("://", 1)[1]
    rest = harbor_ref.split("/", 1)[1] if "/" in harbor_ref else harbor_ref
    if "@" in rest:
        repo, reference = rest.split("@", 1)
    elif ":" in rest:
        repo, reference = rest.rsplit(":", 1)
    else:
        repo, reference = rest, "latest"
    return repo, reference


class StubHarborHandler(BaseHTTPRequestHandler):
    """Minimal OCI Distribution handler. State lives on ``self.server.state``."""

    protocol_version = "HTTP/1.1"
    server_version = "StubHarbor/1.0"

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

    @property
    def state(self) -> _StubHarborState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("stub-harbor: " + format, *args)

    def _send(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(
        self, status: int, obj: Any, extra: dict[str, str] | None = None
    ) -> None:
        body = json.dumps(obj).encode()
        headers = {"Content-Type": "application/json"}
        if extra:
            headers.update(extra)
        self._send(status, body, headers)

    # ------------------------------------------------------------------
    # auth
    # ------------------------------------------------------------------

    def _has_bearer(self) -> bool:
        auth = self.headers.get("Authorization", "")
        return auth.startswith("Bearer ") and auth[7:] == _TOKEN

    def _has_basic(self) -> bool:
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(auth[6:]).decode()
        except Exception:
            return False
        # Accept any non-empty user:password (robot-style creds).
        return ":" in decoded and len(decoded) > 1

    def _challenge(self, scope: str) -> None:
        """401 with a Docker-style bearer challenge (oras-py token dance)."""
        realm = f"{self.state.base_url}/service/token"
        self._send(
            401,
            b"",
            {
                "Content-Type": "application/json",
                "Www-Authenticate": (
                    f'Bearer realm="{realm}",service="{_SERVICE}",scope="{scope}"'
                ),
            },
        )

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("HEAD")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        self.state.record(method, self.path, dict(self.headers))
        path = self.path.split("?", 1)[0]

        if path == "/service/token":
            self._handle_token()
            return

        if path == "/v2/":
            # Capability probe — challenge like a real registry.
            self._challenge("registry:catalog:*")
            return

        m = _PATH_RE.match(path)
        if not m:
            self._send(
                404,
                b'{"errors":[{"code":"UNSUPPORTED","message":"not found"}]}',
                {"Content-Type": "application/json"},
            )
            return

        if not self._has_bearer():
            self._challenge(f"repository:{m.group('repo')}:pull")
            return

        repo, kind, ref = m.group("repo"), m.group("kind"), m.group("ref")

        if kind == "manifests":
            manifest = self.state.get_manifest(repo, ref)
            if manifest is None:
                self._send(
                    404,
                    b'{"errors":[{"code":"MANIFEST_UNKNOWN","message":"manifest unknown"}]}',
                    {"Content-Type": "application/json"},
                )
                return
            body = json.dumps(manifest).encode()
            headers = {
                "Content-Type": MANIFEST_MEDIA_TYPE,
                "Docker-Content-Digest": sha256_digest(body),
            }
            self._send(200, body, headers)
            return

        if kind == "blobs":
            blob = self.state.get_blob(ref)
            if blob is None:
                self._send(
                    404,
                    b'{"errors":[{"code":"BLOB_UNKNOWN","message":"blob unknown"}]}',
                    {"Content-Type": "application/json"},
                )
                return
            self._send(200, blob, {"Content-Type": "application/octet-stream"})
            return

        self._send(404, b"", {})

    def _handle_token(self) -> None:
        if not self._has_basic():
            self._send(
                401,
                b'{"errors":[{"code":"UNAUTHORIZED","message":"auth required"}]}',
                {"Content-Type": "application/json"},
            )
            return
        self._send_json(
            200,
            {
                "token": _TOKEN,
                "access_token": _TOKEN,
                "expires_in": 300,
                "issued_at": "now",
            },
        )


class StubHarbor:
    """Threaded HTTP(S) stub exposing the test API used by fixtures."""

    def __init__(self) -> None:
        self.state = _StubHarborState()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.ca_cert_path: str = ""

    def start(self, tls: tuple[str, str] | None = None) -> str:
        """Bind on a random loopback port and serve in a background thread.

        ``tls`` = (certfile, keyfile) to serve HTTPS (both real clients use
        https against Harbor — oras-py defaults to https, HarborClient is
        pointed at https by the fixture). Trust the cert via
        ``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE`` in subprocess envs.

        Returns the base URL (``http(s)://127.0.0.1:{port}``).
        """
        import socket
        import ssl

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), StubHarborHandler)
        self._httpd.state = self.state  # type: ignore[attr-defined]
        if tls:
            certfile, keyfile = tls
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile, keyfile)
            self._httpd.socket = ctx.wrap_socket(self._httpd.socket, server_side=True)
            scheme = "https"
        else:
            scheme = "http"
        self.state.base_url = f"{scheme}://127.0.0.1:{port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Stub Harbor listening on %s", self.state.base_url)
        return self.state.base_url

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # -- test API ------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self.state.base_url

    def register_model(
        self, harbor_ref: str, files: dict[str, bytes]
    ) -> dict[str, Any]:
        return self.state.register_model(harbor_ref, files)

    def received_requests(self) -> list[tuple[str, str, dict[str, str]]]:
        return self.state.received_requests()

    def received_paths(self) -> list[str]:
        return [path for _, path, _ in self.state.received_requests()]

    def count_requests(self, method: str, path_contains: str) -> int:
        return sum(
            1
            for m, path, _ in self.state.received_requests()
            if m == method and path_contains in path
        )

    def reset(self) -> None:
        self.state.reset()
