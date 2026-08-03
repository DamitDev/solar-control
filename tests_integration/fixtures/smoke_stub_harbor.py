"""Phase-1 smoke: real OrasHelper.pull + HarborClient.verify_artifact vs stub.

Standalone script (not a pytest test) — settles the OCI manifest/layer
format and auth dance before the suite is built. Run with a venv that has
``harbor-oci-client`` installed (solar-host or data-repository venv):

    env -u PYTHONPATH ../solar-host/.venv/bin/python \
        tests_integration/fixtures/smoke_stub_harbor.py

Exits 0 when both clients work against the stub.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.stub_harbor import StubHarbor  # noqa: E402


def _make_certs(cert_dir: str) -> tuple[str, str]:
    """Generate a self-signed CA + server cert for 127.0.0.1 via openssl.

    Returns (server_cert, server_key). Callers must point
    ``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE`` at the CA cert so the real
    clients (httpx, requests) trust it.
    """
    import subprocess

    ca_key = os.path.join(cert_dir, "ca.key")
    ca_crt = os.path.join(cert_dir, "ca.crt")
    srv_key = os.path.join(cert_dir, "server.key")
    srv_csr = os.path.join(cert_dir, "server.csr")
    srv_crt = os.path.join(cert_dir, "server.crt")
    ext = os.path.join(cert_dir, "san.ext")

    def run(*args: str) -> None:
        subprocess.run(args, check=True, capture_output=True)

    run(
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-keyout",
        ca_key,
        "-out",
        ca_crt,
        "-days",
        "30",
        "-nodes",
        "-subj",
        "/CN=StubHarbor Test CA",
        "-addext",
        "basicConstraints=critical,CA:TRUE",
        "-addext",
        "keyUsage=critical,keyCertSign,cRLSign",
    )
    run(
        "openssl",
        "req",
        "-newkey",
        "rsa:2048",
        "-keyout",
        srv_key,
        "-out",
        srv_csr,
        "-nodes",
        "-subj",
        "/CN=127.0.0.1",
    )
    with open(ext, "w") as f:
        f.write(
            "subjectAltName=IP:127.0.0.1\n"
            "keyUsage=digitalSignature,keyEncipherment\n"
            "extendedKeyUsage=serverAuth\n"
        )
    run(
        "openssl",
        "x509",
        "-req",
        "-in",
        srv_csr,
        "-CA",
        ca_crt,
        "-CAkey",
        ca_key,
        "-CAcreateserial",
        "-out",
        srv_crt,
        "-days",
        "30",
        "-extfile",
        ext,
    )
    return srv_crt, srv_key


def main() -> int:

    stub = StubHarbor()
    cert_dir = tempfile.mkdtemp(prefix="stub-harbor-certs-")
    srv_crt, srv_key = _make_certs(cert_dir)
    ca_crt = os.path.join(cert_dir, "ca.crt")
    os.environ["SSL_CERT_FILE"] = ca_crt
    os.environ["REQUESTS_CA_BUNDLE"] = ca_crt

    base = stub.start(tls=(srv_crt, srv_key))
    print(f"stub harbor at {base}")

    # 1. Register a fake HF-model artifact: 3 flat files, one layer each.
    files = {
        "config.json": b'{"architectures": ["BertForSequenceClassification"]}',
        "model.safetensors": b"\x00\x01fake-safetensors-bytes" * 100,
        "tokenizer.json": b'{"version": "1.0"}',
    }
    ref_host = base.split("://", 1)[1]
    harbor_ref = f"{ref_host}/supernova/test-model:v1"
    manifest = stub.register_model(harbor_ref, files)
    print(f"registered {harbor_ref} layers={len(manifest['layers'])}")

    # 2. OrasHelper.pull (the solar-host client path)
    from harbor_oci_client import OrasHelper

    oras = OrasHelper(hostname="127.0.0.1", username="robot$test", password="test")
    with tempfile.TemporaryDirectory() as outdir:
        pulled = oras.pull(harbor_ref, outdir=outdir)
        names = sorted(Path(f).name for f in pulled)
        print(f"OrasHelper.pull -> {names}")
        assert names == ["config.json", "model.safetensors", "tokenizer.json"], names
        for name in files:
            data = Path(outdir, name).read_bytes()
            assert data == files[name], f"{name} content mismatch"
    print("OrasHelper.pull OK (flat files, no tarball)")

    # 3. HarborClient.verify_artifact (the data-repository client path)
    import asyncio

    from harbor_oci_client import HarborClient

    async def verify() -> None:
        client = HarborClient(base_url=base, username="robot$test", password="test")
        try:
            info = await client.verify_artifact(harbor_ref)
            print(
                f"HarborClient.verify_artifact -> digest={info.digest[:20]}... len={info.content_length}"
            )
            assert info.digest.startswith("sha256:")
            assert info.content_length > 0
        finally:
            await client.close()

    asyncio.run(verify())
    print("HarborClient.verify_artifact OK")

    # 4. Negative: unknown tag must 404 for both clients.
    try:
        oras.pull(f"{ref_host}/supernova/test-model:v9", outdir=tempfile.mkdtemp())
        raise SystemExit("ERROR: pull of unknown tag did not fail")
    except Exception as e:  # noqa: BLE001
        print(f"unknown-tag pull failed as expected: {type(e).__name__}")

    reqs = stub.received_requests()
    print(f"\nrequest log ({len(reqs)} entries):")
    for method, path, headers in reqs:
        auth = (
            "Bearer"
            if headers.get("Authorization", "").startswith("Bearer")
            else (
                "Basic" if headers.get("Authorization", "").startswith("Basic") else "-"
            )
        )
        print(f"  {method:5} {path:60} auth={auth}")

    print("\nSMOKE OK")
    stub.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
