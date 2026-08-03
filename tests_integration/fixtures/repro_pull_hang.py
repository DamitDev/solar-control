"""Reproduce the v2 pull hang: host pull_model (pebble subprocess) v1 then v2.

Run with the host venv: env -u PYTHONPATH ../solar-host/.venv/bin/python
tests_integration/fixtures/repro_pull_hang.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixtures.helpers import make_certs  # noqa: E402
from fixtures.stub_harbor import StubHarbor  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "test_model"


def main() -> int:
    import subprocess

    # 1. Fresh stub harbor (TLS)
    tmp = Path(tempfile.mkdtemp(prefix="pull-hang-"))
    ca, srv_crt, srv_key = make_certs(tmp)
    stub = StubHarbor()
    base = stub.start(tls=(str(srv_crt), str(srv_key)))
    port = base.split(":")[-1]

    files_v1 = {p.name: p.read_bytes() for p in MODEL_DIR.iterdir() if p.is_file()}
    files_v2 = dict(files_v1)
    files_v2["model.safetensors"] = files_v2["model.safetensors"] + b"v2"

    stub.register_model(f"127.0.0.1:{port}/supernova/test-model:v1", files_v1)
    stub.register_model(f"127.0.0.1:{port}/supernova/test-model:v2", files_v2)
    print(f"stub on {base}, v1+v2 registered")

    # 2. Host pull_model in a subprocess (pebble) — same env shape as the suite
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(
        {
            "SSL_CERT_FILE": str(ca),
            "REQUESTS_CA_BUNDLE": str(ca),
            "HARBOR_URL": base,
            "HARBOR_USERNAME": "robot$test",
            "HARBOR_PASSWORD": "test",
            "MODELS_DIR": str(tmp / "models"),
            "PULL_USE_SUBPROCESS": "true",
        }
    )
    script = f"""
import sys, time, json
sys.path.insert(0, {str(Path.cwd())!r})
from solar_host import models_manager
import solar_host.config as cfg
cfg.settings.models_dir = {str(tmp / "models")!r}
cfg.settings.harbor_url = {base!r}
cfg.settings.harbor_username = "robot$test"
cfg.settings.harbor_password = "test"

for uri, ref in [
    ("repo://test-model:v1", "127.0.0.1:{port}/supernova/test-model:v1"),
    ("repo://test-model:v2", "127.0.0.1:{port}/supernova/test-model:v2"),
]:
    t0 = time.monotonic()
    print(f"PULL {{uri}} start", flush=True)
    result = models_manager.pull_model(
        source="harbor", source_uri=uri, harbor_ref=ref,
        name="test-model", version=uri.split(":")[-1], category="model",
        checksum="sha256:abc", metadata={{"k": "v"}},
    )
    print(f"PULL {{uri}} done in {{time.monotonic()-t0:.1f}}s -> {{result}}", flush=True)
print("ALL_DONE", flush=True)
"""
    proc = subprocess.run(
        [sys.executable, "-u", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    print("RC:", proc.returncode)
    print(proc.stdout[-2000:])
    print(proc.stderr[-1500:])
    stub.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
