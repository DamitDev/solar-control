# D-017 Cross-Service Integration Test Suite

Proves the full Data Repository → `repo://` → Solar Control → Solar Host pull
→ instance → inference pipeline, plus intent reconciliation (S-040/041/042)
and instance migration (S-037), with real service subprocesses and a stub
Harbor speaking the OCI Distribution protocol.

```
┌──────────────────────────────────────────────────────────────────────┐
│                      pytest process (tests_integration)               │
│                                                                      │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ testcontainers │  │  Stub Harbor │  │ httpx clients            │   │
│  │ Postgres (2 DB)│  │ (TLS :rand)  │  └───────────┬──────────────┘   │
│  │ Redis (:rand)  │  │              │              │                  │
│  └───────┬───────┘  └──────┬───────┘              │                  │
│  ┌───────┴─────────────────┴───────────────────────┴──────────────┐   │
│  │                     Subprocesses (module scope)                 │   │
│  │  Data Repository   Solar Control        Solar Host A           │   │
│  │  uvicorn :rand      uvicorn :rand       uvicorn :rand          │   │
│  │  (PG data_repo)     (PG solar_gateway,  (MODELS_DIR tmp)       │   │
│  │                     Redis, reconciler)                          │   │
│  │                                          Solar Host B           │   │
│  │                                          uvicorn :rand          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  Hosts ──WS /ws/host-channel──▶ Control   (health, instances_update) │
│  Control ──HTTP──▶ Hosts / Data Repo      Control ──never──▶ Harbor │
└──────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker running (testcontainers pulls `postgres:15-alpine`, `redis:7-alpine`)
- `openssl` on PATH (stub Harbor TLS certs)
- Python 3.14 venvs (all three services are Python 3.14 here):

| Venv | Contents |
|---|---|
| `solar-control/.venv` | control deps + **test deps** (pytest, pytest-asyncio, testcontainers, httpx, psycopg2) |
| `solar-host/.venv` | host deps + `.[huggingface]` (torch, transformers — the model fixture backend) |
| `data-repository/.venv` | data-repo `requirements.txt` |

## Install

```bash
# one-time, per venv (see requirements.txt for the exact list)
cd ~/work/solar/solar-control
env -u PYTHONPATH .venv/bin/pip install -r tests_integration/requirements.txt
env -u PYTHONPATH .venv/bin/pip install -e ".[huggingface]"   # in solar-host
# create data-repository/.venv with its requirements.txt
```

> **PYTHONPATH warning:** the Hermes desktop app leaks `PYTHONPATH` pointing
> at its own venv site-packages. Every python invocation for this suite must
> clear it (`env -u PYTHONPATH ...`) — otherwise imports resolve to
> mismatched binaries (`pydantic_core` etc.) and everything breaks.

## Run

```bash
cd ~/work/solar/solar-control
env -u PYTHONPATH .venv/bin/python -m pytest -c tests_integration/pytest.ini tests_integration/ -v
```

Markers (applied automatically by folder):

```bash
-m repo_path        # minimal repo path (registration, resolve, distribute, inference)
-m intent_path      # declarative intents (API, reconcile, scaling, strategies)
-m migration_path   # S-037 migration (explicit, reconciler, guards)
-m infrastructure   # WS seam, gateway registry, model cache, reconciler wake
```

Run one file:

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -c tests_integration/pytest.ini \
    tests_integration/intent_path/test_reconcile_to_ready.py -v
```

A full run takes roughly 8–15 minutes (each test module spawns a fresh
4-process stack; inference tests load torch per instance start).

## Layout

```
tests_integration/
├── conftest.py            # containers, alembic, stub harbor, stack fixture,
│                          # clean_state, HTTP clients, folder markers
├── fixtures/
│   ├── stub_harbor.py     # OCI Distribution v2 stub (TLS, token dance,
│   │                      # request log, register_model)
│   ├── helpers.py         # wait_for, free_port, subprocess spawner, certs
│   ├── seed.py            # DB/API seed helpers, host-log request counting
│   ├── intents.py         # intent payload + readiness polling helpers
│   ├── constants.py       # shared constants (keys, model URIs, harbor_port)
│   ├── generate_test_model.py   # regenerates fixtures/test_model/ (torch)
│   ├── smoke_stub_harbor.py     # Phase-1 smoke: real OrasHelper + HarborClient
│   └── test_model/        # COMMITTED tiny HF classification model (~230 KB)
├── repo_path/             # minimal repo path (9 tests)
├── intent_path/           # declarative path (15 tests)
├── migration_path/        # S-037 (9 tests)
└── infrastructure/        # WS seam, registry, cache, wake (6 tests)
```

## Fixture regeneration

The committed model fixture (`fixtures/test_model/`) is generated once with a
fixed seed — regenerate only if the fixture spec changes:

```bash
cd ~/work/solar/solar-control
env -u PYTHONPATH ../solar-host/.venv/bin/python \
    tests_integration/fixtures/generate_test_model.py
```

## Design notes / pitfalls encoded in the suite

- **Stub Harbor is real OCI Distribution.** Both real clients run against it:
  `HarborClient.verify_artifact` (data-repo, HEAD manifests + token endpoint)
  and `OrasHelper.pull` (host, oras-py token dance: 401 challenge → token →
  bearer manifest+blob GETs). It serves **TLS** because oras-py hardcodes
  https; trust is injected via `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` in every
  service env. The request log proves the task's core assertion: **solar
  control never calls Harbor** (data-repo + hosts are the only clients).
- **Distinct API keys per host.** Control resolves hosts by `api_key`
  (`get_host_by_api_key` → `scalar_one_or_none`); a shared key between the
  two registered rows crashes the WS connect handler with
  `MultipleResultsFound` and silently degrades the whole seam to HTTP polling.
- **Hosts table is truncated per module.** It persists across modules while
  stacks are per-module; stale rows made tests resolve host-a to a dead
  previous module's port. `_build_stack` truncates `hosts, jobs` before
  re-registering.
- **pytest-asyncio loop scoping.** `asyncio_default_test_loop_scope = session`
  + `asyncio_default_fixture_loop_scope = session` in pytest.ini; the
  module-scoped `stack` fixture is deliberately **sync** (wraps the async
  spawn in `asyncio.run`). Async module-scoped generator fixtures get
  re-created per test loop otherwise (one full stack per test).
- **Startup race:** the host marks instances RUNNING ~2s after spawn, before
  torch/transformers finish loading. Inference tests gate on the gateway
  actually routing (`/v1/models` through control queries the instance
  upstream) before firing `/v1/classify`.
- **Soft-deleted intents 404.** `GET /api/intents/{id}` returns 404 for
  soft-deleted rows, so delete tests poll for the 404 rather than a
  `deleted` phase.
- **Disowned markers live in the Redis cache, not the host.** The orphan
  DISOWN clears markers in control's `solar:hosts:instances` cache; the
  host's own config retains them (no host-side PATCH for running
  instances) and management routes reflect the *host* view. Marker
  assertions read the cache directly (`redis_cache_instances` in
  `fixtures/seed.py`).
- **No PUT `/api/intents/{id}`** (spec §12.5): strategy/scale tests mutate
  the `intents` row directly via `update_intent_in_db` (documented,
  intentional).
- **Version-change artifacts:** `_register_v2` registers a second version
  with identical files except a modified `model.safetensors` (idempotent —
  the data-repo rejects a duplicate registration with 409).
- **Cleanup on failure:** `clean_state` stops+deletes all host instances,
  truncates `intents`, flushes volatile Redis keys (keeps `solar:hosts:*`
  connection state), and resets the stub Harbor request log. On fixture
  failure the stack log tails are dumped via `stack.tail()`.

## Known deviations from the plan (documented)

- `test_failed_create_backoff` (infrastructure) kills the **data-repo**
  subprocess rather than a host: dead hosts are excluded from placement
  candidates (`reachable=False`), so no CREATE is ever attempted against
  them. Killing data-repo makes the reconcile-time resolve step fail
  deterministically → `last_error` recorded → respawn → next tick recovers.
- The GGUF/llama.cpp optional path (`skipif` no `llama-server`) is not
  implemented — the HF classification fixture covers the integration contract
  with pure Python.

## Platform bugs this suite found (fixed in solar-control)

1. **DISOWN created a self-referential dict for flat WS cache entries**
   (`app/services/reconciliation.py`): `cfg = inst.get("config", inst)` made
   `cfg is inst` for flat entries, then `inst["config"] = cfg` produced a
   circular dict that json serialization rejected ("Circular reference
   detected") — every orphan intent delete (DELETE ?orphan=true) failed
   with backoff. Fixed by only re-attaching real nested configs.
2. **`_detect_backend_drift` false-positived on every managed instance**
   (`app/services/reconciliation.py`): the WS instances cache is flat by
   design (only id/alias/status/port/backend_type/model_source/...), so
   intent backend fields like `device`/`dtype`/`max_length`/`labels` were
   never present and compared as None → perpetual `action=replace reason=
   backend config drift` → REPLACE-stop loop that wedged the reconciler and
   stop-spammed the host. Fixed by skipping fields the flat cache cannot
   carry (model_source drift — the version-change path — remains fully
   detectable and tested).
3. **Gateway HTTP-poll re-seed dropped ownership markers**
   (`app/gateway.py` `_ws_cache_from_http_instances`): when the registry
   refresh found a connected host with an empty cache it polled the host
   and re-seeded `solar:hosts:instances` WITHOUT `managed_by`/`intent_id`/
   `model_source`/`priority`. The reconciler then saw managed instances as
   unmanaged → duplicate creates on one host (one-replica-per-host
   violated) → surplus/cleanup STOP+DELETE racing in-flight instance
   starts (host-side `404 Instance not found after start`, `exit -15`
   SIGTERMs) and candidate flapping. Fixed by preserving the ownership
   fields in the conversion (matching the WS push shape).

