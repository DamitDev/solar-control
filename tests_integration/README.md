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

A full run takes roughly 2–3 minutes (one session-scoped 4-process stack is
reused by every module; per-test state is reset by `clean_state`). The wake
test builds its own 3600s-interval stack and runs last.

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
  registry listing the alias, then retry `/v1/classify` with a bounded
  budget (`classify_until_ok` in `fixtures/intents.py`) — see "The
  classify-404 flake" below.
- **`GET /v1/models` does NOT prove liveness.** The gateway fabricates a
  fallback entry for an alias whenever the upstream query fails
  (`app/gateway.py get_available_models`), so a crashed server keeps the
  alias listed as long as it lingers in the registry. All registry-derived
  gates (`_alias_visible`, `ready_replicas`, `Available`) can therefore
  regress mid-run; the classify retry is what actually absorbs the
  remaining startup/crash window.
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
- **One session-scoped stack; the wake test runs last.** The stack fixture
  is session-scoped (per-module stacks cost ~10-13s each — 16 sequential
  4-process spawns dominated the runtime). `clean_state` already resets
  per-test state and no test kills a host, so sharing is safe. The wake
  module (3600s interval) is reordered to run last
  (`pytest_collection_modifyitems`): it truncates the shared hosts table and
  stops the session control (two live controls sharing Postgres/Redis
  cross-reconcile — a second control resolves "host-a" to the other stack's
  host and races every intent). `test_failed_create_backoff` lives in its
  own module because it needs a live control.
- **Version-change artifacts:** `_register_v2` registers a second version
  with identical tensors but a different `model.safetensors` (re-saved with
  a `version: v2` header entry via `rewrite_safetensors_with_metadata` —
  valid file, different sha256). v2 replicas therefore actually serve
  inference, and the strategy tests assert classify liveness on the new
  version (idempotent — the data-repo rejects a duplicate registration
  with 409).
- **Cleanup on failure:** `clean_state` stops+deletes all host instances,
  truncates `intents`, flushes volatile Redis keys (keeps `solar:hosts:*`
  connection state), and resets the stub Harbor request log. On fixture
  failure the stack log tails are dumped via `stack.tail()`.

## The classify-404 flake (D-017, fixed)

Intermittent failure of `test_intent_reaches_ready`: the HF classification
server crashes at startup (tokenizer slow→fast conversion error — see
below) while the host has already reported the instance RUNNING; the
gateway registry drops the dead instance on the next refresh and
`POST /v1/classify` returns 404 `"Model '…' not found or no instances
available"` (`attempted == ∅`). The 93ms window in the observed failure:
`/v1/models` listed the alias (fabricated fallback entry, `gateway.py
get_available_models`) at 09:57:08.264 while the upstream was already
dead; the next registry refresh dropped the instance and the classify at
09:57:08.357 404'd.

The suite fixes, all proven load-bearing by deterministic fault injection
(SIGKILL the `hf_server` subprocess, one run per variant):

| Injection | Old behavior | New behavior |
|---|---|---|
| SIGKILL after alias visible, **one** classify | 404 (flake signature) | — |
| Same, **bounded retry** (`classify_until_ok`, 30s) | — | self-heals via reconciler §8.2 RECREATE in ~3.6s |
| SIGKILL at spawn (before ready), 15s ready-wait | times out (15.2s) | 30s ready-wait + retry converge (15.2s) |

The host reports a killed instance as `failed` ("Process exited
unexpectedly") → the reconciler RECREATEs: first attempt at the next
0.5s tick (post-ready kill ≈ 3.6s convergence), with the exponential
backoff (`_BACKOFF_MIN_S = 10`) after a failed start (spawn-kill ≈ 15.2s).
The retry cannot ride a *repeatedly* failing recreate (corrupt cached
artifact → restart-in-place re-reads the same bytes) — that case is
expected to produce the evidence dump, not a pass.

**Evidence preservation:** on classify exhaustion, `dump_instance_evidence`
(`fixtures/helpers.py`) writes `stack0/evidence-<alias>/` with the
instance server logs, sha256 of every pulled file vs the committed
fixture, the gateway registry entries (`supported_endpoints` — separates
the routing trap from a dead server), direct upstream probes (fallback vs
live), and host venv package versions. A loop hit should produce this
dump, not a bare 404.

**Fail-fast on the routing trap:** a 404 with `attempted == ∅` while the
registry still has the alias but no entry supports `/v1/classify` aborts
immediately (the reconciler cannot fix a missing endpoint).

**Tokenizer crash background (F6):** the venv (transformers 5.14.1 /
tokenizers 0.22.2 / torch 2.13.0 / safetensors 0.8.0) has no
sentencepiece/tiktoken; a slow→fast tokenizer conversion fallback crashes
with "You need to have sentencepiece or tiktoken installed". 18/18
isolated cold loads pass, so the trigger is suite-context (pulled copy or
timing), not plain nondeterminism. `solar-host` post-pull sha256
verification against OCI manifest layer digests (Task 4) catches the
truncated-pull leg at the source.

## WS instances_update seam (F5 — investigation outcome)

The suite exercises the **HTTP-poll fallback** for the gateway registry,
not the WS `instances_update` seam: control logs "Host … is connected but
has no cached instances; polling HTTP" continuously, because the
`instances_update` → Redis cache path never populates with the current
`solar-host` `feature/D-017` checkout. The host-side fixes for this
(registration/health re-send after approval; immediate instances_update
push on instance changes) exist only as an unpushed commit
(`eeacbe9 fix(ws_client)` on `fix/d017-host-fixes`) — see the Task 8
finding in the flake plan. The registry is carried entirely by
`refresh_model_registry`'s HTTP polling, which is why every test still
passes; the WS seam should be re-verified once the host fixes are merged.

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

