# Solar Control

A coordinator for multiple solar-host instances with OpenAI-compatible API gateway. Supports multiple backend types including llama.cpp and HuggingFace models.

## Features

- **Stateless, multi-replica ready** - Host connection state and endpoint auth cache in Redis; no in-process state
- **Multi-backend support** - Route to llama.cpp, HuggingFace Causal LM, Classification, and Embedding models
- Manage multiple solar-host instances via Socket.IO (/hosts namespace)
- OpenAI-compatible API gateway with model routing
- **Classification endpoint** - Custom `/v1/classify` endpoint for sequence classification models
- Model alias resolution (exact match; optional prefix fallback)
- Host-aware, model-size-weighted load balancing (prefers free hosts; otherwise chooses lowest active parameter load; round-robin tiebreaker)
- **Endpoint-aware routing** - Routes requests only to instances that support the requested endpoint
- **Multi-tenant API** - Create multiple API endpoints with individual keys, tracked usage stats
- **WebUI Socket.IO namespace** - `/webui` for dashboard: real-time host/instance status, gateway events, pending host approval
- **Pending host approval** - Hosts register first; management API lists and approves/rejects before they join the pool
- Transparent authentication handling (endpoint API keys for gateway; management API key for WebUI and admin routes)
- WebSocket log aggregation
- Docker support with automatic database migrations

## Supported Backend Types

| Backend | Endpoints |
|---------|-----------|
| **llama.cpp** | `/v1/chat/completions`, `/v1/completions`, `/v1/models` |
| **HuggingFace Causal** | `/v1/chat/completions`, `/v1/completions`, `/v1/models` |
| **HuggingFace Classification** | `/v1/classify`, `/v1/models` |
| **HuggingFace Embedding** | `/v1/embeddings`, `/v1/models` |
| **Reranker** | `/v1/rerank`, `/v1/models` |

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file or set environment variables:

```bash
MANAGEMENT_API_KEY=your-management-api-key-here
HOST=0.0.0.0
PORT=8000
DATABASE_URL=postgresql://solar:solar@localhost:5432/solar_gateway
REDIS_URL=redis://localhost:6379/0
```

- **MANAGEMENT_API_KEY** - Required for WebUI and management API (host approval, endpoints, gateway stats). Sent as `X-API-Key` or `Authorization: Bearer <key>` (or via Socket.IO `auth` for the `/webui` namespace).
- **DATABASE_URL** - PostgreSQL connection string. Stores hosts, API endpoints, and gateway request logs.
- **REDIS_URL** - Required. Used for host connection state (sid-to-host, instances, pending hosts), endpoint API key cache, and routing state. Enables stateless operation and multiple replicas.

Gateway API keys are managed through the multi-tenant endpoint system (see `/api/endpoints`).

## Running Natively

```bash
# Run database migrations first
./migrate.sh

# Start the server (ASGI app; Socket.IO and HTTP on same port)
uvicorn app.main:sio_asgi_app --host 0.0.0.0 --port 8000 --reload
```

## Running with Docker

```bash
# Build and start with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

The container entrypoint automatically runs Alembic database migrations before starting the application. PostgreSQL and Redis healthchecks ensure the app only starts after both services are ready.

## Database Migrations

Schema changes are managed with [Alembic](https://alembic.sqlalchemy.org/). SQLAlchemy table models live in `app/database/tables.py`.

```bash
# Apply all pending migrations (runs automatically on container start)
python -m alembic upgrade head

# Check current revision
python -m alembic current

# Create a new migration after modifying tables.py
python -m alembic revision --autogenerate -m "description of change"
```

Migration files are in `app/database/migrations/versions/` and follow a `NNNN_description.py` naming convention.

## API Endpoints

### Host Management

- `GET /api/hosts` - List all solar-hosts
- `POST /api/hosts` - Register a new solar-host (or create with approved config)
- `GET /api/hosts/{host_id}` - Get host details
- `DELETE /api/hosts/{host_id}` - Remove a solar-host
- `GET /api/hosts/{host_id}/instances` - Get instances from a host
- `POST /api/hosts/refresh-all` - Refresh all hosts (reconnect and sync instances)

### Pending Host Approval (Management API)

- `GET /api/hosts/pending` - List hosts awaiting approval
- `POST /api/hosts/pending/{pending_id}/approve` - Approve and add host (body: name, url)
- `POST /api/hosts/pending/{pending_id}/reject` - Reject pending host

Hosts connect via Socket.IO to the `/hosts` namespace; they appear in pending until approved via these endpoints. The WebUI uses the same management API key for Socket.IO and REST.

### API Endpoint Management

- `GET /api/endpoints` - List all API endpoints (tenants)
- `POST /api/endpoints` - Create a new endpoint (generates API key)
- `GET /api/endpoints/{id}` - Get endpoint details
- `PUT /api/endpoints/{id}` - Update endpoint
- `DELETE /api/endpoints/{id}` - Delete endpoint
- `GET /api/endpoints/{id}/usage` - Get usage statistics

### OpenAI Gateway

- `POST /v1/chat/completions` - Chat completions (routed by model)
- `POST /v1/completions` - Text completions (routed by model)
- `GET /v1/models` - List all available models

### Classification Gateway

- `POST /v1/classify` - Text classification (routed by model to HuggingFace Classification instances)

### Embeddings Gateway

- `POST /v1/embeddings` - Text embeddings (routed by model to HuggingFace Embedding instances)

### Rerank Gateway

- `POST /v1/rerank` - Rerank documents against a query (routed by model to reranker instances)

### Gateway Monitoring

- `GET /api/gateway/stats` - Request statistics with per-model and per-host breakdowns
- `GET /api/gateway/requests` - Paginated request history with filtering
- `GET /api/gateway/events/recent` - Recent routing events (errors, reroutes)

### Instance Proxy (via solar-control to host)

- `POST /api/hosts/{host_id}/instances/{instance_id}/start` - Start instance
- `POST /api/hosts/{host_id}/instances/{instance_id}/stop` - Stop instance
- `POST /api/hosts/{host_id}/instances/{instance_id}/restart` - Restart instance
- `GET /api/hosts/{host_id}/instances/{instance_id}/state` - Get runtime state
- `GET /api/hosts/{host_id}/instances/{instance_id}/logs` - Get recent logs

## Authentication

- **Gateway requests** (e.g. `/v1/chat/completions`, `/v1/embeddings`) use **endpoint API keys** created via `/api/endpoints`. Send as `X-API-Key` or `Authorization: Bearer <key>`. The management key also works for gateway requests.
- **Management and WebUI** use **MANAGEMENT_API_KEY**: management REST routes and the Socket.IO `/webui` namespace. WebUI clients can send the key in Socket.IO `auth.api_key` or the reverse proxy can inject `X-API-Key` / `Authorization` on the upgrade request.

Solar-control handles authentication to solar-hosts transparently using stored credentials. Endpoint API keys are cached in Redis with TTL and invalidation on create/update/delete.

## Socket.IO Namespaces

- **/hosts** - Solar-host instances connect here. They register, send heartbeat and instance updates; control stores connection state in Redis.
- **/webui** - Dashboard clients connect here. Authenticated with MANAGEMENT_API_KEY (via `auth.api_key` or proxy-injected headers). Receives `initial_status`, host/instance updates, gateway events, and pending host list.

## Example Host Registration

After a host is approved (or when creating directly via management API):

```json
{
  "name": "GPU Server 1",
  "url": "http://192.168.1.100:8001",
  "api_key": "host-specific-api-key"
}
```

## Gateway Usage Examples

### Chat Completions (llama.cpp or HuggingFace Causal)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: your-endpoint-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3:8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Text Classification (HuggingFace Classification)

```bash
curl http://localhost:8000/v1/classify \
  -H "X-API-Key: your-endpoint-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "classifier:deberta",
    "input": "This product is amazing! I love it."
  }'
```

**Classification Response:**

```json
{
  "id": "clf-abc123",
  "object": "classification",
  "model": "classifier:deberta",
  "choices": [
    {
      "index": 0,
      "label": "positive",
      "score": 0.9876
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```

### Batch Classification

```bash
curl http://localhost:8000/v1/classify \
  -H "X-API-Key: your-endpoint-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "classifier:deberta",
    "input": [
      "This is great!",
      "This is terrible."
    ]
  }'
```

### Text Embeddings (HuggingFace Embedding)

```bash
curl http://localhost:8000/v1/embeddings \
  -H "X-API-Key: your-endpoint-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embed:minilm",
    "input": "Hello, world!"
  }'
```

**Embedding Response:**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0123, -0.0456, 0.0789],
      "index": 0
    }
  ],
  "model": "embed:minilm",
  "usage": {
    "prompt_tokens": 4,
    "total_tokens": 4
  }
}
```

### Batch Embeddings

```bash
curl http://localhost:8000/v1/embeddings \
  -H "X-API-Key: your-endpoint-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embed:minilm",
    "input": [
      "First text to embed",
      "Second text to embed"
    ]
  }'
```

## Routing Behavior

Solar-control automatically routes requests based on:

1. **Model alias** - Matches the `model` field to instance aliases
2. **Endpoint support** - Only routes to instances that support the requested endpoint
3. **Load balancing** - Prefers idle instances; distributes load based on model size

For example:
- A `/v1/chat/completions` request will only route to llama.cpp or HuggingFace Causal instances
- A `/v1/classify` request will only route to HuggingFace Classification instances
- A `/v1/embeddings` request will only route to HuggingFace Embedding instances

The gateway automatically discovers which endpoints each instance supports when the model registry is refreshed.

## Project Structure

```
app/
  config.py              # Settings (pydantic-settings)
  auth.py                # API key authentication middleware
  gateway.py             # Request routing and load balancing
  main.py                # FastAPI + Socket.IO app assembly
  models/                # Pydantic domain models
    host.py              #   Host, HostStatus, MemoryInfo
    openai.py            #   OpenAI-compatible request/response models
    socketio.py          #   Socket.IO event payload models
  database/
    tables.py            # SQLAlchemy declarative table models
    connection.py        # Async engine + session factory
    hosts.py             # Host CRUD
    endpoints.py         # API endpoint CRUD
    logs.py              # Buffered gateway event logging
    migrations/          # Alembic migrations
      env.py
      versions/          # Migration scripts (0001_initial_schema.py, ...)
  redis_state/           # Redis-backed shared state
    connection.py        # Redis client
    hosts.py             # Socket.IO connection tracking
    registry.py          # Model-to-instance registry
    routing.py           # Active request counts, weights, round-robin
    health.py            # Instance health TTLs
  routes/
    openai.py            # /v1/* gateway routes
    management/          # /api/* admin routes
      hosts.py           #   Host management + instance proxy
      endpoints.py       #   API endpoint management
      gateway.py         #   Gateway stats and request history
  socketio_app/
    server.py            # Socket.IO server setup
    host_handlers.py     # /hosts namespace handlers
    webui_handlers.py    # /webui namespace handlers
```
