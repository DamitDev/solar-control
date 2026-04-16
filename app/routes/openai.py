"""OpenAI-compatible API gateway endpoints.

Each request is authenticated against the api_endpoints table.
The resolved endpoint_id is stored in request.state by the auth middleware
and passed through to the gateway for logging.
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.models import (
    ChatCompletionRequest,
    CompletionRequest,
    ClassifyRequest,
    EmbeddingRequest,
    RerankRequest,
)
from app.gateway import gateway

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai"])


def _get_endpoint_id(request: Request) -> str | None:
    return getattr(request.state, "endpoint_id", None)  # set by auth_middleware


def _safe_stream(
    model: str, endpoint: str, request_data: dict, client_ip: str, endpoint_id
):
    """Wrap gateway.stream_request so client disconnects emit a proper error event."""

    async def generator():
        stream = gateway.stream_request(
            model,
            endpoint,
            request_data,
            client_ip,
            endpoint_id=endpoint_id,
        )
        try:
            async for chunk in stream:
                yield chunk
        except Exception as e:
            payload = json.dumps({"error": str(e)})
            yield f"data: {payload}\n\n".encode()
        finally:
            await stream.aclose()

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.get("/models")
async def list_models():
    try:
        models = await gateway.get_available_models()
        return {"object": "list", "data": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, client: Request):
    try:
        client_ip = client.client.host if client.client else "unknown"
        endpoint_id = _get_endpoint_id(client)
        request_data = request.model_dump(exclude_none=True)

        if request.stream:
            return _safe_stream(
                request.model,
                "/v1/chat/completions",
                request_data,
                client_ip,
                endpoint_id,
            )
        else:
            response = await gateway.route_request(
                request.model,
                "/v1/chat/completions",
                request_data,
                client_ip,
                endpoint_id=endpoint_id,
            )
            return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completions")
async def completions(request: CompletionRequest, client: Request):
    try:
        client_ip = client.client.host if client.client else "unknown"
        endpoint_id = _get_endpoint_id(client)
        request_data = request.model_dump(exclude_none=True)

        if request.stream:
            return _safe_stream(
                request.model,
                "/v1/completions",
                request_data,
                client_ip,
                endpoint_id,
            )
        else:
            response = await gateway.route_request(
                request.model,
                "/v1/completions",
                request_data,
                client_ip,
                endpoint_id=endpoint_id,
            )
            return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify")
async def classify(request: ClassifyRequest, client: Request):
    try:
        client_ip = client.client.host if client.client else "unknown"
        endpoint_id = _get_endpoint_id(client)
        request_data = request.model_dump(exclude_none=True)
        response = await gateway.route_request(
            request.model,
            "/v1/classify",
            request_data,
            client_ip,
            required_endpoint="/v1/classify",
            endpoint_id=endpoint_id,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings")
async def embeddings(request: EmbeddingRequest, client: Request):
    try:
        client_ip = client.client.host if client.client else "unknown"
        endpoint_id = _get_endpoint_id(client)
        request_data = request.model_dump(exclude_none=True)
        response = await gateway.route_request(
            request.model,
            "/v1/embeddings",
            request_data,
            client_ip,
            required_endpoint="/v1/embeddings",
            endpoint_id=endpoint_id,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rerank")
async def rerank(request: RerankRequest, client: Request):
    try:
        client_ip = client.client.host if client.client else "unknown"
        endpoint_id = _get_endpoint_id(client)
        request_data = request.model_dump(exclude_none=True)
        response = await gateway.route_request(
            request.model,
            "/v1/rerank",
            request_data,
            client_ip,
            required_endpoint="/v1/rerank",
            endpoint_id=endpoint_id,
        )
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
