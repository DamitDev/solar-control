"""API endpoint management routes (under /api/endpoints)."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator

from app.database.endpoints import endpoint_db
from app.auth import invalidate_endpoint_cache

router = APIRouter(prefix="/endpoints", tags=["endpoints"])


class EndpointCreate(BaseModel):
    name: str
    description: str | None = None
    api_key: str | None = None


class EndpointUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    api_key: str | None = None
    _description_provided: bool = False

    @model_validator(mode="before")
    @classmethod
    def _track_description(cls, values: dict) -> dict:
        if isinstance(values, dict) and "description" in values:
            values["_description_provided"] = True
        return values


@router.get("")
async def list_endpoints():
    endpoints = await endpoint_db.get_all_endpoints()
    return [ep.model_dump() for ep in endpoints]


@router.post("")
async def create_endpoint(data: EndpointCreate):
    try:
        ep = await endpoint_db.create_endpoint(
            name=data.name, description=data.description, api_key=data.api_key
        )
        await invalidate_endpoint_cache()
        return ep.model_dump()
    except Exception as e:
        if "unique" in str(e).lower():
            raise HTTPException(
                status_code=409, detail="Endpoint name or API key already exists"
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{endpoint_id}")
async def get_endpoint(endpoint_id: str):
    ep = await endpoint_db.get_endpoint(endpoint_id)
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    return ep.model_dump()


@router.put("/{endpoint_id}")
async def update_endpoint(endpoint_id: str, data: EndpointUpdate):
    kwargs: dict[str, str | None] = {}
    if data.name is not None:
        kwargs["name"] = data.name
    if data._description_provided:
        kwargs["description"] = data.description
    if data.api_key is not None:
        kwargs["api_key"] = data.api_key

    ep = await endpoint_db.update_endpoint(endpoint_id, **kwargs)
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    await invalidate_endpoint_cache()
    return ep.model_dump()


@router.delete("/{endpoint_id}")
async def delete_endpoint(endpoint_id: str):
    ep = await endpoint_db.get_endpoint(endpoint_id)
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    await endpoint_db.delete_endpoint(endpoint_id)
    await invalidate_endpoint_cache()
    return {"message": f"Endpoint '{ep.name}' deleted", "id": endpoint_id}


@router.get("/{endpoint_id}/usage")
async def get_endpoint_usage(endpoint_id: str, hours: int = 24):
    ep = await endpoint_db.get_endpoint(endpoint_id)
    if not ep:
        raise HTTPException(status_code=404, detail="Endpoint not found")
    stats = await endpoint_db.get_usage_stats(endpoint_id, hours=hours)
    return {"endpoint": ep.model_dump(), "hours": hours, "usage": stats}
