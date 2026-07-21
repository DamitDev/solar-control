"""Instance management API routes (under /api/instances)."""

from fastapi import APIRouter

from app.models.migration import MigrateRequest, MigrationResult
from app.services.migration import execute_migration

router = APIRouter(prefix="/instances", tags=["instances"])


@router.post("/migrate", response_model=MigrationResult)
async def migrate_instance(req: MigrateRequest) -> MigrationResult:
    """Migrate an inference instance from a source host to a target host.

    Stops the source instance, ensures the model is on the target host
    via S-019 distribution, and recreates the instance from the original
    configuration. Enforces one-replica-per-host and validates target
    host fitness.
    """
    return await execute_migration(
        instance_id=req.instance_id,
        source_host_id=req.source_host_id,
        target_host_id=req.target_host_id,
        allow_production=req.allow_production,
    )
