"""Pydantic models for instance migration (S-037)."""

from pydantic import BaseModel, Field


class MigrateRequest(BaseModel):
    """Request to migrate an instance from a source host to a target host."""

    instance_id: str
    source_host_id: str
    target_host_id: str
    allow_production: bool = Field(
        default=False,
        description=(
            "Explicit opt-in required to migrate production instances. "
            "Future automated flows (S-038, S-041) must set this to true "
            "only after an explicit policy decision."
        ),
    )


class MigrationStep(BaseModel):
    """A single step in the migration process."""

    step: str
    status: str  # "ok" | "skipped" | "failed"
    detail: dict | None = None


class MigrationResult(BaseModel):
    """Result of a completed (or failed) migration operation."""

    migration_id: str
    status: str  # "completed" | "failed"
    source_host_id: str
    source_host_name: str
    target_host_id: str
    target_host_name: str
    source_instance_id: str
    target_instance_id: str | None = None
    alias: str
    model_source: str
    priority: str
    steps: list[MigrationStep] = Field(default_factory=list)
    error: str | None = None
