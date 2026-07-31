"""PostgreSQL-backed job CRUD operations using SQLAlchemy ORM."""

from typing import Any
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, delete, update

from app.models.job import Job, JobStatus
from .connection import get_session_factory
from .tables import JobRow

# How long to retain terminal (completed/failed/cancelled) jobs in the
# active-job aggregation view after they finish.
_TERMINAL_RETENTION_MINUTES = 15

# Cap on jobs returned per host. Jobs stuck in pending/running (e.g. the host
# died mid-run) are never aged out, so without a cap the host_status payload
# could grow without bound.
_ACTIVE_JOB_LIMIT = 50


class JobDB:
    """Database-backed job management."""

    def _session(self):
        return get_session_factory()()

    def _row_to_job(self, row: JobRow) -> Job:
        return Job(
            host_id=row.host_id,
            id=row.id,
            status=JobStatus(row.status),
            payload=row.payload or {},
            result=row.result,
            error_message=row.error_message,
            correlation_id=row.correlation_id,
            submission_id=row.submission_id,
            current_step_name=row.current_step_name,
            current_step_index=row.current_step_index,
            created_at=row.created_at,
            updated_at=row.updated_at,
            completed_at=row.completed_at,
        )

    def _job_to_dict(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.id,
            "host_id": job.host_id,
            "status": job.status.value,
            "payload": job.payload,
            "result": job.result,
            "error_message": job.error_message,
            "correlation_id": job.correlation_id,
            "submission_id": job.submission_id,
            "current_step_name": job.current_step_name,
            "current_step_index": job.current_step_index,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "completed_at": job.completed_at,
        }

    async def add_job(self, job: Job) -> Job:
        """Insert a new job record."""
        async with self._session() as session:
            session.add(JobRow(**self._job_to_dict(job)))
            await session.commit()
        return job

    async def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        async with self._session() as session:
            row = await session.get(JobRow, job_id)
            return self._row_to_job(row) if row else None

    async def get_jobs_by_host(self, host_id: str) -> list[Job]:
        """Get all jobs for a given host."""
        async with self._session() as session:
            result = await session.execute(
                select(JobRow)
                .where(JobRow.host_id == host_id)
                .order_by(JobRow.created_at)
            )
            return [self._row_to_job(row) for row in result.scalars()]

    async def get_jobs_by_status(self, status: JobStatus) -> list[Job]:
        """Get all jobs with a given status."""
        async with self._session() as session:
            result = await session.execute(
                select(JobRow)
                .where(JobRow.status == status.value)
                .order_by(JobRow.created_at)
            )
            return [self._row_to_job(row) for row in result.scalars()]

    async def get_all_jobs(self, *, limit: int = 100, offset: int = 0) -> list[Job]:
        """Get all jobs with pagination."""
        async with self._session() as session:
            result = await session.execute(
                select(JobRow)
                .order_by(JobRow.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return [self._row_to_job(row) for row in result.scalars()]

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Update job status and optionally set result or error."""
        values: dict[str, Any] = {
            "status": status.value,
            "updated_at": datetime.now(timezone.utc),
        }
        if result is not None:
            values["result"] = result
        if error_message is not None:
            values["error_message"] = error_message
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            values["completed_at"] = datetime.now(timezone.utc)

        async with self._session() as session:
            result = await session.execute(
                update(JobRow).where(JobRow.id == job_id).values(**values)
            )
            await session.commit()
            return result.rowcount > 0

    async def update_job_host(self, job_id: str, host_id: str) -> bool:
        """Update the host assigned to a job."""
        async with self._session() as session:
            result = await session.execute(
                update(JobRow)
                .where(JobRow.id == job_id)
                .values(host_id=host_id, updated_at=datetime.now(timezone.utc))
            )
            await session.commit()
            return result.rowcount > 0

    async def update_job_step(
        self,
        job_id: str,
        *,
        step_name: str | None = None,
        step_index: int | None = None,
    ) -> bool:
        """Update the current pipeline step for a job."""
        values: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if step_name is not None:
            values["current_step_name"] = step_name
        if step_index is not None:
            values["current_step_index"] = step_index

        async with self._session() as session:
            result = await session.execute(
                update(JobRow).where(JobRow.id == job_id).values(**values)
            )
            await session.commit()
            return result.rowcount > 0

    async def get_active_by_host(
        self,
        host_id: str,
        *,
        terminal_retention_minutes: int = _TERMINAL_RETENTION_MINUTES,
        limit: int = _ACTIVE_JOB_LIMIT,
    ) -> list[Job]:
        """Get active and recently-terminal jobs for a given host.

        Returns jobs that are:
        - Non-terminal (pending, running), OR
        - Terminal (completed, failed, cancelled) within the retention window.

        Active jobs are ordered first, then by creation time descending, so a
        long-running job is never pushed past ``limit`` by newer terminal or
        stuck rows. The result feeds every host_status broadcast, so it must
        stay bounded.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=terminal_retention_minutes
        )
        terminal_statuses = [
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ]

        async with self._session() as session:
            result = await session.execute(
                select(JobRow)
                .where(JobRow.host_id == host_id)
                .where(
                    (JobRow.status.notin_(terminal_statuses))
                    | (
                        (JobRow.status.in_(terminal_statuses))
                        & (JobRow.completed_at >= cutoff)
                    )
                )
                .order_by(
                    JobRow.status.in_(terminal_statuses),
                    JobRow.created_at.desc(),
                )
                .limit(limit)
            )
            return [self._row_to_job(row) for row in result.scalars()]

    async def remove_job(self, job_id: str) -> bool:
        """Delete a job record."""
        async with self._session() as session:
            result = await session.execute(delete(JobRow).where(JobRow.id == job_id))
            await session.commit()
            return result.rowcount > 0


job_db = JobDB()
