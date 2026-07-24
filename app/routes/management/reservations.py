"""Resource reservation API routes (S-038).

POST   /api/resources/reservations      — reserve resources
DELETE /api/resources/reservations/{id}  — release a reservation
"""

import logging

from fastapi import APIRouter

from app.models.reservation import (
    ReservationFailure,
    ReservationReleaseResponse,
    ReservationRequest,
    ReservationResponse,
)
from app.services.reservation import reserve_resources, release_reservation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resources/reservations", tags=["reservations"])


@router.post(
    "",
    response_model=ReservationResponse,
    responses={
        201: {"description": "Reservation created"},
        409: {
            "model": ReservationFailure,
            "description": "No capacity available",
        },
    },
    status_code=201,
)
async def create_reservation(
    request: ReservationRequest,
) -> ReservationResponse:
    """Reserve resources for a job across the Solar cluster.

    Solar Control evaluates all hosts, applies placement policy,
    optionally migrates lower-priority workloads to free capacity,
    and proxies the reservation to the selected host.

    Returns the reservation ID, assigned host, and resource details.
    On failure, returns a deterministic reason.
    """
    return await reserve_resources(request)


@router.delete(
    "/{reservation_id}",
    response_model=ReservationReleaseResponse,
    responses={
        200: {"description": "Reservation released"},
        404: {"description": "Reservation not found"},
    },
)
async def cancel_reservation(
    reservation_id: str,
) -> ReservationReleaseResponse:
    """Release a previously created reservation.

    Proxies DELETE /resources/reservations/{id} to the Solar Host
    and removes the reservation from Solar Control tracking.
    """
    return await release_reservation(reservation_id)
