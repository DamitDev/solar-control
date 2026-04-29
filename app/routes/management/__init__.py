from fastapi import APIRouter
from . import hosts, endpoints, gateway, models

router = APIRouter(prefix="/api", tags=["management"])
router.include_router(hosts.router)
router.include_router(endpoints.router)
router.include_router(gateway.router)
router.include_router(models.router)
