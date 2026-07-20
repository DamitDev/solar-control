from fastapi import APIRouter
from app.jobs import router as jobs_router
from . import hosts, endpoints, gateway, models, resources

router = APIRouter(prefix="/api", tags=["management"])
router.include_router(hosts.router)
router.include_router(endpoints.router)
router.include_router(gateway.router)
router.include_router(models.router)
router.include_router(jobs_router.router)
router.include_router(resources.router)
