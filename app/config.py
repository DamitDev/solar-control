from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000

    management_api_key: str = "change-me-management"

    registry_refresh_interval_s: float = 2.0
    health_check_interval_s: float = 1.0
    health_ttl_s: float = 3.0
    health_cooldown_s: float = 5.0
    route_connect_timeout_s: float = 0.5
    route_max_attempts: int = 3
    route_retry_delay_s: float = 0.15

    health_probe_use_http: bool = False
    health_probe_http_path: str = "/v1/models"

    disconnect_grace_period_s: float = 15.0
    reconnect_request_interval_s: float = 30.0

    database_url: str = "postgresql://solar:solar@localhost:5432/solar_gateway"
    redis_url: str = "redis://localhost:6379/0"


settings = Settings()
