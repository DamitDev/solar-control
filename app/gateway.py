"""OpenAI-compatible API gateway with routing and load balancing.

All routing state is stored in Redis for multi-replica consistency.
"""

import aiohttp
import asyncio
import logging
import re
import uuid
import time
from typing import Any, AsyncIterator
from contextlib import asynccontextmanager
from collections import defaultdict
from datetime import datetime, timezone

from app.config import settings
from app.database.hosts import host_db
from app.models import HostStatus, RegistryEntry
from app.models.socketio import HostStatusPayload
from app.redis_state import registry_store, health_store, routing_store, host_store

logger = logging.getLogger(__name__)


def _task_done_callback(task: asyncio.Task) -> None:
    """Log exceptions from fire-and-forget tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error("Background task %s failed: %s", task.get_name(), exc)


_RETRYABLE_STATUSES = frozenset({502, 503, 504})


class OpenAIGateway:

    def __init__(self) -> None:
        self.session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()
        self._bg_tasks: list[asyncio.Task[None]] = []
        self._pending_tasks: set[asyncio.Task] = set()
        self._stop_event: asyncio.Event | None = None

    async def _ensure_session(self) -> None:
        if self.session is not None and not self.session.closed:
            return
        async with self._session_lock:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()
        for t in self._pending_tasks:
            t.cancel()
        self._pending_tasks.clear()

    def _spawn_task(self, coro, *, name: str | None = None) -> asyncio.Task:
        """Create a tracked fire-and-forget task with error logging."""
        task = asyncio.create_task(coro, name=name)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        task.add_done_callback(_task_done_callback)
        return task

    # ── Model registry ────────────────────────────────────────

    async def refresh_model_registry(self) -> None:
        """Refresh the model registry from all hosts and store in Redis."""
        await self._ensure_session()
        if not self.session:
            return

        from app.socketio_app.host_handlers import (
            get_host_instances,
            is_host_connected,
        )

        new_model_map: dict[str, list[RegistryEntry]] = defaultdict(list)
        hosts = await host_db.get_all_hosts()

        ws_hosts = []
        http_hosts = []
        for host in hosts:
            if await is_host_connected(host.id):
                ws_hosts.append(host)
            else:
                http_hosts.append(host)

        for host in ws_hosts:
            ws_instances = await get_host_instances(host.id)
            await host_db.update_host_status(host.id, HostStatus.ONLINE)

            for instance in ws_instances:
                if instance.get("status") == "running":
                    entry = RegistryEntry.from_ws_instance(
                        host.id, host.url, host.api_key, instance
                    )
                    if entry:
                        new_model_map[entry.model_alias].append(entry)

        if http_hosts:
            now = time.time()
            grace = settings.disconnect_grace_period_s
            reconnect_interval = settings.reconnect_request_interval_s

            grace_hosts = []
            poll_hosts = []

            for host in http_hosts:
                dc_ts = await host_store.get_disconnect_time(host.id)
                if dc_ts is not None and (now - dc_ts) < grace:
                    grace_hosts.append(host)
                else:
                    if dc_ts is not None:
                        last_req = await host_store.get_reconnect_request_time(host.id)
                        if last_req is None or (now - last_req) >= reconnect_interval:
                            self._spawn_task(
                                self._request_host_reconnect(host),
                                name=f"reconnect-{host.id[:8]}",
                            )
                    poll_hosts.append(host)

            for host in grace_hosts:
                cached = await get_host_instances(host.id)
                for instance in cached:
                    if instance.get("status") == "running":
                        entry = RegistryEntry.from_ws_instance(
                            host.id, host.url, host.api_key, instance
                        )
                        if entry:
                            new_model_map[entry.model_alias].append(entry)

            if poll_hosts:

                async def poll_host(host):
                    result_entries: list[RegistryEntry] = []
                    try:
                        url = f"{host.url}/instances"
                        headers = {"X-API-Key": host.api_key}
                        async with self.session.get(
                            url,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as response:
                            if response.status == 200:
                                instances = await response.json()
                                prev_status = host.status
                                await host_db.update_host_status(
                                    host.id, HostStatus.ONLINE
                                )
                                if prev_status != HostStatus.ONLINE:
                                    await self._notify_host_online(host)
                                for instance in instances:
                                    if instance.get("status") == "running":
                                        entry = RegistryEntry.from_http_instance(
                                            host.id, host.url, instance
                                        )
                                        if entry:
                                            result_entries.append(entry)
                            else:
                                await host_db.update_host_status(
                                    host.id, HostStatus.ERROR
                                )
                    except Exception:
                        cached = await get_host_instances(host.id)
                        if cached:
                            for instance in cached:
                                if instance.get("status") == "running":
                                    entry = RegistryEntry.from_ws_instance(
                                        host.id, host.url, host.api_key, instance
                                    )
                                    if entry:
                                        result_entries.append(entry)
                        else:
                            await host_db.update_host_status(
                                host.id, HostStatus.OFFLINE
                            )
                    return result_entries

                results = await asyncio.gather(
                    *[poll_host(h) for h in poll_hosts], return_exceptions=True
                )
                for result in results:
                    if isinstance(result, Exception):
                        continue
                    for entry in result:
                        new_model_map[entry.model_alias].append(entry)

        await registry_store.set_registry(dict(new_model_map))

    async def _notify_host_online(self, host) -> None:
        """Emit host_status to WebUI when HTTP polling discovers a host is online."""
        from app.socketio_app.server import sio

        try:
            refreshed = await host_db.get_host(host.id)
            h = refreshed or host
            payload = HostStatusPayload.from_host(h, connected=False)
            await sio.emit("host_status", payload.model_dump(), namespace="/webui")
        except Exception as e:
            logger.debug("Failed to notify WebUI of host online: %s", e)

    async def _request_host_reconnect(self, host) -> None:
        """Ask a disconnected host to re-establish its Socket.IO connection."""
        await self._ensure_session()
        if not self.session:
            return
        try:
            await host_store.set_reconnect_request_time(host.id)
            url = f"{host.url}/reconnect"
            headers = {"X-API-Key": host.api_key}
            async with self.session.post(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(
                        "Reconnect request to '%s' (%s): %s",
                        host.name,
                        host.id,
                        data.get("status"),
                    )
                else:
                    logger.debug(
                        "Reconnect request to '%s' failed: HTTP %s",
                        host.name,
                        resp.status,
                    )
        except Exception as e:
            logger.debug("Reconnect request to '%s' error: %s", host.name, e)

    # ── Background tasks ──────────────────────────────────────

    async def start_background_tasks(self) -> None:
        if self._stop_event is not None:
            return
        self._stop_event = asyncio.Event()
        await self.refresh_model_registry()
        self._bg_tasks = [
            asyncio.create_task(self._registry_refresh_loop(), name="registry_refresh"),
            asyncio.create_task(self._health_probe_loop(), name="health_probe"),
        ]

    async def stop_background_tasks(self) -> None:
        if self._stop_event is None:
            return
        self._stop_event.set()
        for t in self._bg_tasks:
            t.cancel()
        for t in self._bg_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._bg_tasks = []
        self._stop_event = None

    async def _registry_refresh_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self.refresh_model_registry()
            except Exception:
                logger.exception("Registry refresh failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=settings.registry_refresh_interval_s,
                )
            except asyncio.TimeoutError:
                pass

    async def _health_probe_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self._probe_all_instances_once()
            except Exception:
                logger.exception("Health probe failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=settings.health_check_interval_s,
                )
            except asyncio.TimeoutError:
                pass

    async def _probe_all_instances_once(self) -> None:
        registry = await registry_store.get_registry()
        instances: list[tuple[str, str, str]] = []
        for inst_list in registry.values():
            for inst in inst_list:
                instances.append((inst.host_id, inst.instance_id, inst.url))

        sem = asyncio.Semaphore(20)

        async def _probe_one(host_id: str, instance_id: str, url: str) -> None:
            async with sem:
                ok = await self._tcp_connect_ok(url)
                if ok:
                    await health_store.mark_healthy(
                        host_id, instance_id, ttl_s=settings.health_ttl_s + 2
                    )

        await asyncio.gather(
            *[_probe_one(h, i, u) for h, i, u in instances],
            return_exceptions=True,
        )

    async def _tcp_connect_ok(self, url: str) -> bool:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            hostname = parsed.hostname
            port = parsed.port
            if not hostname or not port:
                return False
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(hostname, port),
                timeout=settings.health_check_interval_s / 2,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    # ── Helpers ────────────────────────────────────────────────

    def _extract_usage_from_result(self, result: dict[str, Any]) -> dict[str, Any]:
        usage = result.get("usage") if isinstance(result, dict) else None
        if not isinstance(usage, dict):
            return {}
        out: dict[str, Any] = {}
        if isinstance(usage.get("prompt_tokens"), (int, float)):
            out["prompt_tokens"] = int(usage["prompt_tokens"])
        if isinstance(usage.get("completion_tokens"), (int, float)):
            out["completion_tokens"] = int(usage["completion_tokens"])
        if isinstance(usage.get("total_tokens"), (int, float)):
            out["total_tokens"] = int(usage["total_tokens"])
        return out

    async def _fetch_last_generation_metrics(
        self, host_id: str, instance_id: str
    ) -> dict[str, Any]:
        try:
            await self._ensure_session()
            if not self.session:
                return {}
            host = await host_db.get_host(host_id)
            if not host:
                return {}
            url = f"{host.url}/instances/{instance_id}/last-generation"
            headers = {"X-API-Key": host.api_key}
            timeout = aiohttp.ClientTimeout(total=3)
            async with self.session.get(url, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()
                out: dict[str, Any] = {}
                if isinstance(data.get("prompt_tokens"), (int, float)):
                    out["prompt_tokens"] = int(data["prompt_tokens"])
                if isinstance(data.get("generated_tokens"), (int, float)):
                    out["completion_tokens"] = int(data["generated_tokens"])
                if "prompt_tokens" in out and "completion_tokens" in out:
                    out["total_tokens"] = (
                        out["prompt_tokens"] + out["completion_tokens"]
                    )
                if isinstance(data.get("decode_tps"), (int, float)):
                    out["decode_tps"] = float(data["decode_tps"])
                if isinstance(data.get("decode_ms_per_token"), (int, float)):
                    out["decode_ms_per_token"] = float(data["decode_ms_per_token"])
                return out
        except Exception:
            return {}

    async def get_available_models(self) -> list[dict[str, Any]]:
        await self._ensure_session()
        if not self.session:
            return []

        registry = await registry_store.get_registry()
        models_dict: dict[str, dict[str, Any]] = {}

        for alias, instances in registry.items():
            if not instances:
                continue
            instance = instances[0]
            try:
                url = f"{instance.url}/v1/models"
                headers = {"Authorization": f"Bearer {instance.api_key}"}
                async with self.session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and data["data"]:
                            for model in data["data"]:
                                model_id = model.get("id", alias)
                                if model_id not in models_dict:
                                    models_dict[model_id] = model
            except Exception:
                if alias not in models_dict:
                    models_dict[alias] = {
                        "id": alias,
                        "object": "model",
                        "created": int(datetime.now(timezone.utc).timestamp()),
                        "owned_by": "solar",
                    }

        return list(models_dict.values())

    async def _resolve_model_name(self, model: str) -> str | None:
        registry = await registry_store.get_registry()
        if model in registry and registry[model]:
            return model
        matching = [m for m in registry if m.startswith(model) and registry[m]]
        if matching:
            return sorted(matching)[0]
        return None

    def _parse_model_size(self, alias: str) -> float | None:
        try:
            size_token = alias.rsplit(":", 1)[-1] if ":" in alias else alias
            match = re.fullmatch(
                r"(?:(\d+)\s*x\s*)?(\d+(?:\.\d+)?)\s*([bBmM])", size_token
            )
            if not match:
                return None
            multiplier_str, value_str, unit = match.groups()
            multiplier = int(multiplier_str) if multiplier_str else 1
            value = float(value_str)
            if unit.lower() == "b":
                return multiplier * value
            if unit.lower() == "m":
                return multiplier * (value / 1000.0)
            return None
        except Exception:
            return None

    async def _get_next_instance(
        self,
        model: str,
        *,
        exclude_keys: set[str] | None = None,
        required_endpoint: str | None = None,
    ) -> RegistryEntry | None:
        """Select the best instance for a model using host-aware load balancing."""
        resolved_model = await self._resolve_model_name(model)
        if not resolved_model:
            return None

        registry = await registry_store.get_registry()
        available = registry.get(resolved_model, [])
        if not available:
            return None

        if required_endpoint:
            available = [
                inst
                for inst in available
                if required_endpoint in inst.supported_endpoints
            ]
            if not available:
                return None

        healthy: list[RegistryEntry] = []
        fallback: list[RegistryEntry] = []
        for inst in available:
            ikey = f"{inst.host_id}-{inst.instance_id}"
            if exclude_keys and ikey in exclude_keys:
                continue
            is_h = await health_store.is_healthy(
                inst.host_id, inst.instance_id, health_ttl_s=settings.health_ttl_s
            )
            if is_h:
                healthy.append(inst)
            else:
                fallback.append(inst)

        candidates = healthy if healthy else fallback
        if not candidates:
            return None

        host_to_instances: dict[str, list[RegistryEntry]] = defaultdict(list)
        for inst in candidates:
            host_to_instances[inst.host_id].append(inst)

        candidate_host_ids = list(host_to_instances.keys())

        free_hosts: list[str] = []
        for hid in candidate_host_ids:
            count = await routing_store.get_host_active(hid)
            if count == 0:
                free_hosts.append(hid)

        if free_hosts:
            host_names = dict(
                zip(
                    free_hosts,
                    await asyncio.gather(
                        *[self._get_host_name(hid) for hid in free_hosts]
                    ),
                )
            )
            chosen_host = sorted(free_hosts, key=lambda h: host_names.get(h, h))[0]
        else:
            host_weights: dict[str, float] = {}
            for hid in candidate_host_ids:
                host_weights[hid] = await routing_store.get_weight(hid)

            min_weight = min(host_weights.values()) if host_weights else 0.0
            min_hosts = [hid for hid, w in host_weights.items() if w == min_weight]

            host_names = dict(
                zip(
                    min_hosts,
                    await asyncio.gather(
                        *[self._get_host_name(hid) for hid in min_hosts]
                    ),
                )
            )
            chosen_host = sorted(min_hosts, key=lambda h: host_names.get(h, h))[0]

        host_insts = host_to_instances[chosen_host]
        if len(host_insts) == 1:
            return host_insts[0]

        min_active = float("inf")
        best: list[RegistryEntry] = []
        for inst in host_insts:
            count = await routing_store.get_active(inst.host_id, inst.instance_id)
            if count < min_active:
                min_active = count
                best = [inst]
            elif count == min_active:
                best.append(inst)

        if len(best) == 1:
            return best[0]

        rr_idx = await routing_store.next_rr_index(resolved_model)
        return best[rr_idx % len(best)]

    async def _get_host_name(self, host_id: str) -> str:
        h = await host_db.get_host(host_id)
        return h.name if h and h.name else host_id

    # ── Routing infrastructure ────────────────────────────────

    async def _broadcast_routing_event(
        self, event_data: dict[str, Any], *, endpoint_id: str | None = None
    ) -> None:
        """Broadcast a routing event to WebUI via Socket.IO and log to database."""
        from dataclasses import asdict
        from app.database.logs import gateway_logger
        from app.socketio_app.webui_handlers import (
            broadcast_to_webui,
            broadcast_gateway_request,
        )

        try:
            summary = await gateway_logger.log_event(
                event_data, endpoint_id=endpoint_id
            )
            if summary:
                await broadcast_gateway_request(asdict(summary))
        except Exception as e:
            logger.error("Logging error: %s", e)

        try:
            event_type = event_data.get("type", "unknown")
            data = dict(event_data.get("data", {}))
            if endpoint_id is not None:
                data["endpoint_id"] = endpoint_id
            await broadcast_to_webui(event_type, data)
        except Exception as e:
            logger.warning("Failed to broadcast routing event to WebUI: %s", e)

    def _ts(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def _emit_success(
        self,
        request_id: str,
        model: str,
        instance: RegistryEntry,
        duration: float,
        usage_fields: dict[str, Any],
        endpoint_id: str | None,
    ) -> None:
        base_data: dict[str, Any] = {
            "request_id": request_id,
            "model": model,
            "host_id": instance.host_id,
            "instance_id": instance.instance_id,
            "duration": duration,
            "timestamp": self._ts(),
        }
        base_data.update(usage_fields)
        await self._broadcast_routing_event(
            {"type": "request_success", "data": base_data},
            endpoint_id=endpoint_id,
        )

    async def _emit_error(
        self,
        request_id: str,
        model: str,
        error_message: str,
        duration: float,
        endpoint_id: str | None,
        instance: RegistryEntry | None = None,
        client_ip: str | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "request_id": request_id,
            "model": model,
            "error_message": error_message,
            "duration": duration,
            "timestamp": self._ts(),
        }
        if instance:
            data["host_id"] = instance.host_id
            data["instance_id"] = instance.instance_id
        if client_ip:
            data["client_ip"] = client_ip
        await self._broadcast_routing_event(
            {"type": "request_error", "data": data},
            endpoint_id=endpoint_id,
        )

    async def _emit_reroute(
        self,
        request_id: str,
        model: str,
        instance: RegistryEntry,
        attempt: int,
        endpoint_id: str | None,
    ) -> None:
        await self._broadcast_routing_event(
            {
                "type": "request_reroute",
                "data": {
                    "request_id": request_id,
                    "model": model,
                    "host_id": instance.host_id,
                    "instance_id": instance.instance_id,
                    "reason": "connect_error",
                    "attempt": attempt,
                    "timestamp": self._ts(),
                },
            },
            endpoint_id=endpoint_id,
        )

    @asynccontextmanager
    async def _routing_context(
        self, instance: RegistryEntry, weight: float | None
    ) -> AsyncIterator[None]:
        """Track active routing state in Redis, cleaning up on exit."""
        await routing_store.increment_active(instance.host_id, instance.instance_id)
        await routing_store.increment_host_active(instance.host_id)
        if weight is not None:
            await routing_store.add_weight(instance.host_id, weight)
        try:
            yield
        finally:
            try:
                await routing_store.decrement_active(
                    instance.host_id, instance.instance_id
                )
            except Exception:
                logger.warning(
                    "Failed to decrement instance active for %s/%s",
                    instance.host_id,
                    instance.instance_id,
                )
            try:
                await routing_store.decrement_host_active(instance.host_id)
            except Exception:
                logger.warning(
                    "Failed to decrement host active for %s",
                    instance.host_id,
                )
            if weight is not None:
                try:
                    await routing_store.remove_weight(instance.host_id, weight)
                except Exception:
                    logger.warning(
                        "Failed to remove weight for %s",
                        instance.host_id,
                    )

    async def _find_instance_or_retry(
        self,
        model: str,
        filter_endpoint: str,
        attempted: set[str],
        retried_once_flag: list[bool],
    ) -> RegistryEntry | None:
        """Try to find an instance, with one registry-refresh retry."""
        instance = await self._get_next_instance(
            model, exclude_keys=attempted, required_endpoint=filter_endpoint
        )
        if instance:
            return instance

        if not retried_once_flag[0]:
            retried_once_flag[0] = True
            attempted.clear()
            try:
                await self.refresh_model_registry()
            except Exception:
                pass
            delay = max(0.0, float(settings.route_retry_delay_s))
            if delay > 0:
                await asyncio.sleep(delay)
            return await self._get_next_instance(
                model, exclude_keys=attempted, required_endpoint=filter_endpoint
            )
        return None

    def _make_route_timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(
            total=settings.route_total_timeout_s,
            connect=settings.route_connect_timeout_s,
        )

    # ── Public routing API ────────────────────────────────────

    async def route_request(
        self,
        model: str,
        endpoint: str,
        data: dict[str, Any],
        client_ip: str = "unknown",
        required_endpoint: str | None = None,
        endpoint_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        start_time = time.time()

        await self._broadcast_routing_event(
            {
                "type": "request_start",
                "data": {
                    "request_id": request_id,
                    "model": model,
                    "endpoint": endpoint,
                    "client_ip": client_ip,
                    "timestamp": self._ts(),
                },
            },
            endpoint_id=endpoint_id,
        )

        await self._ensure_session()
        if not self.session:
            raise RuntimeError("Failed to create aiohttp session")

        attempted: set[str] = set()
        last_error: Exception | None = None
        retried_once = [False]
        filter_endpoint = required_endpoint or endpoint

        for attempt in range(max(1, int(settings.route_max_attempts))):
            instance = await self._find_instance_or_retry(
                model, filter_endpoint, attempted, retried_once
            )
            if not instance:
                break

            instance_key = f"{instance.host_id}-{instance.instance_id}"
            attempted.add(instance_key)
            weight = self._parse_model_size(instance.model_alias)

            host = await host_db.get_host(instance.host_id)
            host_name = host.name if host else "unknown"

            async with self._routing_context(instance, weight):
                try:
                    await self._broadcast_routing_event(
                        {
                            "type": "request_routed",
                            "data": {
                                "request_id": request_id,
                                "model": model,
                                "resolved_model": instance.model_alias,
                                "host_id": instance.host_id,
                                "host_name": host_name,
                                "instance_id": instance.instance_id,
                                "instance_url": instance.url,
                                "client_ip": client_ip,
                                "timestamp": self._ts(),
                                "attempt": attempt + 1,
                            },
                        },
                        endpoint_id=endpoint_id,
                    )

                    url = f"{instance.url}{endpoint}"
                    headers = {
                        "Authorization": f"Bearer {instance.api_key}",
                        "Content-Type": "application/json",
                    }
                    timeout = self._make_route_timeout()

                    async with self.session.post(
                        url, json=data, headers=headers, timeout=timeout
                    ) as response:
                        if response.status == 200:
                            await health_store.mark_healthy(
                                instance.host_id,
                                instance.instance_id,
                                ttl_s=settings.health_ttl_s + 2,
                            )
                            result = await response.json()
                            duration = time.time() - start_time

                            usage_fields = self._extract_usage_from_result(result)
                            if (
                                "prompt_tokens" not in usage_fields
                                or "completion_tokens" not in usage_fields
                            ):
                                host_metrics = (
                                    await self._fetch_last_generation_metrics(
                                        instance.host_id, instance.instance_id
                                    )
                                )
                                usage_fields = {**usage_fields, **host_metrics}

                            await self._emit_success(
                                request_id,
                                model,
                                instance,
                                duration,
                                usage_fields,
                                endpoint_id,
                            )
                            return result
                        elif response.status in _RETRYABLE_STATUSES:
                            error_text = await response.text()
                            logger.warning(
                                "Retryable %d from %s: %s",
                                response.status,
                                instance.url,
                                error_text[:200],
                            )
                            await health_store.mark_failed(
                                instance.host_id,
                                instance.instance_id,
                                cooldown_s=settings.health_cooldown_s,
                            )
                            last_error = Exception(f"Upstream {response.status}")
                            await self._emit_reroute(
                                request_id,
                                model,
                                instance,
                                attempt + 1,
                                endpoint_id,
                            )
                            continue
                        else:
                            error_text = await response.text()
                            duration = time.time() - start_time
                            msg = f"Request failed: {response.status} - {error_text}"
                            await self._emit_error(
                                request_id,
                                model,
                                msg,
                                duration,
                                endpoint_id,
                                instance=instance,
                            )
                            raise Exception(msg)

                except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                    await health_store.mark_failed(
                        instance.host_id,
                        instance.instance_id,
                        cooldown_s=settings.health_cooldown_s,
                    )
                    last_error = e
                    await self._emit_reroute(
                        request_id, model, instance, attempt + 1, endpoint_id
                    )
                except Exception as e:
                    duration = time.time() - start_time
                    await self._emit_error(
                        request_id,
                        model,
                        str(e),
                        duration,
                        endpoint_id,
                        instance=instance,
                    )
                    raise

        error_msg = (
            f"Model '{model}' not found or no instances available"
            if not attempted
            else f"Failed to connect to model '{model}' after {len(attempted)} attempts: {last_error}"
        )
        await self._emit_error(
            request_id,
            model,
            error_msg,
            time.time() - start_time,
            endpoint_id,
            client_ip=client_ip,
        )
        if attempted:
            raise Exception(error_msg)
        raise ValueError(error_msg)

    async def stream_request(
        self,
        model: str,
        endpoint: str,
        data: dict[str, Any],
        client_ip: str = "unknown",
        required_endpoint: str | None = None,
        endpoint_id: str | None = None,
    ):
        request_id = str(uuid.uuid4())
        start_time = time.time()
        completed = False

        await self._broadcast_routing_event(
            {
                "type": "request_start",
                "data": {
                    "request_id": request_id,
                    "model": model,
                    "endpoint": endpoint,
                    "stream": True,
                    "client_ip": client_ip,
                    "timestamp": self._ts(),
                },
            },
            endpoint_id=endpoint_id,
        )

        await self._ensure_session()
        if not self.session:
            raise RuntimeError("Failed to create aiohttp session")

        filter_endpoint = required_endpoint or endpoint
        attempted: set[str] = set()
        last_error: Exception | None = None
        retried_once = [False]

        for attempt in range(max(1, int(settings.route_max_attempts))):
            instance = await self._find_instance_or_retry(
                model, filter_endpoint, attempted, retried_once
            )
            if not instance:
                break

            instance_key = f"{instance.host_id}-{instance.instance_id}"
            attempted.add(instance_key)
            weight = self._parse_model_size(instance.model_alias)

            host = await host_db.get_host(instance.host_id)
            host_name = host.name if host else "unknown"

            async with self._routing_context(instance, weight):
                try:
                    await self._broadcast_routing_event(
                        {
                            "type": "request_routed",
                            "data": {
                                "request_id": request_id,
                                "model": model,
                                "resolved_model": instance.model_alias,
                                "host_id": instance.host_id,
                                "host_name": host_name,
                                "instance_id": instance.instance_id,
                                "instance_url": instance.url,
                                "client_ip": client_ip,
                                "timestamp": self._ts(),
                                "attempt": attempt + 1,
                            },
                        },
                        endpoint_id=endpoint_id,
                    )

                    url = f"{instance.url}{endpoint}"
                    headers = {
                        "Authorization": f"Bearer {instance.api_key}",
                        "Content-Type": "application/json",
                    }
                    timeout = self._make_route_timeout()

                    async with self.session.post(
                        url, json=data, headers=headers, timeout=timeout
                    ) as response:
                        if response.status == 200:
                            await health_store.mark_healthy(
                                instance.host_id,
                                instance.instance_id,
                                ttl_s=settings.health_ttl_s + 2,
                            )
                            async for line in response.content:
                                yield line

                            completed = True
                            duration = time.time() - start_time
                            usage_fields = await self._fetch_last_generation_metrics(
                                instance.host_id, instance.instance_id
                            )
                            await self._emit_success(
                                request_id,
                                model,
                                instance,
                                duration,
                                usage_fields,
                                endpoint_id,
                            )
                            return
                        elif response.status in _RETRYABLE_STATUSES:
                            error_text = await response.text()
                            logger.warning(
                                "Retryable %d from %s: %s",
                                response.status,
                                instance.url,
                                error_text[:200],
                            )
                            await health_store.mark_failed(
                                instance.host_id,
                                instance.instance_id,
                                cooldown_s=settings.health_cooldown_s,
                            )
                            last_error = Exception(f"Upstream {response.status}")
                            await self._emit_reroute(
                                request_id,
                                model,
                                instance,
                                attempt + 1,
                                endpoint_id,
                            )
                            continue
                        else:
                            error_text = await response.text()
                            duration = time.time() - start_time
                            msg = f"Request failed: {response.status} - {error_text}"
                            await self._emit_error(
                                request_id,
                                model,
                                msg,
                                duration,
                                endpoint_id,
                                instance=instance,
                            )
                            raise Exception(msg)

                except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                    await health_store.mark_failed(
                        instance.host_id,
                        instance.instance_id,
                        cooldown_s=settings.health_cooldown_s,
                    )
                    last_error = e
                    await self._emit_reroute(
                        request_id, model, instance, attempt + 1, endpoint_id
                    )
                except GeneratorExit:
                    if not completed:
                        try:
                            await self._emit_error(
                                request_id,
                                model,
                                "Client disconnected",
                                time.time() - start_time,
                                endpoint_id,
                                instance=instance,
                                client_ip=client_ip,
                            )
                        except Exception:
                            pass
                    return
                except Exception as e:
                    duration = time.time() - start_time
                    await self._emit_error(
                        request_id,
                        model,
                        str(e),
                        duration,
                        endpoint_id,
                        instance=instance,
                    )
                    raise

        error_msg = (
            f"Model '{model}' not found or no instances available"
            if not attempted
            else f"Failed to connect to model '{model}' after {len(attempted)} attempts: {last_error}"
        )
        await self._emit_error(
            request_id,
            model,
            error_msg,
            time.time() - start_time,
            endpoint_id,
            client_ip=client_ip,
        )
        if attempted:
            raise Exception(error_msg)
        raise ValueError(error_msg)


gateway = OpenAIGateway()
