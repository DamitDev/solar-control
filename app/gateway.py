"""OpenAI-compatible API gateway with routing and load balancing.

All routing state is stored in Redis for multi-replica consistency.
"""

import aiohttp
import asyncio
import logging
import re
import uuid
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
from datetime import datetime, timezone

from app.config import settings
from app.database.hosts import host_db
from app.models import HostStatus
from app.redis_state import registry_store, health_store, routing_store

logger = logging.getLogger(__name__)


class OpenAIGateway:

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self._bg_tasks: List[asyncio.Task] = []
        self._stop_event: Optional[asyncio.Event] = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    # -------------------------
    # Model registry
    # -------------------------

    async def refresh_model_registry(self):
        """Refresh the model registry from all hosts and store in Redis."""
        await self._ensure_session()
        if not self.session:
            return

        from app.socketio_app.host_handlers import (
            get_host_instances,
            is_host_connected,
        )

        new_model_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        hosts = await host_db.get_all_hosts()

        ws_hosts = []
        http_hosts = []
        for host in hosts:
            if await is_host_connected(host.id):
                ws_hosts.append(host)
            else:
                http_hosts.append(host)

        # WebSocket-connected hosts (instant, from cache)
        for host in ws_hosts:
            ws_instances = await get_host_instances(host.id)
            await host_db.update_host_status(host.id, HostStatus.ONLINE)

            for instance in ws_instances:
                if instance.get("status") == "running":
                    alias = instance.get("alias", "unknown")
                    port = instance.get("port")
                    if not port:
                        continue

                    host_base = host.url.rsplit(":", 1)[0]
                    instance_url = f"{host_base}:{port}"
                    supported_endpoints = instance.get(
                        "supported_endpoints",
                        ["/v1/chat/completions", "/v1/completions", "/v1/models"],
                    )
                    backend_type = instance.get("backend_type", "llamacpp")

                    entry = {
                        "host_id": host.id,
                        "instance_id": instance["id"],
                        "url": instance_url,
                        "api_key": host.api_key,
                        "model_alias": alias,
                        "supported_endpoints": supported_endpoints,
                        "backend_type": backend_type,
                    }
                    new_model_map[alias].append(entry)

        # HTTP hosts in parallel
        if http_hosts:

            async def poll_host(host):
                result_entries = []
                try:
                    url = f"{host.url}/instances"
                    headers = {"X-API-Key": host.api_key}
                    async with self.session.get(
                        url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            instances = await response.json()
                            prev_status = host.status
                            await host_db.update_host_status(host.id, HostStatus.ONLINE)
                            if prev_status != HostStatus.ONLINE:
                                await self._notify_host_online(host)
                            for instance in instances:
                                if instance.get("status") == "running":
                                    alias = instance["config"]["alias"]
                                    port = instance.get("port")
                                    host_base = host.url.rsplit(":", 1)[0]
                                    instance_url = f"{host_base}:{port}"
                                    instance_api_key = instance["config"]["api_key"]
                                    supported_endpoints = instance.get(
                                        "supported_endpoints",
                                        [
                                            "/v1/chat/completions",
                                            "/v1/completions",
                                            "/v1/models",
                                        ],
                                    )
                                    backend_type = instance.get("config", {}).get(
                                        "backend_type", "llamacpp"
                                    )
                                    entry = {
                                        "host_id": host.id,
                                        "instance_id": instance["id"],
                                        "url": instance_url,
                                        "api_key": instance_api_key,
                                        "model_alias": alias,
                                        "supported_endpoints": supported_endpoints,
                                        "backend_type": backend_type,
                                    }
                                    result_entries.append((alias, entry))
                        else:
                            await host_db.update_host_status(host.id, HostStatus.ERROR)
                except Exception:
                    cached = await get_host_instances(host.id)
                    if cached:
                        for instance in cached:
                            if instance.get("status") == "running":
                                alias = instance.get("alias", "unknown")
                                port = instance.get("port")
                                if not port:
                                    continue
                                host_base = host.url.rsplit(":", 1)[0]
                                instance_url = f"{host_base}:{port}"
                                entry = {
                                    "host_id": host.id,
                                    "instance_id": instance["id"],
                                    "url": instance_url,
                                    "api_key": host.api_key,
                                    "model_alias": alias,
                                    "supported_endpoints": instance.get(
                                        "supported_endpoints",
                                        [
                                            "/v1/chat/completions",
                                            "/v1/completions",
                                            "/v1/models",
                                        ],
                                    ),
                                    "backend_type": instance.get(
                                        "backend_type", "llamacpp"
                                    ),
                                }
                                result_entries.append((alias, entry))
                    else:
                        await host_db.update_host_status(host.id, HostStatus.OFFLINE)
                return result_entries

            results = await asyncio.gather(
                *[poll_host(h) for h in http_hosts], return_exceptions=True
            )
            for result in results:
                if isinstance(result, Exception):
                    continue
                for alias, entry in result:
                    new_model_map[alias].append(entry)

        # Store in Redis
        await registry_store.set_registry(dict(new_model_map))

    async def _notify_host_online(self, host):
        """Emit host_status to WebUI when HTTP polling discovers a host is online."""
        from app.socketio_app.server import sio

        try:
            refreshed = await host_db.get_host(host.id)
            h = refreshed or host
            await sio.emit(
                "host_status",
                {
                    "host_id": h.id,
                    "name": h.name,
                    "status": "online",
                    "url": h.url,
                    "memory": h.memory.model_dump() if h.memory else None,
                    "gpu_type": h.gpu_type,
                    "roles": h.roles,
                    "disk_total_gb": h.disk_total_gb,
                    "disk_used_gb": h.disk_used_gb,
                    "disk_available_gb": h.disk_available_gb,
                    "memory_available_gb": h.memory_available_gb,
                    "connected": False,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                namespace="/webui",
            )
        except Exception as e:
            logger.debug("Failed to notify WebUI of host online: %s", e)

    # -------------------------
    # Background tasks
    # -------------------------

    async def start_background_tasks(self):
        if self._stop_event is not None:
            return
        self._stop_event = asyncio.Event()
        await self.refresh_model_registry()
        self._bg_tasks = [
            asyncio.create_task(self._registry_refresh_loop(), name="registry_refresh"),
            asyncio.create_task(self._health_probe_loop(), name="health_probe"),
        ]

    async def stop_background_tasks(self):
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

    async def _registry_refresh_loop(self):
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self.refresh_model_registry()
            except Exception:
                pass
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=settings.registry_refresh_interval_s,
                )
            except asyncio.TimeoutError:
                pass

    async def _health_probe_loop(self):
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self._probe_all_instances_once()
            except Exception:
                pass
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=settings.health_check_interval_s,
                )
            except asyncio.TimeoutError:
                pass

    async def _probe_all_instances_once(self):
        registry = await registry_store.get_registry()
        instances: List[Tuple[str, str, str]] = []  # (host_id, instance_id, url)
        for alias, inst_list in registry.items():
            for inst in inst_list:
                instances.append(
                    (inst["host_id"], inst["instance_id"], inst.get("url", ""))
                )

        sem = asyncio.Semaphore(20)

        async def _probe_one(host_id: str, instance_id: str, url: str):
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

    # -------------------------
    # Helpers
    # -------------------------

    def _extract_usage_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        usage = result.get("usage") if isinstance(result, dict) else None
        if not isinstance(usage, dict):
            return {}
        out: Dict[str, Any] = {}
        if isinstance(usage.get("prompt_tokens"), (int, float)):
            out["prompt_tokens"] = int(usage["prompt_tokens"])
        if isinstance(usage.get("completion_tokens"), (int, float)):
            out["completion_tokens"] = int(usage["completion_tokens"])
        if isinstance(usage.get("total_tokens"), (int, float)):
            out["total_tokens"] = int(usage["total_tokens"])
        return out

    async def _fetch_last_generation_metrics(
        self, host_id: str, instance_id: str
    ) -> Dict[str, Any]:
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
                out: Dict[str, Any] = {}
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

    async def get_available_models(self) -> List[Dict[str, Any]]:
        await self._ensure_session()
        if not self.session:
            return []

        registry = await registry_store.get_registry()
        models_dict = {}

        for alias, instances in registry.items():
            if not instances:
                continue
            instance = instances[0]
            try:
                url = f"{instance['url']}/v1/models"
                headers = {"Authorization": f"Bearer {instance['api_key']}"}
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

    async def _resolve_model_name(self, model: str) -> Optional[str]:
        registry = await registry_store.get_registry()
        if model in registry and registry[model]:
            return model
        matching = [m for m in registry if m.startswith(model) and registry[m]]
        if matching:
            return sorted(matching)[0]
        return None

    def _parse_model_size(self, alias: str) -> Optional[float]:
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
        exclude_keys: Optional[Set[str]] = None,
        required_endpoint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select the best instance for a model using host-aware load balancing.

        Reads all state from Redis for cross-replica consistency.
        """
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
                if required_endpoint in inst.get("supported_endpoints", [])
            ]
            if not available:
                return None

        # Health filtering
        healthy = []
        fallback = []
        for inst in available:
            ikey = f"{inst['host_id']}-{inst['instance_id']}"
            if exclude_keys and ikey in exclude_keys:
                continue
            is_h = await health_store.is_healthy(
                inst["host_id"], inst["instance_id"], health_ttl_s=settings.health_ttl_s
            )
            if is_h:
                healthy.append(inst)
            else:
                fallback.append(inst)

        candidates = healthy if healthy else fallback
        if not candidates:
            return None

        # Group by host
        host_to_instances: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for inst in candidates:
            host_to_instances[inst["host_id"]].append(inst)

        candidate_host_ids = list(host_to_instances.keys())

        # Prefer hosts with zero active requests
        free_hosts = []
        for hid in candidate_host_ids:
            count = await routing_store.get_host_active(hid)
            if count == 0:
                free_hosts.append(hid)

        if free_hosts:

            async def host_name(hid: str) -> str:
                h = await host_db.get_host(hid)
                return h.name if h and h.name else hid

            names = {hid: await host_name(hid) for hid in free_hosts}
            chosen_host = sorted(free_hosts, key=lambda h: names.get(h, h))[0]
        else:
            # All busy - choose smallest active weight
            host_weights = {}
            for hid in candidate_host_ids:
                host_weights[hid] = await routing_store.get_weight(hid)

            min_weight = min(host_weights.values()) if host_weights else 0.0
            min_hosts = [hid for hid, w in host_weights.items() if w == min_weight]

            async def host_name(hid: str) -> str:
                h = await host_db.get_host(hid)
                return h.name if h and h.name else hid

            names = {hid: await host_name(hid) for hid in min_hosts}
            chosen_host = sorted(min_hosts, key=lambda h: names.get(h, h))[0]

        # Pick instance on chosen host (least active, then round-robin)
        host_insts = host_to_instances[chosen_host]
        if len(host_insts) == 1:
            return host_insts[0]

        min_active = float("inf")
        best = []
        for inst in host_insts:
            count = await routing_store.get_active(inst["host_id"], inst["instance_id"])
            if count < min_active:
                min_active = count
                best = [inst]
            elif count == min_active:
                best.append(inst)

        if len(best) == 1:
            return best[0]

        # Round-robin tiebreak
        rr_idx = await routing_store.next_rr_index(resolved_model)
        return best[rr_idx % len(best)]

    # -------------------------
    # Routing
    # -------------------------

    async def _broadcast_routing_event(
        self, event_data: dict, *, endpoint_id: Optional[str] = None
    ):
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

    async def route_request(
        self,
        model: str,
        endpoint: str,
        data: Dict[str, Any],
        client_ip: str = "unknown",
        required_endpoint: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            },
            endpoint_id=endpoint_id,
        )

        await self._ensure_session()
        if not self.session:
            raise RuntimeError("Failed to create aiohttp session")

        attempted: Set[str] = set()
        last_error: Optional[Exception] = None
        retried_once = False
        filter_endpoint = required_endpoint or endpoint

        for attempt in range(max(1, int(settings.route_max_attempts))):
            instance = await self._get_next_instance(
                model,
                exclude_keys=attempted,
                required_endpoint=filter_endpoint,
            )
            if not instance:
                if not retried_once:
                    retried_once = True
                    attempted.clear()
                    try:
                        await self.refresh_model_registry()
                    except Exception:
                        pass
                    delay = max(0.0, float(settings.route_retry_delay_s))
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                break

            instance_key = f"{instance['host_id']}-{instance['instance_id']}"
            attempted.add(instance_key)
            weight = self._parse_model_size(instance["model_alias"])

            # Atomically track active state in Redis
            await routing_store.increment_active(
                instance["host_id"], instance["instance_id"]
            )
            await routing_store.increment_host_active(instance["host_id"])
            if weight is not None:
                await routing_store.add_weight(instance["host_id"], weight)

            host = await host_db.get_host(instance["host_id"])
            host_name = host.name if host else "unknown"

            try:
                await self._broadcast_routing_event(
                    {
                        "type": "request_routed",
                        "data": {
                            "request_id": request_id,
                            "model": model,
                            "resolved_model": instance["model_alias"],
                            "host_id": instance["host_id"],
                            "host_name": host_name,
                            "instance_id": instance["instance_id"],
                            "instance_url": instance["url"],
                            "client_ip": client_ip,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "attempt": attempt + 1,
                        },
                    },
                    endpoint_id=endpoint_id,
                )

                url = f"{instance['url']}{endpoint}"
                headers = {
                    "Authorization": f"Bearer {instance['api_key']}",
                    "Content-Type": "application/json",
                }
                timeout = aiohttp.ClientTimeout(
                    total=None, connect=settings.route_connect_timeout_s
                )

                async with self.session.post(
                    url, json=data, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        await health_store.mark_healthy(
                            instance["host_id"],
                            instance["instance_id"],
                            ttl_s=settings.health_ttl_s + 2,
                        )
                        result = await response.json()
                        duration = time.time() - start_time

                        usage_fields = self._extract_usage_from_result(result)
                        if (
                            "prompt_tokens" not in usage_fields
                            or "completion_tokens" not in usage_fields
                        ):
                            host_metrics = await self._fetch_last_generation_metrics(
                                instance["host_id"], instance["instance_id"]
                            )
                            usage_fields = {**usage_fields, **host_metrics}

                        base_data = {
                            "request_id": request_id,
                            "model": model,
                            "host_id": instance["host_id"],
                            "instance_id": instance["instance_id"],
                            "duration": duration,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        base_data.update(usage_fields)
                        await self._broadcast_routing_event(
                            {"type": "request_success", "data": base_data},
                            endpoint_id=endpoint_id,
                        )
                        return result
                    else:
                        error_text = await response.text()
                        duration = time.time() - start_time
                        await self._broadcast_routing_event(
                            {
                                "type": "request_error",
                                "data": {
                                    "request_id": request_id,
                                    "model": model,
                                    "host_id": instance["host_id"],
                                    "instance_id": instance["instance_id"],
                                    "error_message": f"Request failed: {response.status} - {error_text}",
                                    "duration": duration,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                },
                            },
                            endpoint_id=endpoint_id,
                        )
                        raise Exception(
                            f"Request failed: {response.status} - {error_text}"
                        )

            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                await health_store.mark_failed(
                    instance["host_id"],
                    instance["instance_id"],
                    cooldown_s=settings.health_cooldown_s,
                )
                last_error = e
                await self._broadcast_routing_event(
                    {
                        "type": "request_reroute",
                        "data": {
                            "request_id": request_id,
                            "model": model,
                            "host_id": instance.get("host_id"),
                            "instance_id": instance.get("instance_id"),
                            "reason": "connect_error",
                            "attempt": attempt + 1,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    },
                    endpoint_id=endpoint_id,
                )
            except Exception as e:
                duration = time.time() - start_time
                await self._broadcast_routing_event(
                    {
                        "type": "request_error",
                        "data": {
                            "request_id": request_id,
                            "model": model,
                            "host_id": instance.get("host_id") if instance else None,
                            "instance_id": (
                                instance.get("instance_id") if instance else None
                            ),
                            "error_message": str(e),
                            "duration": duration,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    },
                    endpoint_id=endpoint_id,
                )
                raise
            finally:
                try:
                    await routing_store.decrement_active(
                        instance["host_id"], instance["instance_id"]
                    )
                    await routing_store.decrement_host_active(instance["host_id"])
                    if weight is not None:
                        await routing_store.remove_weight(instance["host_id"], weight)
                except Exception:
                    pass

        error_msg = (
            f"Model '{model}' not found or no instances available"
            if not attempted
            else f"Failed to connect to model '{model}' after {len(attempted)} attempts: {last_error}"
        )
        await self._broadcast_routing_event(
            {
                "type": "request_error",
                "data": {
                    "request_id": request_id,
                    "model": model,
                    "error_message": error_msg,
                    "client_ip": client_ip,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            },
            endpoint_id=endpoint_id,
        )
        if attempted:
            raise Exception(error_msg)
        raise ValueError(error_msg)

    async def stream_request(
        self,
        model: str,
        endpoint: str,
        data: Dict[str, Any],
        client_ip: str = "unknown",
        required_endpoint: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ):
        request_id = str(uuid.uuid4())
        start_time = time.time()

        await self._broadcast_routing_event(
            {
                "type": "request_start",
                "data": {
                    "request_id": request_id,
                    "model": model,
                    "endpoint": endpoint,
                    "stream": True,
                    "client_ip": client_ip,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            },
            endpoint_id=endpoint_id,
        )

        await self._ensure_session()
        if not self.session:
            raise RuntimeError("Failed to create aiohttp session")

        filter_endpoint = required_endpoint or endpoint
        attempted: Set[str] = set()
        last_error: Optional[Exception] = None
        retried_once = False

        for attempt in range(max(1, int(settings.route_max_attempts))):
            instance = await self._get_next_instance(
                model,
                exclude_keys=attempted,
                required_endpoint=filter_endpoint,
            )
            if not instance:
                if not retried_once:
                    retried_once = True
                    attempted.clear()
                    try:
                        await self.refresh_model_registry()
                    except Exception:
                        pass
                    delay = max(0.0, float(settings.route_retry_delay_s))
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
                break

            instance_key = f"{instance['host_id']}-{instance['instance_id']}"
            attempted.add(instance_key)
            weight = self._parse_model_size(instance["model_alias"])

            await routing_store.increment_active(
                instance["host_id"], instance["instance_id"]
            )
            await routing_store.increment_host_active(instance["host_id"])
            if weight is not None:
                await routing_store.add_weight(instance["host_id"], weight)

            host = await host_db.get_host(instance["host_id"])
            host_name = host.name if host else "unknown"

            try:
                await self._broadcast_routing_event(
                    {
                        "type": "request_routed",
                        "data": {
                            "request_id": request_id,
                            "model": model,
                            "resolved_model": instance["model_alias"],
                            "host_id": instance["host_id"],
                            "host_name": host_name,
                            "instance_id": instance["instance_id"],
                            "instance_url": instance["url"],
                            "client_ip": client_ip,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "attempt": attempt + 1,
                        },
                    },
                    endpoint_id=endpoint_id,
                )

                url = f"{instance['url']}{endpoint}"
                headers = {
                    "Authorization": f"Bearer {instance['api_key']}",
                    "Content-Type": "application/json",
                }
                timeout = aiohttp.ClientTimeout(
                    total=None, connect=settings.route_connect_timeout_s
                )

                async with self.session.post(
                    url, json=data, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        await health_store.mark_healthy(
                            instance["host_id"],
                            instance["instance_id"],
                            ttl_s=settings.health_ttl_s + 2,
                        )
                        async for line in response.content:
                            yield line

                        duration = time.time() - start_time
                        usage_fields = await self._fetch_last_generation_metrics(
                            instance["host_id"], instance["instance_id"]
                        )
                        base_data = {
                            "request_id": request_id,
                            "model": model,
                            "host_id": instance["host_id"],
                            "instance_id": instance["instance_id"],
                            "duration": duration,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        base_data.update(usage_fields)
                        await self._broadcast_routing_event(
                            {"type": "request_success", "data": base_data},
                            endpoint_id=endpoint_id,
                        )
                        return
                    else:
                        error_text = await response.text()
                        duration = time.time() - start_time
                        await self._broadcast_routing_event(
                            {
                                "type": "request_error",
                                "data": {
                                    "request_id": request_id,
                                    "model": model,
                                    "host_id": instance["host_id"],
                                    "instance_id": instance["instance_id"],
                                    "error_message": f"Request failed: {response.status} - {error_text}",
                                    "duration": duration,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                },
                            },
                            endpoint_id=endpoint_id,
                        )
                        raise Exception(
                            f"Request failed: {response.status} - {error_text}"
                        )

            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                await health_store.mark_failed(
                    instance["host_id"],
                    instance["instance_id"],
                    cooldown_s=settings.health_cooldown_s,
                )
                last_error = e
                await self._broadcast_routing_event(
                    {
                        "type": "request_reroute",
                        "data": {
                            "request_id": request_id,
                            "model": model,
                            "host_id": instance.get("host_id"),
                            "instance_id": instance.get("instance_id"),
                            "reason": "connect_error",
                            "attempt": attempt + 1,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    },
                    endpoint_id=endpoint_id,
                )
            except Exception as e:
                duration = time.time() - start_time
                await self._broadcast_routing_event(
                    {
                        "type": "request_error",
                        "data": {
                            "request_id": request_id,
                            "model": model,
                            "host_id": instance.get("host_id") if instance else None,
                            "instance_id": (
                                instance.get("instance_id") if instance else None
                            ),
                            "error_message": str(e),
                            "duration": duration,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    },
                    endpoint_id=endpoint_id,
                )
                raise
            finally:
                try:
                    await routing_store.decrement_active(
                        instance["host_id"], instance["instance_id"]
                    )
                    await routing_store.decrement_host_active(instance["host_id"])
                    if weight is not None:
                        await routing_store.remove_weight(instance["host_id"], weight)
                except Exception:
                    pass

        error_msg = (
            f"Model '{model}' not found or no instances available"
            if not attempted
            else f"Failed to connect to model '{model}' after {len(attempted)} attempts: {last_error}"
        )
        await self._broadcast_routing_event(
            {
                "type": "request_error",
                "data": {
                    "request_id": request_id,
                    "model": model,
                    "error_message": error_msg,
                    "client_ip": client_ip,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            },
            endpoint_id=endpoint_id,
        )
        if attempted:
            raise Exception(error_msg)
        raise ValueError(error_msg)


gateway = OpenAIGateway()
