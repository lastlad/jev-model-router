import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger

from .core.config import RouterConfig, Tier, load_config
from .core.judge import JevJudge
from .core.ledger import Ledger
from .core.request import fingerprint_assistant
from .core.router import Router

log = logging.getLogger("jev_router")
META_KEY = "jev_router"
LOG_FILE = os.environ.get("JEV_ROUTER_LOG_FILE")
if os.environ.get("JEV_ROUTER_LOG_LEVEL"):
    log.setLevel(os.environ["JEV_ROUTER_LOG_LEVEL"])
    log.addHandler(logging.StreamHandler())


def log_event(kind: str, payload: dict[str, Any]) -> None:
    log.info("%s %s", kind, json.dumps(payload, default=str))
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps({"event": kind, **payload}, default=str) + "\n")


class DualCacheKV:
    def __init__(self, cache: Any) -> None:
        self.cache = cache

    async def get(self, key: str) -> Any | None:
        return await self.cache.async_get_cache(key)

    async def set(self, key: str, value: Any, ttl: int) -> None:
        await self.cache.async_set_cache(key, value, ttl=ttl)


def resolve_tiers(cfg: RouterConfig, model_list: Sequence[Mapping[str, Any]] | None = None) -> None:
    """Fill tier.model_id/provider from the proxy's router, or from a config.yaml model_list."""
    if model_list is None:
        try:
            from litellm.proxy.proxy_server import llm_router

            model_list = list(llm_router.get_model_list() or []) if llm_router else None
        except Exception:
            model_list = None
    by_name = {d["model_name"]: d["litellm_params"].get("model") for d in model_list or []}
    for tier in cfg.tiers:
        model_id = tier.model_id or by_name.get(tier.model) or tier.model
        tier.model_id = model_id
        if tier.provider is None:
            try:
                tier.provider = litellm.get_llm_provider(model_id)[1]
            except Exception:
                tier.provider = None


def usage_of(response: Any) -> tuple[int, int]:
    """Return (prompt_tokens, cached_tokens) from an OpenAI- or Anthropic-shaped usage."""
    u = getattr(response, "usage", None) or (response.get("usage") if isinstance(response, dict) else None)
    if u is None:
        return 0, 0
    g = (lambda k: getattr(u, k, None)) if not isinstance(u, dict) else u.get
    details = g("prompt_tokens_details")
    cached = (
        (
            getattr(details, "cached_tokens", None)
            if details is not None and not isinstance(details, dict)
            else (details or {}).get("cached_tokens")
        )
        or g("cache_read_input_tokens")
        or 0
    )
    prompt = g("prompt_tokens") or g("input_tokens") or 0
    return int(prompt), int(cached)


def response_fingerprint(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if choices:
        msg = choices[0].message
        return fingerprint_assistant(
            {"content": msg.content, "tool_calls": [c.model_dump() for c in msg.tool_calls or []]}
        )
    content = getattr(response, "content", None) or (response.get("content") if isinstance(response, dict) else None)
    if isinstance(content, list):
        ids = [b.get("id") for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
        text = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        return fingerprint_assistant({"content": text, "tool_calls": [{"id": i} for i in ids]})
    return None


def metadata_key(call_type: str) -> str:
    """Chat completions use `metadata`; /v1/messages and /v1/responses keep proxy metadata in `litellm_metadata`."""
    return "metadata" if call_type.endswith("completion") else "litellm_metadata"


def request_info(md: dict[str, Any] | None) -> dict[str, Any] | None:
    return (md or {}).get(META_KEY)


def apply_effort(data: dict[str, Any], tier: Tier, call_type: str) -> None:
    if tier.effort is None:
        return
    if call_type.endswith("anthropic_messages") and tier.provider == "anthropic":
        data["output_config"] = {**(data.get("output_config") or {}), "effort": tier.effort}
    else:
        data["reasoning_effort"] = tier.effort


class JevRouterPlugin(CustomLogger):
    def __init__(self, config_path: str | None = None) -> None:
        super().__init__()
        self.cfg = load_config(config_path or os.environ.get("JEV_ROUTER_CONFIG", "deploy/router.yaml"))
        self.router: Router | None = None

    def _ensure(self, cache: Any) -> Router:
        if self.router is None:
            resolve_tiers(self.cfg)
            judge = JevJudge(model=self.cfg.jev.model, timeout_s=self.cfg.jev.timeout_ms / 1000)
            self.router = Router(self.cfg, judge, Ledger(DualCacheKV(cache), self.cfg.cache_ttl_seconds))
        return self.router

    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: str) -> dict:  # type: ignore[override]
        if data.get("model") != self.cfg.alias:
            return data
        router = self._ensure(cache)
        start = time.time()
        try:
            decision, turn = await router.decide(data, call_type, start)
        except Exception:
            log.exception("routing failed; using default tier")
            decision, turn = None, None
        tier = self.cfg.tier(decision.tier if decision else self.cfg.default)
        if not self.cfg.shadow:
            data["model"] = tier.model
            apply_effort(data, tier, call_type)
        info = {
            "start": start,
            "call_type": call_type,
            "shadow": self.cfg.shadow,
            "stable_prefix_tokens": turn.stable_prefix_tokens if turn else 0,
            "decision": decision.to_dict() if decision else {"tier": tier.name, "reason": "error"},
        }
        data.setdefault(metadata_key(call_type), {})[META_KEY] = info
        if call_id := data.get("litellm_call_id"):
            await router.ledger.kv.set(f"jev_router:pending:{call_id}", info, 600)
        log_event("decision", {"jev_ms": round((time.time() - start) * 1000), **info})
        return data

    async def _pending(self, call_id: str | None, *metadata: dict[str, Any] | None) -> dict[str, Any] | None:
        if call_id and self.router is not None:
            info = await self.router.ledger.kv.get(f"jev_router:pending:{call_id}")
            if info:
                return info
        for md in metadata:
            if info := request_info(md):
                return info
        return None

    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        lp = kwargs.get("litellm_params") or {}
        info = await self._pending(kwargs.get("litellm_call_id"), lp.get("metadata"), lp.get("litellm_metadata"))
        if not info or info.get("shadow") or self.router is None:
            return
        from .core.request import Turn
        from .core.scorer import Decision

        d = info["decision"]
        decision = Decision(d["tier"], d["reason"], d["conversation_id"])
        turn = Turn(messages=[], stable_prefix_tokens=info.get("stable_prefix_tokens", 0))
        prompt, cached = usage_of(response_obj)
        await self.router.observe(decision, turn, cached, response_fingerprint(response_obj), info["start"])
        log_event(
            "observed",
            {
                "conversation_id": d["conversation_id"],
                "tier": d["tier"],
                "model": kwargs.get("model"),
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "cost": kwargs.get("response_cost"),
            },
        )

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: str | None = None,
    ) -> None:  # type: ignore[override]
        info = await self._pending(
            request_data.get("litellm_call_id"), request_data.get("metadata"), request_data.get("litellm_metadata")
        )
        if info and self.router is not None and not info.get("shadow"):
            await self.router.forget_cache(info["decision"]["conversation_id"])


proxy_handler_instance = JevRouterPlugin()
