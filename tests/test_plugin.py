from types import SimpleNamespace

import pytest
from conftest import TIERS, judgment

from jev_router.core.config import RouterConfig, Tier, load_configs
from jev_router.core.judge import RecordedJudge
from jev_router.core.ledger import MemoryKV
from jev_router.litellm_plugin import (
    JevRouterPlugin,
    apply_effort,
    metadata_key,
    response_fingerprint,
    usage_of,
)


def test_usage_openai_shape():
    r = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            prompt_tokens_details=SimpleNamespace(cached_tokens=900),
            cache_creation_input_tokens=0,
        )
    )
    assert usage_of(r) == (1000, 900)


def test_usage_anthropic_shape():
    assert usage_of({"usage": {"input_tokens": 100, "cache_read_input_tokens": 80}}) == (100, 80)


def test_effort_placement():
    t = Tier(name="sonnet-low", model="sonnet", effort="low", level=1, provider="anthropic")
    d = {}
    apply_effort(d, t, "aanthropic_messages")
    assert d["output_config"] == {"effort": "low"}
    d = {}
    apply_effort(d, t, "completion")
    assert d["reasoning_effort"] == "low"


def test_response_fingerprint_shapes():
    assert response_fingerprint({"content": [{"type": "tool_use", "id": "toolu_1"}]}) == "tc:toolu_1"
    msg = SimpleNamespace(content="hello", tool_calls=None)
    assert response_fingerprint(SimpleNamespace(choices=[SimpleNamespace(message=msg)])).startswith("tx:")


def test_metadata_key_by_route():
    assert metadata_key("acompletion") == "metadata"
    assert metadata_key("aanthropic_messages") == "litellm_metadata"
    assert metadata_key("aresponses") == "litellm_metadata"


def _router_yaml(alias: str, tiers: list[dict], default: str) -> str:
    cfg = RouterConfig(tiers=tiers, default=default, alias=alias)
    import yaml

    return yaml.safe_dump(cfg.model_dump(), sort_keys=False)


def test_load_configs_reads_a_directory_and_rejects_duplicate_aliases(tmp_path):
    (tmp_path / "a.yaml").write_text(_router_yaml("jev-a", TIERS[:2], "luna"))
    (tmp_path / "b.yaml").write_text(_router_yaml("jev-b", TIERS[2:4], "sonnet-low"))
    (tmp_path / "notes.md").write_text("ignored")
    cfgs = load_configs(tmp_path)
    assert [c.alias for c in cfgs] == ["jev-a", "jev-b"]
    assert [c.alias for c in load_configs(tmp_path / "a.yaml")] == ["jev-a"]
    (tmp_path / "c.yaml").write_text(_router_yaml("jev-a", TIERS[:2], "luna"))
    with pytest.raises(ValueError, match="jev-a"):
        load_configs(tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no router configs"):
        load_configs(tmp_path / "empty")


class _Cache:
    """The DualCache surface the plugin uses, over a dict."""

    def __init__(self) -> None:
        self.kv = MemoryKV()

    async def async_get_cache(self, key):
        return await self.kv.get(key)

    async def async_set_cache(self, key, value, ttl=None):
        await self.kv.set(key, value, ttl or 0)


async def test_plugin_dispatches_on_alias_and_keeps_ledgers_apart(tmp_path, monkeypatch):
    (tmp_path / "a.yaml").write_text(_router_yaml("jev-a", TIERS[:2], "luna"))
    (tmp_path / "b.yaml").write_text(_router_yaml("jev-b", TIERS[2:5], "sonnet-low"))
    plugin = JevRouterPlugin(str(tmp_path))
    monkeypatch.setattr("jev_router.litellm_plugin.JevJudge", lambda **kw: RecordedJudge(judgment(level=1)))
    cache = _Cache()
    msgs = [{"role": "user", "content": "Refactor the parser to support nested lists."}]

    untouched = await plugin.async_pre_call_hook({}, cache, {"model": "gpt-4o", "messages": msgs}, "acompletion")
    assert untouched["model"] == "gpt-4o" and "metadata" not in untouched

    a = await plugin.async_pre_call_hook(
        {},
        cache,
        {"model": "jev-a", "messages": msgs, "litellm_session_id": "s", "litellm_call_id": "c1"},
        "acompletion",
    )
    assert a["model"] in ("haiku", "luna") and a["metadata"]["jev_router"]["alias"] == "jev-a"
    await plugin.async_log_success_event(
        {"litellm_call_id": "c1", "model": a["model"], "litellm_params": {"metadata": a["metadata"]}},
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, prompt_tokens_details=None), choices=[]),
        None,
        None,
    )
    b = await plugin.async_pre_call_hook(
        {},
        cache,
        {"model": "jev-b", "messages": msgs, "litellm_session_id": "s", "litellm_call_id": "c2"},
        "acompletion",
    )
    info_a, info_b = a["metadata"]["jev_router"], b["metadata"]["jev_router"]
    assert b["model"].startswith("sonnet") and info_b["alias"] == "jev-b"
    assert info_a["decision"]["conversation_id"] == info_b["decision"]["conversation_id"]
    assert info_b["decision"]["reason"] == "fresh", "router b must not inherit router a's incumbent"
    assert set(plugin.routers) == {"jev-a", "jev-b"}
