from types import SimpleNamespace

from jev_router.core.config import Tier
from jev_router.litellm_plugin import apply_effort, response_fingerprint, usage_of


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
