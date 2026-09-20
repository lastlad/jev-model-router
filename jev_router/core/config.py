from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Objective(BaseModel):
    lambda_cost: float = 1.0
    lambda_quality: float = 3.0


class JevConfig(BaseModel):
    model: str = "jev-latest"
    timeout_ms: int = 1500
    min_confidence: float = 0.55
    complaint_threshold: float = 0.6


class StateConfig(BaseModel):
    budget_tokens: int = 8000
    current_message_chars: int = 6000
    recent_messages: int = 6
    recent_user_chars: int = 1500
    recent_assistant_chars: int = 800
    tool_result_chars: int = 300
    older_stubs: int = 20
    older_stub_chars: int = 120
    first_message_stub_chars: int = 400
    system_prompt_chars: int = 600
    routing_history_entries: int = 5
    filter: str | None = None


class SwitchCost(BaseModel):
    base: float = 0.002
    continuity: float = 0.02
    thinking_loss: float = 0.005
    cross_provider: float = 0.01


class Tier(BaseModel):
    name: str
    model: str  # LiteLLM deployment model_name
    effort: str | None = None
    level: int = Field(ge=0, le=3)  # capability level on Jev's required_tier scale
    model_id: str | None = None  # provider-qualified id for pricing; resolved from the proxy
    provider: str | None = None


class RouterConfig(BaseModel):
    alias: str = "jev-auto"
    shadow: bool = False
    objective: Objective = Objective()
    jev: JevConfig = JevConfig()
    state: StateConfig = StateConfig()
    switch_cost: SwitchCost = SwitchCost()
    tier_penalty: list[float] = [0.0, 0.05, 0.2, 0.6]
    expected_output_tokens: list[int] = [200, 800, 3000]
    effort_output_multiplier: dict[str, float] = {
        "low": 0.7,
        "medium": 1.0,
        "high": 1.5,
        "xhigh": 2.5,
        "max": 4.0,
    }
    cache_ttl_seconds: int = 300
    tiers: list[Tier]
    default: str

    def tier(self, name: str) -> Tier:
        for t in self.tiers:
            if t.name == name:
                return t
        raise KeyError(name)

    def rank(self, name: str) -> int:
        return [t.name for t in self.tiers].index(name)


def load_config(path: str | Path) -> RouterConfig:
    with open(path) as f:
        return RouterConfig.model_validate(yaml.safe_load(f))


def load_configs(path: str | Path) -> list[RouterConfig]:
    """One router per file. `path` is a router.yaml, or a directory of them (*.yaml, sorted by name).

    Each router must have a distinct `alias`; the proxy dispatches on the request's model name.
    """
    path = Path(path)
    files = sorted(p for p in path.iterdir() if p.suffix in (".yaml", ".yml")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"no router configs found in {path}")
    cfgs = [load_config(f) for f in files]
    seen: dict[str, Path] = {}
    for cfg, f in zip(cfgs, files, strict=True):
        if cfg.alias in seen:
            raise ValueError(f"router alias {cfg.alias!r} is defined in both {seen[cfg.alias]} and {f}")
        seen[cfg.alias] = f
    return cfgs
