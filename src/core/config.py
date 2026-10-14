"""Model tiering, ablation configs, thresholds. Single source of truth."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------- model tiering
# Cheap models for the two advocates (they run in parallel, so latency doesn't stack),
# the strongest model for the Judge alone. ~60% cheaper than tiering everything at the top.
MODELS: dict[str, str] = {
    "advocate": os.getenv("ARBITRA_MODEL_ADVOCATE", "gpt-4o-mini"),
    "judge": os.getenv("ARBITRA_MODEL_JUDGE", "gpt-4o"),
    "officer": os.getenv("ARBITRA_MODEL_OFFICER", "gpt-4o-mini"),
}

TEMPERATURE = float(os.getenv("ARBITRA_TEMPERATURE", "0.0"))

# ---------------------------------------------------------------- thresholds
CONFIDENCE_THRESHOLD = float(os.getenv("ARBITRA_CONFIDENCE_THRESHOLD", "0.75"))
CRITICAL_PRIORITY_ESCALATES = True  # safety-related disputes go to a human regardless

# ---------------------------------------------------------------- ablation
@dataclass
class Flags:
    """What this configuration switches on. The eval harness reports per-config."""

    use_advocates: bool = True
    use_rebuttal: bool = True
    use_motions: bool = True
    use_settlement: bool = True
    use_precedent: bool = True
    use_fraud: bool = True

    def as_dict(self) -> dict[str, bool]:
        return {
            "use_advocates": self.use_advocates,
            "use_rebuttal": self.use_rebuttal,
            "use_motions": self.use_motions,
            "use_settlement": self.use_settlement,
            "use_precedent": self.use_precedent,
            "use_fraud": self.use_fraud,
        }


CONFIGS: dict[str, Flags] = {
    # A — one agent does everything. The baseline we must beat.
    "A": Flags(
        use_advocates=False,
        use_rebuttal=False,
        use_motions=False,
        use_settlement=False,
        use_precedent=False,
        use_fraud=False,
    ),
    # B — three agents, no debate, no fast paths.
    "B": Flags(
        use_advocates=True,
        use_rebuttal=False,
        use_motions=False,
        use_settlement=False,
        use_precedent=False,
        use_fraud=False,
    ),
    # C — full tribunal.
    "C": Flags(),
}


def flags_for(config: str) -> Flags:
    return CONFIGS.get(config.upper(), CONFIGS["C"])


# ---------------------------------------------------------------- stub mode
# --stub runs the whole pipeline with canned verdicts, no API key needed.
# Use it on day 1 to prove the graph compiles and state flows before writing real prompts.
_STUB = False


def set_stub(value: bool) -> None:
    global _STUB
    _STUB = value


def is_stub() -> bool:
    return _STUB or os.getenv("ARBITRA_STUB", "").lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES_DIR = os.path.join(ROOT, "tests", "fixtures")
DATA_DIR = os.getenv("ARBITRA_DATA_DIR", os.path.join(ROOT, "data"))
RESULTS_DIR = os.path.join(ROOT, "results")


@dataclass
class Cost:
    """Accumulated token spend for the run. Feeds the cost-per-case metric."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    by_tier: dict[str, int] = field(default_factory=dict)

    def add(self, tier: str, usage: dict[str, int] | None) -> None:
        self.calls += 1
        self.by_tier[tier] = self.by_tier.get(tier, 0) + 1
        if not usage:
            return
        self.prompt_tokens += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
