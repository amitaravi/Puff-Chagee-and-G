"""
The contract between Person A (data) and Person B (agents).

Person B builds against MockTools (fixture-backed) from day 1 and never waits.
Person A ships src/tools/real_tools.py with these four functions; flipping
ARBITRA_TOOLS=real swaps them in. Nothing in agents/ changes.
"""

from __future__ import annotations

import os
from typing import Protocol

from ..core.state import EvidencePack, FraudSignals, Policy, Precedent


class ToolProvider(Protocol):
    def get_evidence_pack(self, dispute_id: str) -> EvidencePack:
        """Full case file. Includes judge-only fields — redact before serving advocates."""
        ...

    def get_policy(self, policy_id: str) -> Policy: ...

    def search_precedents(self, facts: dict, k: int = 3) -> list[Precedent]:
        """RAG over the precedent knowledge base. Judge-only."""
        ...

    def get_fraud_signals(self, dispute_id: str) -> FraudSignals:
        """Judge-only. Never served to advocates."""
        ...


def get_tools() -> ToolProvider:
    if os.getenv("ARBITRA_TOOLS", "mock").lower() == "real":
        from .real_tools import RealTools  # noqa: PLC0415  (Person A owns this file)

        return RealTools()
    from .mock_tools import MockTools  # noqa: PLC0415

    return MockTools()
