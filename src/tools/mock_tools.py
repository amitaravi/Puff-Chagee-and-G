"""
Fixture-backed tool provider owned by Person B.

Loads the three hand-written golden fixtures in tests/fixtures/ and serves them
through the same interface Person A's real layer will implement. Ground truth
(_ground_truth.json) is deliberately NOT loadable here — see tests/test_leakage.py.
"""

from __future__ import annotations

import json
import os

from ..core.config import FIXTURES_DIR
from ..core.state import EvidencePack, FraudSignals, Policy, Precedent


class MockTools:
    def __init__(self, fixtures_dir: str | None = None):
        self.dir = fixtures_dir or FIXTURES_DIR
        self._cases: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.isdir(self.dir):
            return
        for fname in sorted(os.listdir(self.dir)):
            if fname.startswith("_"):  # _ground_truth.json — never loaded by tools
                continue
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(self.dir, fname), encoding="utf-8") as fh:
                data = json.load(fh)
            self._cases[data["dispute"]["dispute_id"]] = data

    def dispute_ids(self) -> list[str]:
        return sorted(self._cases)

    def get_evidence_pack(self, dispute_id: str) -> EvidencePack:
        if dispute_id not in self._cases:
            raise KeyError(f"no fixture for {dispute_id}; have {self.dispute_ids()}")
        raw = dict(self._cases[dispute_id])
        # top-level keys that are not part of EvidencePack
        for key in ("ground_truth", "_ground_truth"):
            raw.pop(key, None)
        return EvidencePack(**raw)

    def get_policy(self, policy_id: str) -> Policy:
        for case in self._cases.values():
            for pol in case.get("policies", []):
                if pol["policy_id"] == policy_id:
                    return Policy(**pol)
        raise KeyError(policy_id)

    def search_precedents(self, facts: dict, k: int = 3) -> list[Precedent]:
        category = facts.get("category")
        found: list[Precedent] = []
        for case in self._cases.values():
            for pre in case.get("precedents", []):
                if category and pre.get("category") != category:
                    continue
                found.append(Precedent(**pre))
        return found[:k]

    def get_fraud_signals(self, dispute_id: str) -> FraudSignals:
        raw = self._cases[dispute_id].get("fraud")
        if not raw:
            return FraudSignals(dispute_id=dispute_id)
        return FraudSignals(**raw)
