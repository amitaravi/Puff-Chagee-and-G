"""
Reference implementation of the Person A tool layer, over the layout in DATA_CONTRACT.md.

    data/
      policies.json      precedents.json
      cases/DSP-*.json   index.json
      ground_truth.json  <-- deliberately never read here

Person A is free to rewrite this against his own raw tables (riders.json, trips.json,
gps_telemetry.jsonl, whatever). KEEP THE FOUR SIGNATURES — nothing in agents/ reads
anything else.

Swap in with:  ARBITRA_TOOLS=real
"""

from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
from typing import Any

from ..core.config import DATA_DIR
from ..core.state import (
    EvidencePack,
    FraudSignals,
    Policy,
    Precedent,
)

# Fields used to rank precedents by factual similarity. Everything else is noise for
# retrieval purposes — stray metadata makes near-neighbour matching worse, not better.
SIMILARITY_FIELDS = (
    "deviation_pct",
    "traffic_delay_min",
    "duration_delta_min",
    "driver_min_distance_to_pickup_m",
    "wait_time_min",
    "surge_multiplier",
)


def _read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class RealTools:
    def __init__(self, data_dir: str | None = None):
        self.dir = data_dir or DATA_DIR
        self._cases: dict[str, dict] = {}
        self._policies: dict[str, Policy] = {}
        self._precedents: list[Precedent] = []
        self._index: dict = {}
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        case_dir = os.path.join(self.dir, "cases")
        if os.path.isdir(case_dir):
            for fname in sorted(os.listdir(case_dir)):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(case_dir, fname), encoding="utf-8") as fh:
                    data = json.load(fh)
                self._cases[data["dispute"]["dispute_id"]] = data
        elif os.path.isdir(self.dir):  # flat fallback: cases/*.json replaced by data/*.json
            for fname in sorted(os.listdir(self.dir)):
                if fname.startswith("_") or not fname.endswith(".json"):
                    continue
                with open(os.path.join(self.dir, fname), encoding="utf-8") as fh:
                    data = json.load(fh)
                if "dispute" not in data:
                    continue
                self._cases[data["dispute"]["dispute_id"]] = data

        for pol in _read_json(os.path.join(self.dir, "policies.json"), []):
            self._policies[pol["policy_id"]] = Policy(**pol)

        for pre in _read_json(os.path.join(self.dir, "precedents.json"), []):
            self._precedents.append(Precedent(**pre))

        self._index = _read_json(os.path.join(self.dir, "index.json"), {})

    # ------------------------------------------------------------------ helpers
    def dispute_ids(self) -> list[str]:
        return sorted(self._cases)

    def ids_for_split(self, split: str = "all") -> list[str]:
        splits = self._index.get("splits")
        if not splits:
            return self.dispute_ids()
        if split == "all":
            return sorted({d for ids in splits.values() for d in ids})
        return sorted(splits.get(split, []))

    def pairs(self, role: str | None = None) -> dict[str, list[str]]:
        """pair_id -> [dispute_id, dispute_id, ...] for twin / threshold / bias pairs."""
        out: dict[str, list[str]] = {}
        for pid, ids in self._index.get("pairs", {}).items():
            if role and self._index.get("pair_roles", {}).get(pid) != role:
                continue
            out[pid] = ids
        return out

    def ground_truth(self, dispute_id: str | None = None):
        """
        EVAL ONLY. Never call this from src/agents/ or src/tools/ — tests/test_leakage.py
        fails the build if you do.
        """
        gt = _read_json(os.path.join(self.dir, "ground_truth.json"), {})
        return gt.get(dispute_id) if dispute_id else gt

    # ------------------------------------------------------------------ contract
    def get_evidence_pack(self, dispute_id: str) -> EvidencePack:
        if dispute_id not in self._cases:
            raise KeyError(
                f"no case for {dispute_id} under {self.dir}; have {len(self._cases)} case(s). "
                f"Run: python scripts/generate_dataset.py"
            )
        raw = dict(self._cases[dispute_id])
        for key in ("ground_truth", "_ground_truth", "correct_ruling", "should_escalate"):
            raw.pop(key, None)
        return EvidencePack(**raw)

    def get_policy(self, policy_id: str) -> Policy:
        if policy_id in self._policies:
            return self._policies[policy_id]
        for case in self._cases.values():  # policies may also be inlined on the case
            for pol in case.get("policies", []):
                if pol["policy_id"] == policy_id:
                    return Policy(**pol)
        raise KeyError(f"unknown policy {policy_id}")

    def search_precedents(self, facts: dict, k: int = 3) -> list[Precedent]:
        """
        Retrieval over the precedent knowledge base. Judge-only.

        Category-filtered, then ranked by similarity on the numeric key facts. Cheap
        and deterministic — a real build would embed `fact_summary`, but for the MVP a
        deterministic retriever is better anyway: it makes the precedent contribution
        measurable instead of stochastic.

        Deliberately does NOT filter for agreement with any particular outcome. Two
        precedents with the same facts and opposite rulings is the honest state of a
        real precedent base, and it is what makes the conflict penalty fire.
        """
        category = facts.get("category")
        pool = [p for p in self._precedents if not category or p.category == category]
        if not pool:
            pool = list(self._precedents)

        def score(p: Precedent) -> float:
            if not facts:
                return 0.0
            total, hits = 0, 0.0
            for field in SIMILARITY_FIELDS:
                a = facts.get(field)
                b = p.key_facts.get(field)
                if a is None or b is None:
                    continue
                total += 1
                try:
                    a, b = float(a), float(b)
                except (TypeError, ValueError):
                    hits += float(SequenceMatcher(None, str(a), str(b)).ratio())
                    continue
                span = max(abs(a), abs(b), 1e-6)
                hits += max(0.0, 1.0 - abs(a - b) / span)
            return hits / total if total else 0.0

        ranked = sorted(pool, key=lambda p: (-score(p), p.precedent_id))
        return ranked[:k]

    def get_fraud_signals(self, dispute_id: str) -> FraudSignals:
        raw = self._cases.get(dispute_id, {}).get("fraud")
        if not raw:
            return FraudSignals(dispute_id=dispute_id)
        return FraudSignals(**raw)
