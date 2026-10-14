"""
The leakage test. Ten lines, and it is the first thing a sharp judge will ask about.

If any file the agents can read contains ground truth, the entire eval is worthless
and you would never know.
"""

import json
import os

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
FORBIDDEN = {
    "correct_ruling",
    "correct_refund_amount",
    "correct_policy_id",
    "expected_confidence_band",
    "should_escalate",
}


def test_no_label_leakage_in_fixtures():
    for fname in sorted(os.listdir(FIXTURES)):
        if fname.startswith("_"):
            continue  # _ground_truth.json is deliberately excluded from the tool layer
        with open(os.path.join(FIXTURES, fname), encoding="utf-8") as fh:
            blob = fh.read()
        for key in FORBIDDEN:
            assert key not in blob, f"leak: '{key}' found in {fname}"


def test_ground_truth_file_is_underscore_prefixed():
    """MockTools skips underscore-prefixed files. This asserts that contract holds."""
    assert os.path.exists(os.path.join(FIXTURES, "_ground_truth.json"))
    with open(os.path.join(FIXTURES, "_ground_truth.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    assert "DSP-0001" in data


def test_mock_tools_never_exposes_labels():
    from src.tools.mock_tools import MockTools

    tools = MockTools()
    for dispute_id in tools.dispute_ids():
        pack = tools.get_evidence_pack(dispute_id)
        for key in FORBIDDEN:
            assert key not in pack.model_dump()


# ---------------------------------------------------------------------------
# The same guarantee, for the generated dataset. Skipped until data/ exists.
# ---------------------------------------------------------------------------

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cases")
DATASET_FORBIDDEN = FORBIDDEN | {"ground_truth", "confidence_target"}


@pytest.mark.skipif(not os.path.isdir(DATA), reason="data/ not generated yet")
def test_no_label_leakage_in_dataset():
    for fname in sorted(os.listdir(DATA)):
        with open(os.path.join(DATA, fname), encoding="utf-8") as fh:
            blob = fh.read()
        for key in DATASET_FORBIDDEN:
            assert f'"{key}"' not in blob, f"leak: '{key}' found in data/cases/{fname}"


@pytest.mark.skipif(not os.path.isdir(DATA), reason="data/ not generated yet")
def test_dataset_labels_match_the_policy_engine():
    """Ground truth is DERIVED from apply_policy(). If they drift, the eval lies."""
    import json

    from src.core.state import EvidencePack
    from src.policy.apply_policy import apply_policy

    with open(os.path.join(os.path.dirname(DATA), "ground_truth.json"), encoding="utf-8") as fh:
        gt = json.load(fh)

    checked = 0
    for fname in sorted(os.listdir(DATA))[:60]:
        with open(os.path.join(DATA, fname), encoding="utf-8") as fh:
            pack = EvidencePack(**json.load(fh))
        truth = apply_policy(pack)
        stored = gt[pack.dispute.dispute_id]
        assert stored["correct_ruling"] == truth.ruling, pack.dispute.dispute_id
        assert stored["should_escalate"] == truth.should_escalate, pack.dispute.dispute_id
        checked += 1
    assert checked > 0


def test_grounding_check_catches_a_hallucinated_reference():
    """The label-free hallucination check actually fires. Verify it, don't assume it."""
    from src.eval.metrics import resolve_ref
    from src.tools.mock_tools import MockTools

    pack = MockTools().get_evidence_pack("DSP-0001")
    assert resolve_ref(pack, "TRP-00431.route_analysis.deviation_pct")[0] is True
    assert resolve_ref(pack, "TRP-00431.fare.distance_fare")[0] is True
    assert resolve_ref(pack, "TRP-00431.route_analysis.fuel_consumption_litres")[0] is False
    assert resolve_ref(pack, "TRP-99999.route.deviation_pct")[0] is False


def test_policy_engine_is_deterministic():
    from src.policy.apply_policy import apply_policy
    from src.tools.mock_tools import MockTools

    pack = MockTools().get_evidence_pack("DSP-0001")
    assert apply_policy(pack).as_dict() == apply_policy(pack).as_dict()
