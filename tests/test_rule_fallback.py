import os

# Force fixture source and no API key to test deterministic fallback path.
os.environ["ARBITRA_TOOLS"] = "mock"
os.environ.pop("OPENAI_API_KEY", None)

from src.core import config  # noqa: E402
from src.graph.workflow import run_case  # noqa: E402


def test_non_stub_run_works_without_model_key():
    config.set_stub(False)
    state = run_case("DSP-0001", "C")
    verdict = state.get("verdict")
    assert verdict is not None
    assert verdict.cited_policy_id is not None
    assert verdict.findings_of_fact
    assert "placeholder" not in verdict.remedy_rationale.lower()


def test_advocate_briefs_are_cited_in_fallback_mode():
    config.set_stub(False)
    state = run_case("DSP-0002", "C")
    for side in ("rider", "driver"):
        brief = state.get(f"{side}_brief")
        assert brief is not None
        assert brief.claims
        for claim in brief.claims:
            assert claim.evidence_refs or claim.policy_refs
