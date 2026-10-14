"""
API wrapper for Arbitra.

Run locally:
    uvicorn src.api.app:app --reload --port 8000
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..core import config
from ..graph.workflow import run_case
from ..tools.interface import get_tools
from ..eval.run_eval import load_context, run_split, summarize, run_probes

app = FastAPI(title="Arbitra API", version="0.1.0")


class RunCaseRequest(BaseModel):
    config: Literal["A", "B", "C"] = "C"
    stub: bool = False


class EvalRequest(BaseModel):
    config: Literal["A", "B", "C"] = "C"
    split: str = "dev"
    source: Literal["auto", "data", "fixtures"] = "auto"
    limit: int | None = Field(default=None, ge=1)
    probes: bool = False


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "tools": os.getenv("ARBITRA_TOOLS", "mock"),
        "stub": config.is_stub(),
    }


@app.get("/cases")
def list_cases() -> dict:
    tools = get_tools()
    return {
        "source": type(tools).__name__,
        "count": len(tools.dispute_ids()),
        "cases": tools.dispute_ids(),
    }


@app.get("/cases/{dispute_id}/truth")
def case_truth(dispute_id: str) -> dict:
    tools = get_tools()
    gt = getattr(tools, "ground_truth", None)
    if not callable(gt):
        raise HTTPException(status_code=404, detail="Ground truth not available for this tool source")
    record = gt(dispute_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No truth record for {dispute_id}")
    return {"dispute_id": dispute_id, "truth": record}


@app.post("/cases/{dispute_id}/run")
def run_one(dispute_id: str, req: RunCaseRequest) -> dict:
    tools = get_tools()
    if dispute_id not in tools.dispute_ids():
        raise HTTPException(status_code=404, detail=f"Unknown dispute_id: {dispute_id}")

    prev = config.is_stub()
    try:
        config.set_stub(req.stub)
        state = run_case(dispute_id, req.config)
    finally:
        config.set_stub(prev)

    verdict = state.get("verdict")
    return {
        "dispute_id": dispute_id,
        "config": req.config,
        "escalated": bool(state.get("escalated")),
        "errors": state.get("errors", []),
        "trace": state.get("trace", []),
        "verdict": verdict.model_dump() if verdict else None,
        "motion_outcome": state.get("motion_outcome"),
        "settlement_accepted": state.get("settlement_accepted", False),
    }


@app.post("/eval")
def run_eval_api(req: EvalRequest) -> dict:
    source = req.source
    if source == "auto":
        source = "fixtures" if not os.path.isdir(os.path.join(config.DATA_DIR, "cases")) else "data"

    tools, truth, splits, pairs, pair_roles, label = load_context(source)
    ids = splits.get(req.split) or splits.get("all") or sorted(truth)
    if not ids:
        raise HTTPException(status_code=400, detail=f"Split '{req.split}' is empty")

    results = run_split(ids, req.config, truth, tools, req.limit, quiet=True)
    summary = summarize(results, req.config, req.split)
    if req.probes and pairs:
        summary.probes = run_probes(tools, pairs, pair_roles, req.config, truth, quiet=True)

    return {
        "source": label,
        "n": len(results),
        "summary": summary.as_dict(),
    }
