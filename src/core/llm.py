"""LLM factory. Stub-aware so the pipeline runs with no API key on day 1."""

from __future__ import annotations

from typing import Any

from . import config


class StubLLM:
    """Returns a fixed payload. Used only by --stub / ARBITRA_STUB."""

    def __init__(self, payload: Any):
        self.payload = payload

    def with_structured_output(self, schema):  # noqa: ANN001, ARG002
        return self

    def invoke(self, _messages):  # noqa: ANN001
        return self.payload

    def bind(self, **_kwargs):  # noqa: ANN003
        return self


def get_llm(tier: str = "advocate", stub_payload: Any = None):  # noqa: ANN001
    """
    tier: "advocate" | "judge" | "officer"

    Advocates get the cheap model (they run in parallel so latency doesn't stack);
    the Judge alone gets the strong one.
    """
    if config.is_stub():
        if stub_payload is None:
            raise ValueError("stub mode requires a stub_payload")
        return StubLLM(stub_payload)

    from langchain_openai import ChatOpenAI  # imported lazily: no key, no crash on import

    return ChatOpenAI(
        model=config.MODELS[tier],
        temperature=config.TEMPERATURE,
        timeout=60,
        max_retries=2,
    )
