"""LLM factory. Stub-aware so the pipeline runs with no API key on day 1."""

from __future__ import annotations

import os
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


def _api_key() -> str | None:
    """Provider key resolution: OpenAI first, then Groq."""
    return os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")


def _base_url() -> str | None:
    """OpenAI-compatible base URL override (works for Groq and proxies)."""
    explicit = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if explicit:
        return explicit
    if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        return "https://api.groq.com/openai/v1"
    return None


def model_ready() -> bool:
    """True when a real model can be called in this environment."""
    return bool(_api_key())


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

    if not model_ready():
        raise RuntimeError("No API key found. Set OPENAI_API_KEY or GROQ_API_KEY, or use --stub.")

    from langchain_openai import ChatOpenAI  # imported lazily: no key, no crash on import

    kwargs = {
        "model": config.MODELS[tier],
        "temperature": config.TEMPERATURE,
        "timeout": 60,
        "max_retries": 2,
        "api_key": _api_key(),
    }
    base = _base_url()
    if base:
        kwargs["base_url"] = base

    # Groq's on-demand OTPM limits can reject large expected outputs. Keep generated
    # outputs bounded unless explicitly overridden.
    if os.getenv("GROQ_API_KEY") or (base and "groq.com" in base):
        kwargs["max_tokens"] = int(os.getenv("ARBITRA_MAX_TOKENS", "400"))

    return ChatOpenAI(**kwargs)
