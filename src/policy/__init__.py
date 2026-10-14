"""Deterministic policy engine — the answer key. Never an LLM."""

from .apply_policy import Truth, apply_policy, band_for, truth_for_pack

__all__ = ["Truth", "apply_policy", "band_for", "truth_for_pack"]
