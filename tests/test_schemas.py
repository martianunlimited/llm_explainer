"""Unit tests for Pydantic domain contracts and application settings."""

import pytest
from pydantic import ValidationError

from src.config import AppSettings
from src.schemas import LogitCandidate, SamplingConfig


def test_sampling_config_defaults_and_bounds() -> None:
    """Validate default bounds and invalid range rejection on SamplingConfig."""
    cfg = SamplingConfig()
    assert cfg.temperature == 0.8
    assert cfg.top_k == 40
    assert cfg.top_p == 0.9

    with pytest.raises(ValidationError):
        SamplingConfig(temperature=-0.1)

    with pytest.raises(ValidationError):
        SamplingConfig(top_p=0.0)

    with pytest.raises(ValidationError):
        SamplingConfig(repetition_penalty=0.5)


def test_logit_candidate_fields() -> None:
    """Ensure LogitCandidate preserves token metadata cleanly."""
    cand = LogitCandidate(
        token=" Paris",
        display_token="·Paris",
        token_id=6342,
        logit=13.81,
        prob=0.612,
    )
    assert cand.token_id == 6342
    assert cand.prob == pytest.approx(0.612)


def test_settings_default_presets_reference_companion_demos() -> None:
    """Verify configured presets reference the user's companion labs."""
    cfg = AppSettings()
    urls = {p.companion_url for p in cfg.preset_specs}
    assert any("llm_tokenizer" in u for u in urls)
    assert any("llm_embedding" in u for u in urls)
    assert any("llm_attention" in u for u in urls)
    assert any("llm_sampling" in u for u in urls)
