"""TDD suite for softmax, sampling filters, entropy, and decoding math."""

import pytest

from src.core.sampling_math import apply_sampling_pipeline, compute_entropy_bits, stable_softmax
from src.schemas import LogitCandidate, SamplingConfig


def _sample_candidates() -> list[LogitCandidate]:
    return [
        LogitCandidate(token=" Paris", display_token="·Paris", token_id=10, logit=10.0, prob=0.8),
        LogitCandidate(token=" Lyon", display_token="·Lyon", token_id=11, logit=8.0, prob=0.11),
        LogitCandidate(token=" Nice", display_token="·Nice", token_id=12, logit=6.0, prob=0.05),
        LogitCandidate(
            token=" Marseille", display_token="·Marseille", token_id=13, logit=4.0, prob=0.04
        ),
    ]


def test_stable_softmax_sums_to_one_and_handles_large_logits() -> None:
    """Softmax must remain overflow-safe and normalized to 1.0."""
    probs = stable_softmax([1000.0, 999.0, 998.0])
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)
    assert probs[0] > probs[1] > probs[2]


def test_greedy_decoding_when_temperature_zero() -> None:
    """Temperature 0.0 must assign prob 1.0 to the highest logit and 0 to others."""
    cands = _sample_candidates()
    result = apply_sampling_pipeline(
        cands,
        SamplingConfig(temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0),
    )
    assert result.filtered_candidates[0].token_id == 10
    assert result.filtered_candidates[0].prob == pytest.approx(1.0)
    assert all(c.prob == 0.0 for c in result.filtered_candidates[1:])
    assert result.entropy_bits == pytest.approx(0.0, abs=1e-9)


def test_top_k_filtering_limits_surviving_tokens() -> None:
    """Top-K = 2 must keep exactly two candidates summing to 1.0."""
    cands = _sample_candidates()
    result = apply_sampling_pipeline(
        cands,
        SamplingConfig(temperature=1.0, top_k=2, top_p=1.0, repetition_penalty=1.0),
    )
    kept = [c for c in result.filtered_candidates if c.prob > 0.0]
    assert len(kept) == 2
    assert {c.token_id for c in kept} == {10, 11}
    assert sum(c.prob for c in kept) == pytest.approx(1.0)


def test_top_p_nucleus_filtering_cuts_tail() -> None:
    """Low Top-P preserves top token plus minimal nucleus crossing threshold."""
    cands = _sample_candidates()
    result = apply_sampling_pipeline(
        cands,
        SamplingConfig(temperature=1.0, top_k=0, top_p=0.5, repetition_penalty=1.0),
    )
    kept = [c for c in result.filtered_candidates if c.prob > 0.0]
    assert len(kept) == 1
    assert kept[0].token_id == 10


def test_repetition_penalty_reorders_seen_token() -> None:
    """Repetition penalty penalizes already-seen token IDs according to standard rule."""
    cands = _sample_candidates()
    result = apply_sampling_pipeline(
        cands,
        SamplingConfig(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.6),
        seen_token_ids={10},
    )
    # 10.0 / 1.6 = 6.25 < 8.0 (Lyon), so token_id 11 overtakes token_id 10.
    assert result.filtered_candidates[0].token_id == 11


def test_entropy_non_negative_and_zero_for_point_mass() -> None:
    """Shannon entropy in bits is 0 for deterministic distribution and >0 for uniform."""
    assert compute_entropy_bits([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert compute_entropy_bits([0.25, 0.25, 0.25, 0.25]) == pytest.approx(2.0, abs=1e-6)
