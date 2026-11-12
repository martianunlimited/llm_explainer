"""Pure mathematical utilities for LLM logit transformation and decoding filters."""

import math

from pydantic import BaseModel

from src.schemas import LogitCandidate, SamplingConfig


class SamplingResult(BaseModel):
    """Result of applying temperature, penalty, top-k, top-p, and min-p decoding."""

    filtered_candidates: list[LogitCandidate]
    entropy_bits: float
    surviving_count: int
    clipped_mass: float


def stable_softmax(logits: list[float]) -> list[float]:
    """Compute overflow-safe softmax probabilities over a vector of logits."""
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(val - max_logit) for val in logits]
    total = sum(exps)
    if total <= 0.0:
        uniform = 1.0 / len(logits)
        return [uniform for _ in logits]
    return [val / total for val in exps]


def compute_entropy_bits(probs: list[float]) -> float:
    """Compute Shannon entropy H = -sum(p * log2(p)) in bits."""
    entropy = 0.0
    for prob in probs:
        if prob > 1e-12:
            entropy -= prob * math.log2(prob)
    return max(0.0, entropy)


def _apply_repetition_penalty(logit: float, penalty: float) -> float:
    """Adjust logit for already-generated tokens following HuggingFace sign convention."""
    if penalty == 1.0:
        return logit
    return logit / penalty if logit > 0.0 else logit * penalty


def apply_sampling_pipeline(
    candidates: list[LogitCandidate],
    config: SamplingConfig,
    seen_token_ids: set[int] | None = None,
) -> SamplingResult:
    """Apply repetition penalty, temperature, Top-K, Top-P, and Min-P filtering."""
    if not candidates:
        return SamplingResult(
            filtered_candidates=[],
            entropy_bits=0.0,
            surviving_count=0,
            clipped_mass=0.0,
        )

    seen = seen_token_ids or set()
    penalized: list[tuple[LogitCandidate, float]] = [
        (
            cand,
            _apply_repetition_penalty(cand.logit, config.repetition_penalty)
            if cand.token_id in seen
            else cand.logit,
        )
        for cand in candidates
    ]
    penalized.sort(key=lambda item: item[1], reverse=True)

    # Greedy fast path for temperature == 0.0
    if config.temperature <= 1e-6:
        top_cand, top_logit = penalized[0]
        greedy_list = [
            LogitCandidate(
                token=cand.token,
                display_token=cand.display_token,
                token_id=cand.token_id,
                logit=adj_logit,
                prob=1.0 if idx == 0 else 0.0,
            )
            for idx, (cand, adj_logit) in enumerate(penalized)
        ]
        return SamplingResult(
            filtered_candidates=greedy_list,
            entropy_bits=0.0,
            surviving_count=1,
            clipped_mass=1.0 - top_cand.prob,
        )

    scaled_logits = [adj / config.temperature for _, adj in penalized]
    base_probs = stable_softmax(scaled_logits)
    top_prob = base_probs[0] if base_probs else 0.0

    # Determine active mask sequentially: Top-K -> Min-P -> Top-P
    keep = [True] * len(penalized)
    if config.top_k > 0:
        for idx in range(config.top_k, len(keep)):
            keep[idx] = False

    if config.min_p > 0.0 and top_prob > 0.0:
        min_threshold = config.min_p * top_prob
        for idx, prob in enumerate(base_probs):
            if prob < min_threshold:
                keep[idx] = False

    if config.top_p < 1.0:
        cumulative = 0.0
        for idx, prob in enumerate(base_probs):
            if not keep[idx]:
                continue
            if idx > 0 and cumulative >= config.top_p:
                keep[idx] = False
            cumulative += prob

    # Guarantee at least the top-1 candidate remains selectable
    if not any(keep):
        keep[0] = True

    retained_mass = sum(prob for idx, prob in enumerate(base_probs) if keep[idx])
    clipped_mass = max(0.0, 1.0 - retained_mass)
    final_probs = [
        (prob / retained_mass) if keep[idx] and retained_mass > 0.0 else 0.0
        for idx, prob in enumerate(base_probs)
    ]

    result_candidates = [
        LogitCandidate(
            token=cand.token,
            display_token=cand.display_token,
            token_id=cand.token_id,
            logit=adj_logit,
            prob=final_probs[idx],
        )
        for idx, (cand, adj_logit) in enumerate(penalized)
    ]

    return SamplingResult(
        filtered_candidates=result_candidates,
        entropy_bits=compute_entropy_bits(final_probs),
        surviving_count=sum(1 for flag in keep if flag),
        clipped_mass=clipped_mass,
    )
