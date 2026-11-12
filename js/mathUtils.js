/**
 * Pure client-side mathematics matching src/core/sampling_math.py.
 * Handles stable softmax, Shannon entropy, temperature/top-k/top-p/min-p filtering,
 * repetition penalties, and perceptually balanced color mapping.
 */

export function formatDisplayToken(raw) {
  if (!raw) return "∅";
  return raw
    .replace(/\n/g, "↵")
    .replace(/\r/g, "")
    .replace(/\t/g, "⇥")
    .replace(/ /g, "·");
}

export function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function stableSoftmax(logits) {
  if (!logits || logits.length === 0) return [];
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }
  const exps = new Float64Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const val = Math.exp(logits[i] - maxLogit);
    exps[i] = val;
    total += val;
  }
  if (total <= 0) {
    const uniform = 1 / logits.length;
    return Array.from({ length: logits.length }, () => uniform);
  }
  const probs = new Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) {
    probs[i] = exps[i] / total;
  }
  return probs;
}

export function computeEntropyBits(probs) {
  let entropy = 0;
  for (let i = 0; i < probs.length; i += 1) {
    const p = probs[i];
    if (p > 1e-12) {
      entropy -= p * Math.log2(p);
    }
  }
  return Math.max(0, entropy);
}

function adjustRepetitionLogit(logit, penalty) {
  if (penalty === 1) return logit;
  return logit > 0 ? logit / penalty : logit * penalty;
}

/**
 * Apply repetition penalty, temperature, Top-K, Min-P, and Top-P nucleus filtering.
 */
export function applySamplingPipeline(candidates, config, seenTokenIds = new Set()) {
  if (!candidates || candidates.length === 0) {
    return { filtered: [], entropyBits: 0, survivingCount: 0, clippedMass: 0 };
  }

  const penalty = config.repetitionPenalty ?? 1.0;
  const temperature = config.temperature ?? 0.8;
  const topK = config.topK ?? 0;
  const topP = config.topP ?? 1.0;
  const minP = config.minP ?? 0.0;

  const penalized = candidates
    .map((c) => {
      const seen = seenTokenIds.has(c.token_id);
      const adjLogit = seen ? adjustRepetitionLogit(c.logit, penalty) : c.logit;
      return { cand: c, adjLogit, seen };
    })
    .sort((a, b) => b.adjLogit - a.adjLogit);

  // Greedy path for T -> 0
  if (temperature <= 1e-4) {
    const filtered = penalized.map((item, idx) => ({
      ...item.cand,
      logit: item.adjLogit,
      rawProb: item.cand.prob,
      prob: idx === 0 ? 1.0 : 0.0,
      kept: idx === 0,
      seenPenalty: item.seen && penalty > 1.0,
    }));
    return {
      filtered,
      entropyBits: 0,
      survivingCount: 1,
      clippedMass: Math.max(0, 1 - (filtered[0]?.rawProb ?? 1)),
    };
  }

  const scaled = penalized.map((item) => item.adjLogit / temperature);
  const baseProbs = stableSoftmax(scaled);
  const topProb = baseProbs[0] ?? 0;

  const keep = new Array(penalized.length).fill(true);
  if (topK > 0) {
    for (let i = topK; i < keep.length; i += 1) {
      keep[i] = false;
    }
  }
  if (minP > 0 && topProb > 0) {
    const minThresh = minP * topProb;
    for (let i = 0; i < baseProbs.length; i += 1) {
      if (baseProbs[i] < minThresh) keep[i] = false;
    }
  }
  if (topP < 1.0) {
    let cum = 0;
    for (let i = 0; i < baseProbs.length; i += 1) {
      if (!keep[i]) continue;
      if (i > 0 && cum >= topP) {
        keep[i] = false;
      }
      cum += baseProbs[i];
    }
  }

  if (!keep.some(Boolean)) keep[0] = true;

  let retainedMass = 0;
  for (let i = 0; i < baseProbs.length; i += 1) {
    if (keep[i]) retainedMass += baseProbs[i];
  }

  const probsOut = baseProbs.map((p, i) => (keep[i] && retainedMass > 0 ? p / retainedMass : 0));
  const filtered = penalized.map((item, i) => ({
    ...item.cand,
    logit: item.adjLogit,
    rawProb: baseProbs[i],
    prob: probsOut[i],
    kept: keep[i],
    seenPenalty: item.seen && penalty > 1.0,
  }));

  return {
    filtered,
    entropyBits: computeEntropyBits(probsOut),
    survivingCount: keep.filter(Boolean).length,
    clippedMass: Math.max(0, 1 - retainedMass),
  };
}

export function drawSampledCandidate(filtered, uniformRoll = Math.random()) {
  let cumulative = 0;
  for (const item of filtered) {
    if (item.prob <= 0) continue;
    cumulative += item.prob;
    if (uniformRoll <= cumulative) {
      return { chosen: item, roll: uniformRoll, cumulativeEnd: cumulative };
    }
  }
  const fallback = filtered.find((c) => c.prob > 0) || filtered[0];
  return { chosen: fallback, roll: uniformRoll, cumulativeEnd: 1.0 };
}

export function heatColor(value, maxVal = 1.0) {
  const t = Math.max(0, Math.min(1, maxVal > 0 ? value / maxVal : 0));
  if (t < 0.01) return "rgba(15, 23, 42, 0.72)";
  // Smooth cyan -> indigo -> pink perceptual ramp
  if (t < 0.5) {
    const k = t * 2;
    const r = Math.round(14 + k * (99 - 14));
    const g = Math.round(165 + k * (102 - 165));
    const b = Math.round(233 + k * (241 - 233));
    return `rgba(${r}, ${g}, ${b}, ${(0.2 + 0.65 * k).toFixed(2)})`;
  }
  const k = (t - 0.5) * 2;
  const r = Math.round(99 + k * (236 - 99));
  const g = Math.round(102 + k * (72 - 102));
  const b = Math.round(241 + k * (153 - 241));
  return `rgba(${r}, ${g}, ${b}, ${(0.85 + 0.15 * k).toFixed(2)})`;
}

export function signedVectorColor(val, scale = 0.35) {
  const norm = Math.max(-1, Math.min(1, val / scale));
  if (norm >= 0) {
    const alpha = (0.15 + 0.85 * norm).toFixed(2);
    return `rgba(6, 182, 212, ${alpha})`;
  }
  const alpha = (0.15 + 0.85 * Math.abs(norm)).toFixed(2);
  return `rgba(236, 72, 153, ${alpha})`;
}
