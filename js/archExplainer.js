/**
 * Stage 1 -> Stage 4 interactive renderer for pre-extracted real PyTorch GPT-2 traces.
 */

import {
  applySamplingPipeline,
  drawSampledCandidate,
  escapeHtml,
  heatColor,
  signedVectorColor,
} from "./mathUtils.js";

const TAPE_PALETTE = [
  "#06b6d4",
  "#6366f1",
  "#8b5cf6",
  "#ec4899",
  "#10b981",
  "#f59e0b",
  "#3b82f6",
  "#14b8a6",
];

export class ArchExplainerController {
  constructor(dataset) {
    this.dataset = dataset;
    this.presets = dataset.presets || [];
    this.activePresetIdx = 0;
    this.selectedTokenIdx = 0;
    this.selectedLayerIdx = 0;
    this.selectedHeadIdx = 0;
    this.cosineMode = "wte";
    this.arStepCount = 0;
    this.lastRoll = null;
    this.lastSampledToken = null;
  }

  get trace() {
    return this.presets[this.activePresetIdx];
  }

  init() {
    this.bindControls();
    this.selectPreset(0);
  }

  bindControls() {
    // Stage jump buttons in top pipeline rail
    document.querySelectorAll(".pipe-node[data-jump]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".pipe-node").forEach((n) => n.classList.remove("active"));
        btn.classList.add("active");
        const target = document.getElementById(btn.dataset.jump);
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });

    // Autoregressive stepper buttons
    document.getElementById("btn-ar-step")?.addEventListener("click", () => {
      const maxSteps = this.trace.continuation_steps?.length ?? 0;
      if (this.arStepCount < maxSteps) {
        this.arStepCount += 1;
        this.lastRoll = null;
        this.lastSampledToken = null;
        this.renderStage1();
        this.renderStage4();
      }
    });

    document.getElementById("btn-ar-reset")?.addEventListener("click", () => {
      this.arStepCount = 0;
      this.lastRoll = null;
      this.lastSampledToken = null;
      this.renderStage1();
      this.renderStage4();
    });

    // Cosine similarity view toggle
    document.querySelectorAll("#cosine-mode-btns .pill-selector").forEach((btn) => {
      btn.addEventListener("click", () => {
        document
          .querySelectorAll("#cosine-mode-btns .pill-selector")
          .forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.cosineMode = btn.dataset.cos;
        this.renderCosineMatrix();
      });
    });

    // Decoding presets
    const presetsMap = {
      greedy: { temp: 0.0, topk: 24, topp: 1.0, reppen: 1.0 },
      balanced: { temp: 0.8, topk: 24, topp: 0.9, reppen: 1.1 },
      creative: { temp: 1.35, topk: 24, topp: 0.95, reppen: 1.15 },
    };

    document.querySelectorAll("[data-decode-preset]").forEach((btn) => {
      btn.addEventListener("click", () => {
        document
          .querySelectorAll("[data-decode-preset]")
          .forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const cfg = presetsMap[btn.dataset.decodePreset];
        if (!cfg) return;
        document.getElementById("slider-temp").value = String(cfg.temp);
        document.getElementById("slider-topk").value = String(cfg.topk);
        document.getElementById("slider-topp").value = String(cfg.topp);
        document.getElementById("slider-reppen").value = String(cfg.reppen);
        this.lastRoll = null;
        this.renderStage4();
      });
    });

    // Sampling sliders
    ["slider-temp", "slider-topk", "slider-topp", "slider-reppen"].forEach((id) => {
      document.getElementById(id)?.addEventListener("input", () => {
        document
          .querySelectorAll("[data-decode-preset]")
          .forEach((b) => b.classList.remove("active"));
        this.lastRoll = null;
        this.renderStage4();
      });
    });

    document.getElementById("btn-roll-sampler")?.addEventListener("click", () => {
      const { filtered } = this.computeCurrentSampling();
      const roll = Math.random();
      const draw = drawSampledCandidate(filtered, roll);
      this.lastRoll = roll;
      this.lastSampledToken = draw.chosen?.display_token || "—";
      this.renderStage4();
    });
  }

  selectPreset(index) {
    this.activePresetIdx = index;
    const t = this.trace;
    this.selectedTokenIdx = Math.max(0, t.tokens.length - 1);
    // Start around a rich middle/late layer if available
    this.selectedLayerIdx = Math.min(8, t.layers.length - 1);
    this.selectedHeadIdx = 0;
    this.arStepCount = 0;
    this.lastRoll = null;
    this.lastSampledToken = null;

    this.renderPresetGrid();
    this.renderAllStages();
  }

  renderPresetGrid() {
    const grid = document.getElementById("preset-grid");
    const t = this.trace;
    grid.innerHTML = this.presets
      .map(
        (preset, idx) => `
      <button class="preset-btn ${idx === this.activePresetIdx ? "active" : ""}" data-preset-idx="${idx}" type="button">
        <span class="preset-cat">${escapeHtml(preset.category)}</span>
        <span class="preset-prompt">${escapeHtml(preset.prompt)}</span>
      </button>
    `
      )
      .join("");

    grid.querySelectorAll(".preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.selectPreset(Number(btn.dataset.presetIdx));
      });
    });

    const noteEl = document.getElementById("preset-note");
    if (noteEl) {
      noteEl.innerHTML = `<strong>${escapeHtml(t.title)}:</strong> ${escapeHtml(
        t.highlight_note
      )} <span class="mono" style="color:var(--accent-cyan)">[Model: ${escapeHtml(
        t.architecture.model_id
      )} · ${t.architecture.n_layers}L × ${t.architecture.n_heads}H · d_model=${
        t.architecture.d_model
      }]</span>`;
    }

    const linkEl = document.getElementById("explainer-companion-link");
    if (linkEl) {
      linkEl.href = t.companion_url;
      linkEl.textContent = `Open ${t.companion_lab} ↗`;
    }
  }

  renderAllStages() {
    this.updatePipelineShapes();
    this.renderStage1();
    this.renderStage2();
    this.renderStage3();
    this.renderStage4();
  }

  updatePipelineShapes() {
    const t = this.trace;
    const seqLen = t.tokens.length;
    const arch = t.architecture;
    document.getElementById("shape-stage-1").textContent = `text → ids [${seqLen}]`;
    document.getElementById("shape-stage-2").textContent = `h₀ = [${seqLen}, ${arch.d_model}]`;
    document.getElementById(
      "shape-stage-3"
    ).textContent = `${arch.n_layers}L × ${arch.n_heads}H · d_mlp=${arch.d_mlp}`;
    document.getElementById(
      "shape-stage-4"
    ).textContent = `logits [${arch.vocab_size.toLocaleString()}]`;
  }

  renderStage1() {
    const t = this.trace;
    const extraTokens = (t.continuation_steps || [])
      .slice(0, this.arStepCount)
      .map((s) => ({ ...s.appended_token, appended: true }));
    const allTokens = [...t.tokens.map((tok) => ({ ...tok, appended: false })), ...extraTokens];

    const ribbon = document.getElementById("token-ribbon");
    ribbon.innerHTML = allTokens
      .map((tok, idx) => {
        const clickable = !tok.appended;
        const selectedClass = clickable && idx === this.selectedTokenIdx ? "selected" : "";
        const appendedClass = tok.appended ? "appended" : "";
        return `
        <button class="token-chip ${selectedClass} ${appendedClass}" data-tok-idx="${idx}" type="button" ${
          clickable ? "" : 'title="Autoregressively appended step"'
        }>
          <span class="token-text">${escapeHtml(tok.display_text)}</span>
          <span class="token-meta">id:${tok.token_id} · p:${tok.position}</span>
        </button>
      `;
      })
      .join("");

    ribbon.querySelectorAll(".token-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const idx = Number(chip.dataset.tokIdx);
        if (idx < t.tokens.length) {
          this.selectedTokenIdx = idx;
          this.renderStage1();
          this.renderStage2();
          this.renderBipartiteSvg();
        }
      });
    });

    const maxSteps = t.continuation_steps?.length ?? 0;
    const stepBtn = document.getElementById("btn-ar-step");
    const resetBtn = document.getElementById("btn-ar-reset");
    const arStatus = document.getElementById("ar-status");

    if (stepBtn) stepBtn.disabled = this.arStepCount >= maxSteps;
    if (resetBtn) resetBtn.disabled = this.arStepCount === 0;
    if (arStatus) {
      arStatus.textContent =
        this.arStepCount === 0
          ? `Base prompt (${t.tokens.length} tokens) · ${maxSteps} pre-extracted continuation steps ready`
          : `Step +${this.arStepCount}/${maxSteps} active (${allTokens.length} total tokens)`;
    }
  }

  renderStage2() {
    const t = this.trace;
    const emb = t.embeddings;
    const tokIdx = Math.min(this.selectedTokenIdx, t.tokens.length - 1);
    const tok = t.tokens[tokIdx];

    document.getElementById(
      "emb-selected-label"
    ).textContent = `Token #${tokIdx} '${tok.display_text}' (Dims 0–${emb.slice_dims - 1} of ${
      emb.hidden_size
    })`;
    document.getElementById("emb-norms-badge").textContent = `||wte||=${emb.wte_norms[
      tokIdx
    ].toFixed(2)}  ||wpe||=${emb.wpe_norms[tokIdx].toFixed(2)}  ||h₀||=${emb.combined_norms[
      tokIdx
    ].toFixed(2)}`;

    const renderStrip = (containerId, values) => {
      const container = document.getElementById(containerId);
      container.innerHTML = values
        .map(
          (val, dimIdx) =>
            `<div class="vec-cell" style="background:${signedVectorColor(
              val,
              0.32
            )}" title="dim ${dimIdx}: ${val.toFixed(4)}"></div>`
        )
        .join("");
    };

    renderStrip("vec-wte-strip", emb.wte_slices[tokIdx]);
    renderStrip("vec-wpe-strip", emb.wpe_slices[tokIdx]);
    renderStrip("vec-combined-strip", emb.combined_slices[tokIdx]);

    this.renderCosineMatrix();
  }

  renderCosineMatrix() {
    const t = this.trace;
    let matrix = t.embeddings.wte_cosine_matrix;
    if (this.cosineMode === "combined") matrix = t.embeddings.combined_cosine_matrix;
    if (this.cosineMode === "final") matrix = t.final_cosine_matrix;

    const wrap = document.getElementById("cosine-matrix-wrap");
    const headers = t.tokens
      .map((tok) => `<th class="col-hdr">${escapeHtml(tok.display_text)}</th>`)
      .join("");

    const rows = matrix
      .map((row, rIdx) => {
        const cells = row
          .map((val, cIdx) => {
            const bg = signedVectorColor(val, 1.0);
            return `<td style="background:${bg};color:${
              Math.abs(val) > 0.55 ? "#fff" : "#cbd5e1"
            }" title="cos(${t.tokens[rIdx].display_text}, ${t.tokens[cIdx].display_text}) = ${val.toFixed(
              3
            )}">${val.toFixed(2)}</td>`;
          })
          .join("");
        return `<tr><th class="row-hdr">${escapeHtml(t.tokens[rIdx].display_text)}</th>${cells}</tr>`;
      })
      .join("");

    wrap.innerHTML = `<table class="attn-table"><thead><tr><th></th>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
  }

  renderStage3() {
    const t = this.trace;
    const layerScrub = document.getElementById("layer-scrubber");
    const headScrub = document.getElementById("head-scrubber");

    layerScrub.innerHTML = t.layers
      .map(
        (layer) => `
      <button class="pill-selector ${
        layer.layer_idx === this.selectedLayerIdx ? "active" : ""
      }" data-layer-idx="${layer.layer_idx}" type="button">L${layer.layer_idx}</button>
    `
      )
      .join("");

    const curLayer = t.layers[this.selectedLayerIdx];
    headScrub.innerHTML = curLayer.heads
      .map(
        (head) => `
      <button class="pill-selector ${
        head.head_idx === this.selectedHeadIdx ? "active" : ""
      }" data-head-idx="${head.head_idx}" type="button">H${head.head_idx}</button>
    `
      )
      .join("");

    layerScrub.querySelectorAll("[data-layer-idx]").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.selectedLayerIdx = Number(btn.dataset.layerIdx);
        this.renderStage3();
      });
    });

    headScrub.querySelectorAll("[data-head-idx]").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.selectedHeadIdx = Number(btn.dataset.headIdx);
        this.renderStage3();
      });
    });

    const head = curLayer.heads[this.selectedHeadIdx];
    document.getElementById(
      "attn-panel-title"
    ).textContent = `Layer ${curLayer.layer_idx} · Head ${head.head_idx} Causal Attention`;
    document.getElementById(
      "attn-entropy-badge"
    ).textContent = `Last-Tok Entropy: ${head.last_token_entropy.toFixed(2)} bits · Top Key: '${
      t.tokens[head.last_token_top_key_idx]?.display_text ?? "?"
    }'`;

    this.renderBipartiteSvg();
    this.renderAttentionMatrix();
    this.renderLogitLens(curLayer);
    this.renderMlpNeurons(curLayer);
  }

  renderBipartiteSvg() {
    const t = this.trace;
    const layer = t.layers[this.selectedLayerIdx];
    const head = layer.heads[this.selectedHeadIdx];
    const svg = document.getElementById("bipartite-svg");
    const n = t.tokens.length;
    const height = 280;
    const width = 640;
    const padY = 22;
    const stepY = n > 1 ? (height - 2 * padY) / (n - 1) : 0;

    // Highlight incoming attention for either selected query token or final token
    const focusQ = Math.min(this.selectedTokenIdx, n - 1);
    const lines = [];

    for (let q = 0; q < n; q += 1) {
      for (let k = 0; k <= q; k += 1) {
        const w = head.matrix[q][k];
        if (w < 0.04) continue;
        const y1 = padY + q * stepY;
        const y2 = padY + k * stepY;
        const isFocus = q === focusQ;
        const stroke = isFocus ? "#ec4899" : "#38bdf8";
        const opacity = isFocus ? Math.min(0.95, 0.25 + w) : Math.min(0.32, w * 0.45);
        const strokeW = isFocus ? (1.2 + w * 4.5).toFixed(2) : (0.6 + w * 2).toFixed(2);
        lines.push(
          `<path d="M 185 ${y1.toFixed(1)} C 320 ${y1.toFixed(1)}, 320 ${y2.toFixed(
            1
          )}, 455 ${y2.toFixed(1)}" fill="none" stroke="${stroke}" stroke-opacity="${opacity.toFixed(
            2
          )}" stroke-width="${strokeW}" />`
        );
      }
    }

    const leftLabels = t.tokens
      .map((tok, i) => {
        const y = padY + i * stepY;
        const isFocus = i === focusQ;
        return `<text x="175" y="${(y + 4).toFixed(
          1
        )}" text-anchor="end" font-family="JetBrains Mono, monospace" font-size="11.5" fill="${
          isFocus ? "#f472b6" : "#cbd5e1"
        }" font-weight="${isFocus ? "700" : "400"}">Q${i}: ${escapeHtml(tok.display_text)}</text>`;
      })
      .join("");

    const rightLabels = t.tokens
      .map((tok, i) => {
        const y = padY + i * stepY;
        const w = i <= focusQ ? head.matrix[focusQ][i] : 0;
        return `<text x="465" y="${(y + 4).toFixed(
          1
        )}" text-anchor="start" font-family="JetBrains Mono, monospace" font-size="11.5" fill="${
          w > 0.15 ? "#38bdf8" : "#94a3b8"
        }">K${i}: ${escapeHtml(tok.display_text)} (${(w * 100).toFixed(0)}%)</text>`;
      })
      .join("");

    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `${lines.join("")}${leftLabels}${rightLabels}`;
  }

  renderAttentionMatrix() {
    const t = this.trace;
    const head = t.layers[this.selectedLayerIdx].heads[this.selectedHeadIdx];
    const headers = t.tokens
      .map((tok, idx) => `<th class="col-hdr" title="Key ${idx}">${escapeHtml(tok.display_text)}</th>`)
      .join("");

    const rows = head.matrix
      .map((row, qIdx) => {
        const cells = row
          .map((val, kIdx) => {
            if (kIdx > qIdx) {
              return `<td class="attn-cell-causal-mask" title="Causal mask (-∞)">×</td>`;
            }
            const bg = heatColor(val, 0.85);
            return `<td style="background:${bg};color:${
              val > 0.35 ? "#fff" : "#cbd5e1"
            }" title="Q=${t.tokens[qIdx].display_text} -> K=${t.tokens[kIdx].display_text}: ${(
              val * 100
            ).toFixed(1)}%">${val.toFixed(2)}</td>`;
          })
          .join("");
        return `<tr><th class="row-hdr">${escapeHtml(t.tokens[qIdx].display_text)}</th>${cells}</tr>`;
      })
      .join("");

    document.getElementById(
      "attn-matrix-wrap"
    ).innerHTML = `<table class="attn-table"><thead><tr><th>Q \\ K</th>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
  }

  renderLogitLens(layer) {
    document.getElementById("lens-layer-num").textContent = String(layer.layer_idx);
    const maxProb = Math.max(...layer.logit_lens.map((c) => c.prob), 0.05);
    document.getElementById("logit-lens-bars").innerHTML = layer.logit_lens
      .map(
        (cand, idx) => `
      <div class="logit-bar-row">
        <div class="bar-token-label" title="id:${cand.token_id}">${escapeHtml(cand.display_token)}</div>
        <div class="bar-track">
          <div class="bar-fill ${idx === 0 ? "winner" : ""}" style="width:${Math.max(
          2,
          (cand.prob / maxProb) * 100
        ).toFixed(1)}%"></div>
        </div>
        <div class="bar-metrics">${(cand.prob * 100).toFixed(1)}% · z=${cand.logit.toFixed(1)}</div>
      </div>
    `
      )
      .join("");
  }

  renderMlpNeurons(layer) {
    document.getElementById(
      "block-delta-badge"
    ).textContent = `ΔAttn=${layer.attn_delta_norm.toFixed(1)} · ΔMLP=${layer.mlp_delta_norm.toFixed(1)}`;

    const maxAct = Math.max(...layer.top_mlp_neurons.map((n) => Math.abs(n.activation)), 1.0);
    document.getElementById("mlp-neuron-bars").innerHTML = layer.top_mlp_neurons
      .slice(0, 6)
      .map((n) => {
        const pct = Math.min(100, (Math.abs(n.activation) / maxAct) * 100);
        return `
        <div class="logit-bar-row">
          <div class="bar-token-label">Neuron #${n.neuron_idx}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${pct.toFixed(1)}%;background:${
          n.activation >= 0
            ? "linear-gradient(90deg, #8b5cf6, #ec4899)"
            : "linear-gradient(90deg, #f59e0b, #ef4444)"
        }"></div>
          </div>
          <div class="bar-metrics">${n.activation.toFixed(2)} (pre ${n.pre_activation.toFixed(
          2
        )})</div>
        </div>
      `;
      })
      .join("");
  }

  getActiveCandidatesAndEntropy() {
    const t = this.trace;
    if (this.arStepCount === 0) {
      return {
        candidates: t.final_candidates,
        rawEntropyBits: t.final_vocab_entropy_bits,
        seenTokenIds: new Set(t.tokens.map((tok) => tok.token_id)),
      };
    }
    const step = t.continuation_steps[this.arStepCount - 1];
    const seen = new Set(t.tokens.map((tok) => tok.token_id));
    for (let i = 0; i < this.arStepCount; i += 1) {
      seen.add(t.continuation_steps[i].appended_token.token_id);
    }
    return {
      candidates: step.top_candidates,
      rawEntropyBits: step.vocab_entropy_bits,
      seenTokenIds: seen,
    };
  }

  computeCurrentSampling() {
    const temp = Number(document.getElementById("slider-temp").value);
    const topK = Number(document.getElementById("slider-topk").value);
    const topP = Number(document.getElementById("slider-topp").value);
    const repetitionPenalty = Number(document.getElementById("slider-reppen").value);

    document.getElementById("val-temp").textContent = temp.toFixed(2);
    document.getElementById("val-topk").textContent = topK === 0 ? "Off" : String(topK);
    document.getElementById("val-topp").textContent = topP.toFixed(2);
    document.getElementById("val-reppen").textContent = repetitionPenalty.toFixed(2);

    const { candidates, rawEntropyBits, seenTokenIds } = this.getActiveCandidatesAndEntropy();
    const res = applySamplingPipeline(
      candidates,
      { temperature: temp, topK, topP, repetitionPenalty, minP: 0 },
      seenTokenIds
    );
    return { ...res, rawEntropyBits };
  }

  renderStage4() {
    const res = this.computeCurrentSampling();

    document.getElementById("kpi-vocab-entropy").textContent = `${res.rawEntropyBits.toFixed(2)} bits`;
    document.getElementById("kpi-filtered-entropy").textContent = `${res.entropyBits.toFixed(2)} bits`;
    document.getElementById("kpi-surviving").textContent = `${res.survivingCount} / ${res.filtered.length}`;
    document.getElementById("kpi-sampled-token").textContent = this.lastSampledToken
      ? `${this.lastSampledToken} (u=${this.lastRoll.toFixed(3)})`
      : "Click Roll Dice";

    // Render Probability Tape
    const tapeEl = document.getElementById("explainer-sampling-tape");
    const activeItems = res.filtered.filter((c) => c.prob > 0.002);
    const segments = activeItems
      .map((item, idx) => {
        const widthPct = (item.prob * 100).toFixed(2);
        const color = TAPE_PALETTE[idx % TAPE_PALETTE.length];
        const label = item.prob > 0.055 ? `${escapeHtml(item.display_token)}` : "";
        return `<div class="tape-segment" style="width:${widthPct}%;background:${color}" title="${escapeHtml(
          item.display_token
        )}: ${(item.prob * 100).toFixed(2)}%">${label}</div>`;
      })
      .join("");

    const marker =
      this.lastRoll !== null
        ? `<div class="tape-marker" style="left:${(this.lastRoll * 100).toFixed(2)}%"></div>`
        : "";
    tapeEl.innerHTML = `${segments}${marker}`;

    // Render Logit & Probability Bars (top 14 candidates)
    const barsEl = document.getElementById("explainer-logits-bars");
    const topShow = res.filtered.slice(0, 14);
    const maxP = Math.max(...topShow.map((c) => c.prob), 0.1);

    barsEl.innerHTML = topShow
      .map((cand, idx) => {
        const widthPct = cand.kept ? Math.max(2, (cand.prob / maxP) * 100) : 0;
        return `
        <div class="logit-bar-row ${cand.kept ? "" : "filtered-out"}">
          <div class="bar-token-label" title="id:${cand.token_id}${
          cand.seenPenalty ? " (repetition penalty applied)" : ""
        }">${escapeHtml(cand.display_token)}</div>
          <div class="bar-track">
            <div class="bar-fill ${idx === 0 && cand.kept ? "winner" : ""}" style="width:${widthPct.toFixed(
          1
        )}%"></div>
          </div>
          <div class="bar-metrics">${(cand.prob * 100).toFixed(1)}% · z=${cand.logit.toFixed(1)}</div>
        </div>
      `;
      })
      .join("");
  }
}
