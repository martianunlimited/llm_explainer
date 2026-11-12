/**
 * UI Controller for Tab 2: Live Browser LLM Studio (WebGPU / WASM ONNX execution).
 */

import { LiveWebGPUEngine } from "./liveWebGPUEngine.js";
import { applySamplingPipeline, drawSampledCandidate, escapeHtml } from "./mathUtils.js";

export class LiveStudioController {
  constructor() {
    this.engine = new LiveWebGPUEngine();
    this.latestInspect = null;
    this.isGenerating = false;
    this.abortRequested = false;
    this.activeTokenIds = [];
  }

  init() {
    this.bindControls();
  }

  bindControls() {
    document.getElementById("btn-load-model")?.addEventListener("click", () => this.handleLoadModel());
    document.getElementById("btn-live-inspect")?.addEventListener("click", () => this.runInspect());
    document.getElementById("btn-live-step")?.addEventListener("click", () => this.stepOneToken());
    document
      .getElementById("btn-live-generate")
      ?.addEventListener("click", () => this.autoGenerate(16));
    document.getElementById("btn-live-stop")?.addEventListener("click", () => {
      this.abortRequested = true;
    });

    document.querySelectorAll("[data-live-preset]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        document.getElementById("live-prompt-input").value = btn.dataset.livePreset;
        this.activeTokenIds = [];
        if (this.engine.isLoaded()) {
          await this.runInspect();
        }
      });
    });

    document.getElementById("live-prompt-input")?.addEventListener("input", () => {
      this.activeTokenIds = [];
    });

    ["live-temp", "live-topk", "live-topp"].forEach((id) => {
      document.getElementById(id)?.addEventListener("input", () => {
        if (this.latestInspect) this.renderDistribution();
      });
    });
  }

  async handleLoadModel() {
    const modelId = document.getElementById("live-model-select").value;
    const loadBtn = document.getElementById("btn-load-model");
    const progWrap = document.getElementById("live-progress-wrap");
    const progText = document.getElementById("live-progress-text");
    const progBar = document.getElementById("live-progress-bar");
    const badge = document.getElementById("live-device-badge");

    loadBtn.disabled = true;
    progWrap.style.display = "block";
    badge.textContent = "Loading Model...";

    try {
      const info = await this.engine.loadModel(modelId, ({ stage, progress }) => {
        progText.textContent = stage;
        progBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
      });
      badge.textContent = `${info.device.toUpperCase()} · Ready`;
      document.getElementById("live-kpi-backend").textContent = info.device.toUpperCase();
      document.getElementById("btn-live-inspect").disabled = false;
      document.getElementById("btn-live-step").disabled = false;
      document.getElementById("btn-live-generate").disabled = false;
      await this.runInspect();
    } catch (err) {
      progText.textContent = `Load error: ${err?.message || err}`;
      badge.textContent = "Error";
    } finally {
      loadBtn.disabled = false;
    }
  }

  async runInspect(useCurrentTokenIds = false) {
    if (!this.engine.isLoaded()) return;
    const promptText = document.getElementById("live-prompt-input").value;
    const payload =
      useCurrentTokenIds && this.activeTokenIds.length > 0
        ? { tokenIds: this.activeTokenIds, topK: 20 }
        : { text: promptText, topK: 20 };

    const res = await this.engine.forwardNextToken(payload);
    this.latestInspect = res;
    this.activeTokenIds = res.tokenIds;

    document.getElementById("live-kpi-backend").textContent = res.backend.toUpperCase();
    document.getElementById("live-kpi-latency").textContent = `${res.latencyMs.toFixed(1)} ms`;
    document.getElementById("live-kpi-tokens").textContent = `${res.tokens.length} toks`;
    document.getElementById("live-kpi-entropy").textContent = `${res.vocabEntropyBits.toFixed(2)} bits`;

    this.renderTokenRibbon(res.tokens);
    this.renderDistribution();
  }

  renderTokenRibbon(tokens) {
    document.getElementById("live-tok-count").textContent = `${tokens.length} BPE tokens`;
    document.getElementById("live-token-ribbon").innerHTML = tokens
      .map(
        (tok) => `
      <div class="token-chip">
        <span class="token-text">${escapeHtml(tok.display_text)}</span>
        <span class="token-meta">id:${tok.token_id} · p:${tok.position}</span>
      </div>
    `
      )
      .join("");
  }

  getSamplingConfig() {
    const temperature = Number(document.getElementById("live-temp").value);
    const topK = Number(document.getElementById("live-topk").value);
    const topP = Number(document.getElementById("live-topp").value);
    document.getElementById("live-val-temp").textContent = temperature.toFixed(2);
    document.getElementById("live-val-topk").textContent = topK === 0 ? "Off" : String(topK);
    document.getElementById("live-val-topp").textContent = topP.toFixed(2);
    return { temperature, topK, topP, repetitionPenalty: 1.1, minP: 0 };
  }

  renderDistribution(sampledBadge = "") {
    if (!this.latestInspect) return;
    const cfg = this.getSamplingConfig();
    const seen = new Set(this.latestInspect.tokenIds);
    const sampled = applySamplingPipeline(this.latestInspect.candidates, cfg, seen);

    if (sampledBadge) {
      document.getElementById("live-sampled-badge").textContent = sampledBadge;
    }

    const topShow = sampled.filtered.slice(0, 12);
    const maxP = Math.max(...topShow.map((c) => c.prob), 0.1);
    document.getElementById("live-logits-bars").innerHTML = topShow
      .map((cand, idx) => {
        const widthPct = cand.kept ? Math.max(2, (cand.prob / maxP) * 100) : 0;
        return `
        <div class="logit-bar-row ${cand.kept ? "" : "filtered-out"}">
          <div class="bar-token-label" title="id:${cand.token_id}">${escapeHtml(
          cand.display_token
        )}</div>
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
    return sampled;
  }

  async stepOneToken() {
    if (!this.engine.isLoaded()) return;
    if (!this.latestInspect) await this.runInspect(false);

    const sampled = this.renderDistribution();
    const draw = drawSampledCandidate(sampled.filtered, Math.random());
    if (!draw.chosen) return;

    this.activeTokenIds = [...this.activeTokenIds, draw.chosen.token_id];
    const promptInput = document.getElementById("live-prompt-input");
    promptInput.value = this.engine.tokenizer.decode(this.activeTokenIds, {
      skip_special_tokens: false,
    });
    await this.runInspect(true);
    document.getElementById(
      "live-sampled-badge"
    ).textContent = `Appended '${draw.chosen.display_token}'`;
  }

  async autoGenerate(steps = 16) {
    if (this.isGenerating || !this.engine.isLoaded()) return;
    this.isGenerating = true;
    this.abortRequested = false;

    const genBtn = document.getElementById("btn-live-generate");
    const stepBtn = document.getElementById("btn-live-step");
    const stopBtn = document.getElementById("btn-live-stop");
    genBtn.disabled = true;
    stepBtn.disabled = true;
    stopBtn.disabled = false;

    try {
      for (let i = 0; i < steps; i += 1) {
        if (this.abortRequested) break;
        await this.stepOneToken();
      }
    } finally {
      this.isGenerating = false;
      genBtn.disabled = false;
      stepBtn.disabled = false;
      stopBtn.disabled = true;
    }
  }
}
