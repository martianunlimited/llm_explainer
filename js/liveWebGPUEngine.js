/**
 * Authentic client-side LLM execution via @huggingface/transformers v3 (WebGPU + WASM).
 * Loads real quantized ONNX weights and runs causal forward passes for live logits.
 */

import { formatDisplayToken, stableSoftmax, computeEntropyBits } from "./mathUtils.js";

const TRANSFORMERS_CDN =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3/dist/transformers.min.js";

export const BROWSER_MODELS = {
  "Xenova/distilgpt2": {
    id: "Xenova/distilgpt2",
    dtype: "q8",
    label: "DistilGPT-2 (81 MB q8 · 6L × 12H)",
  },
  "Xenova/gpt2": {
    id: "Xenova/gpt2",
    dtype: "q8",
    label: "GPT-2 Small (122 MB q8 · 12L × 12H)",
  },
  "HuggingFaceTB/SmolLM2-135M-Instruct": {
    id: "HuggingFaceTB/SmolLM2-135M-Instruct",
    dtype: "q4f16",
    label: "SmolLM2-135M (112 MB q4f16 · 30L Llama GQA)",
  },
};

export class LiveWebGPUEngine {
  constructor() {
    this.transformers = null;
    this.tokenizer = null;
    this.model = null;
    this.activeModelId = null;
    this.activeDevice = null;
  }

  async _ensureLibrary() {
    if (!this.transformers) {
      this.transformers = await import(TRANSFORMERS_CDN);
      this.transformers.env.allowLocalModels = false;
      this.transformers.env.useBrowserCache = true;
    }
    return this.transformers;
  }

  async detectBestDevice() {
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) return "webgpu";
      } catch {
        // Ignore adapter probe error and fall back cleanly to WASM.
      }
    }
    return "wasm";
  }

  async loadModel(modelId, onProgress) {
    const lib = await this._ensureLibrary();
    const spec = BROWSER_MODELS[modelId] || { id: modelId, dtype: "q8", label: modelId };

    if (onProgress) onProgress({ stage: "Probing WebGPU adapter...", progress: 2 });
    const preferredDevice = await this.detectBestDevice();

    if (onProgress) onProgress({ stage: `Loading tokenizer for ${spec.id}...`, progress: 8 });
    this.tokenizer = await lib.AutoTokenizer.from_pretrained(spec.id, {
      progress_callback: (evt) => {
        if (evt.status === "progress" && onProgress) {
          onProgress({
            stage: `Downloading ${evt.file || "tokenizer"} (${Math.round(evt.progress || 0)}%)`,
            progress: 8 + Math.round((evt.progress || 0) * 0.12),
          });
        }
      },
    });

    const loadWithDevice = async (device) => {
      return lib.AutoModelForCausalLM.from_pretrained(spec.id, {
        dtype: spec.dtype,
        device,
        progress_callback: (evt) => {
          if (evt.status === "progress" && onProgress) {
            onProgress({
              stage: `Loading ${evt.file || "ONNX weights"} on ${device.toUpperCase()} (${Math.round(
                evt.progress || 0
              )}%)`,
              progress: 20 + Math.round((evt.progress || 0) * 0.78),
            });
          }
        },
      });
    };

    try {
      this.model = await loadWithDevice(preferredDevice);
      this.activeDevice = preferredDevice;
    } catch (err) {
      if (preferredDevice === "webgpu") {
        if (onProgress) {
          onProgress({
            stage: "WebGPU adapter unsupported for this ONNX graph; falling back to WASM...",
            progress: 22,
          });
        }
        this.model = await loadWithDevice("wasm");
        this.activeDevice = "wasm";
      } else {
        throw err;
      }
    }

    this.activeModelId = spec.id;
    if (onProgress) {
      onProgress({
        stage: `Ready on ${this.activeDevice.toUpperCase()} (${spec.id})`,
        progress: 100,
      });
    }
    return { modelId: this.activeModelId, device: this.activeDevice };
  }

  isLoaded() {
    return Boolean(this.model && this.tokenizer);
  }

  tokenizeText(text) {
    if (!this.tokenizer) throw new Error("Load a model before tokenizing.");
    const encoded = this.tokenizer(text);
    const ids = Array.from(encoded.input_ids.data, (v) => Number(v));
    const tokens = ids.map((tid, pos) => {
      const piece = this.tokenizer.decode([tid], { skip_special_tokens: false });
      return {
        position: pos,
        token_id: tid,
        text: piece,
        display_text: formatDisplayToken(piece),
      };
    });
    return { encoded, ids, tokens };
  }

  /**
   * Run a genuine ONNX forward pass over either text or explicit token IDs.
   * Extracts exact full-vocabulary logits, softmax probabilities, entropy, and top-K candidates.
   */
  async forwardNextToken({ text = null, tokenIds = null, topK = 20 }) {
    if (!this.isLoaded()) throw new Error("Load a model before running inference.");
    const lib = await this._ensureLibrary();

    let ids;
    let tokens;
    let modelInputs;

    if (Array.isArray(tokenIds) && tokenIds.length > 0) {
      ids = tokenIds;
      const bigIds = BigInt64Array.from(ids.map((v) => BigInt(v)));
      const mask = new BigInt64Array(ids.length).fill(1n);
      modelInputs = {
        input_ids: new lib.Tensor("int64", bigIds, [1, ids.length]),
        attention_mask: new lib.Tensor("int64", mask, [1, ids.length]),
      };
      tokens = ids.map((tid, pos) => {
        const piece = this.tokenizer.decode([tid], { skip_special_tokens: false });
        return {
          position: pos,
          token_id: tid,
          text: piece,
          display_text: formatDisplayToken(piece),
        };
      });
    } else {
      const parsed = this.tokenizeText(text || " ");
      ids = parsed.ids;
      tokens = parsed.tokens;
      modelInputs = {
        input_ids: parsed.encoded.input_ids,
        attention_mask: parsed.encoded.attention_mask,
      };
    }

    const t0 = performance.now();
    const outputs = await this.model(modelInputs);
    const latencyMs = performance.now() - t0;

    const logitsTensor = outputs.logits;
    const [, seqLen, vocabSize] = logitsTensor.dims;
    const offset = (seqLen - 1) * vocabSize;
    const lastLogits = logitsTensor.data.subarray(offset, offset + vocabSize);

    // Overflow-safe full-vocabulary softmax
    let maxLogit = -Infinity;
    for (let i = 0; i < vocabSize; i += 1) {
      if (lastLogits[i] > maxLogit) maxLogit = lastLogits[i];
    }

    let sumExp = 0;
    const exps = new Float64Array(vocabSize);
    for (let i = 0; i < vocabSize; i += 1) {
      const e = Math.exp(lastLogits[i] - maxLogit);
      exps[i] = e;
      sumExp += e;
    }

    let entropyBits = 0;
    for (let i = 0; i < vocabSize; i += 1) {
      const p = exps[i] / sumExp;
      if (p > 1e-12) entropyBits -= p * Math.log2(p);
    }

    // Extract Top-K via partial selection scan
    const indices = Array.from({ length: vocabSize }, (_, idx) => idx);
    indices.sort((a, b) => lastLogits[b] - lastLogits[a]);
    const limit = Math.min(topK, vocabSize);
    const candidates = [];

    for (let rank = 0; rank < limit; rank += 1) {
      const tid = indices[rank];
      const tokenStr = this.tokenizer.decode([tid], { skip_special_tokens: false });
      candidates.push({
        token: tokenStr,
        display_token: formatDisplayToken(tokenStr),
        token_id: tid,
        logit: Number(lastLogits[tid].toFixed(4)),
        prob: Number((exps[tid] / sumExp).toFixed(5)),
      });
    }

    // Recompute entropy fallback cleanly just in case
    if (!Number.isFinite(entropyBits)) {
      entropyBits = computeEntropyBits(stableSoftmax(candidates.map((c) => c.logit)));
    }

    return {
      backend: this.activeDevice,
      modelId: this.activeModelId,
      latencyMs,
      tokenIds: ids,
      tokens,
      vocabSize,
      vocabEntropyBits: entropyBits,
      candidates,
    };
  }
}
