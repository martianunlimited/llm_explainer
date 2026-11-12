# End-to-End LLM Explainer & Live WebGPU / WASM Studio

An interactive GitHub Pages walkthrough showing the complete computational pipeline inside a causal decoder-only Transformer — connecting every stage directly to the companion labs:

- **#1 Tokenization Lab:** [`martianunlimited/llm_tokenizer`](https://github.com/martianunlimited/llm_tokenizer) ([Live Demo](https://martianunlimited.github.io/llm_tokenizer/))
- **#2 Embedding Geometry Lab:** [`martianunlimited/llm_embedding`](https://github.com/martianunlimited/llm_embedding) ([Live Demo](https://martianunlimited.github.io/llm_embedding/))
- **#3 Positional Encoding Lab:** [`martianunlimited.github.io/llm_positional`](https://martianunlimited.github.io/llm_positional/)
- **#4 Attention Mechanisms Lab:** [`martianunlimited.github.io/llm_attention`](https://martianunlimited.github.io/llm_attention/)
- **#5 Decoding & Sampling Lab:** [`martianunlimited/llm_sampling`](https://github.com/martianunlimited/llm_sampling) ([Live Demo](https://martianunlimited.github.io/llm_sampling/))
- **#6 KV-Cache Lab:** [`martianunlimited.github.io/llm_kvcache`](https://martianunlimited.github.io/llm_kvcache/)

---

## Dual Authentic Modes

### 1. End-to-End Transformer Trace (`0 MB · Instant Load`)
Visualizes **genuine PyTorch `openai-community/gpt2` forward-pass tensors** extracted via `src/core/extractor_service.py` across all 4 stages:
1. **Stage 1 · BPE Tokenization & Autoregressive Loop:** Token IDs, leading-space BPE subwords (`·strawberry` single-token blindness, `'·Harry'` -> `'·Potter'`), and 4 real continuation steps.
2. **Stage 2 · Embeddings + Positional Encoding:** Slice comparison of `wte`, `wpe[pos]`, and combined `h₀ = wte + wpe`, plus side-by-side cosine similarity evolution from **Raw WTE** → **WTE + WPE** → **Post Layer 11**.
3. **Stage 3 · 12× Decoder Blocks (12L × 12H):** Interactive layer/head scrubber with bipartite token-to-token attention arcs, strictly lower-triangular `T × T` causal attention heatmaps, top `NewGELUActivation` MLP neurons (`d_mlp = 3072`), and **Logit Lens** projections at every layer.
4. **Stage 4 · LM Head & Interactive Sampling:** Adjust Temperature, Top-K, Top-P nucleus, and Repetition Penalty over the real `50,257`-token softmax distribution and roll the continuous probability tape `[0, 1]`.

### 2. Live Browser LLM (`WebGPU / WASM ONNX`)
Runs real quantized neural network weights directly in the browser via `@huggingface/transformers@3.3.3`:
- `Xenova/distilgpt2` (~81 MB `q8`, 6L × 12H)
- `Xenova/gpt2` (~122 MB `q8`, 12L × 12H)
- `HuggingFaceTB/SmolLM2-135M-Instruct` (~112 MB `q4f16`, 30L Llama GQA + RoPE)

Enter any custom prompt, inspect live BPE token IDs, extract genuine next-token logits and full-vocabulary Shannon entropy from `await model({ input_ids, attention_mask })`, filter live probabilities with Temperature / Top-K / Top-P, and step or stream tokens client-side.

---

## Python 3.13 Trace Extractor & Verification

```bash
# 1. Inspect available cross-lab presets
python -m src.cli presets

# 2. Re-extract real PyTorch GPT-2 traces to data/traces.json & js/embeddedTraces.js
python -m src.cli extract

# 3. Inspect any custom prompt from the CLI
python -m src.cli inspect "Harry Potter opened the door. Harry"

# 4. Run quality gates
ruff check . --fix
ruff format .
pytest -q
```

---

## Deploying to GitHub Pages

This repository is pure static HTML/CSS/ES-modules + pre-extracted JSON and includes `.nojekyll` and `.github/workflows/static.yml`. Pushing to `main` deploys directly via GitHub Actions Pages without any Node build step.
