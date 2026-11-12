"""Real PyTorch transformer forward-pass trace extraction service."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import AppSettings, PresetSpec
from src.schemas import (
    AutoregressiveStep,
    EmbeddingSnapshot,
    HeadAttention,
    LayerSnapshot,
    LLMTrace,
    LogitCandidate,
    ModelArchitectureInfo,
    NeuronActivation,
    TokenInfo,
    TraceDataset,
)

logger = logging.getLogger(__name__)


def format_display_token(raw_token: str) -> str:
    """Make whitespace, tabs, and newlines visually explicit in token pills."""
    return (
        raw_token.replace("\n", "↵").replace("\r", "").replace("\t", "⇥").replace(" ", "·")
    ) or "∅"


def _round_list(values: list[float], precision: int) -> list[float]:
    return [round(float(v), precision) for v in values]


def _round_matrix(matrix: torch.Tensor, precision: int) -> list[list[float]]:
    return [[round(float(cell), precision) for cell in row] for row in matrix.tolist()]


def _cosine_similarity_matrix(vectors: torch.Tensor, precision: int) -> list[list[float]]:
    normalized = F.normalize(vectors.float(), p=2, dim=-1, eps=1e-8)
    sim = torch.matmul(normalized, normalized.transpose(0, 1))
    return _round_matrix(sim, precision)


def _compute_tensor_entropy_bits(probs: torch.Tensor) -> float:
    clipped = probs.clamp(min=1e-12)
    entropy = -(probs * torch.log2(clipped)).sum().item()
    return max(0.0, float(entropy))


class TraceExtractorService:
    """Extracts complete per-layer forward pass tensors from causal GPT-2 models."""

    def __init__(
        self,
        settings: AppSettings,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.settings = settings
        self._model = model
        self._tokenizer = tokenizer

    def ensure_loaded(self) -> tuple[Any, Any]:
        """Lazily load PyTorch model (eager attention) and matching fast tokenizer."""
        if self._tokenizer is None:
            logger.info("Loading tokenizer %s", self.settings.model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id)
        if self._model is None:
            logger.info("Loading model %s on %s", self.settings.model_id, self.settings.device)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_id,
                attn_implementation="eager",
            )
            self._model.to(self.settings.device)
            self._model.eval()
        return self._model, self._tokenizer

    def _decode_token_info(self, tokenizer: Any, token_ids: list[int]) -> list[TokenInfo]:
        tokens: list[TokenInfo] = []
        for pos, tid in enumerate(token_ids):
            text = tokenizer.decode([tid])
            tokens.append(
                TokenInfo(
                    position=pos,
                    token_id=int(tid),
                    text=text,
                    display_text=format_display_token(text),
                    utf8_bytes=list(text.encode("utf-8", errors="replace")),
                )
            )
        return tokens

    def _extract_logits_candidates(
        self,
        logits_1d: torch.Tensor,
        tokenizer: Any,
        top_k: int,
    ) -> tuple[list[LogitCandidate], float]:
        precision = self.settings.float_precision
        probs = torch.softmax(logits_1d.float(), dim=-1)
        entropy_bits = round(_compute_tensor_entropy_bits(probs), precision)

        limit = min(top_k, int(logits_1d.shape[-1]))
        top_vals, top_ids = torch.topk(logits_1d.float(), k=limit)
        candidates: list[LogitCandidate] = []
        for logit_val, tid in zip(top_vals.tolist(), top_ids.tolist(), strict=True):
            token_text = tokenizer.decode([int(tid)])
            candidates.append(
                LogitCandidate(
                    token=token_text,
                    display_token=format_display_token(token_text),
                    token_id=int(tid),
                    logit=round(float(logit_val), precision),
                    prob=round(float(probs[int(tid)].item()), precision),
                )
            )
        return candidates, entropy_bits

    def _extract_embedding_snapshot(
        self,
        model: Any,
        input_ids: torch.Tensor,
        hidden_initial: torch.Tensor,
    ) -> EmbeddingSnapshot:
        precision = self.settings.float_precision
        seq_len = int(input_ids.shape[1])
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        wte = model.transformer.wte(input_ids).squeeze(0).detach().cpu()
        wpe = model.transformer.wpe(pos_ids).squeeze(0).detach().cpu()
        combined = hidden_initial.squeeze(0).detach().cpu()

        hidden_size = int(wte.shape[-1])
        slice_dims = min(self.settings.embedding_slice_dims, hidden_size)

        return EmbeddingSnapshot(
            hidden_size=hidden_size,
            slice_dims=slice_dims,
            wte_slices=_round_matrix(wte[:, :slice_dims], precision),
            wpe_slices=_round_matrix(wpe[:, :slice_dims], precision),
            combined_slices=_round_matrix(combined[:, :slice_dims], precision),
            wte_norms=_round_list(torch.linalg.vector_norm(wte, dim=-1).tolist(), precision),
            wpe_norms=_round_list(torch.linalg.vector_norm(wpe, dim=-1).tolist(), precision),
            combined_norms=_round_list(
                torch.linalg.vector_norm(combined, dim=-1).tolist(), precision
            ),
            wte_cosine_matrix=_cosine_similarity_matrix(wte, precision),
            combined_cosine_matrix=_cosine_similarity_matrix(combined, precision),
        )

    def _build_layer_snapshots(
        self,
        model: Any,
        tokenizer: Any,
        outputs: Any,
        hook_cache: dict[str, dict[int, Any]],
    ) -> list[LayerSnapshot]:
        precision = self.settings.float_precision
        layers: list[LayerSnapshot] = []
        n_layers = len(outputs.attentions)

        for layer_idx in range(n_layers):
            attn_tensor = outputs.attentions[layer_idx].squeeze(0).detach().cpu()
            post_hidden = outputs.hidden_states[layer_idx + 1].squeeze(0).detach().cpu()

            heads: list[HeadAttention] = []
            for head_idx in range(int(attn_tensor.shape[0])):
                mat = attn_tensor[head_idx]
                last_row = mat[-1]
                heads.append(
                    HeadAttention(
                        head_idx=head_idx,
                        matrix=_round_matrix(mat, precision),
                        last_token_entropy=round(_compute_tensor_entropy_bits(last_row), precision),
                        last_token_top_key_idx=int(torch.argmax(last_row).item()),
                    )
                )

            # MLP top neurons at the final token position
            pre_act, post_act = hook_cache["mlp_act"][layer_idx]
            last_pre = pre_act[-1]
            last_post = post_act[-1]
            top_n = min(self.settings.top_mlp_neurons, int(last_post.shape[-1]))
            top_indices = torch.topk(last_post.abs(), k=top_n).indices.tolist()
            top_neurons = [
                NeuronActivation(
                    neuron_idx=int(n_idx),
                    activation=round(float(last_post[n_idx].item()), precision),
                    pre_activation=round(float(last_pre[n_idx].item()), precision),
                )
                for n_idx in top_indices
            ]

            # Logit Lens: apply final ln_f + lm_head to this intermediate block's residual state
            last_hidden = outputs.hidden_states[layer_idx + 1][:, -1, :]
            lens_logits = model.lm_head(model.transformer.ln_f(last_hidden)).squeeze(0).cpu()
            logit_lens, _ = self._extract_logits_candidates(
                lens_logits,
                tokenizer,
                self.settings.top_logit_lens,
            )

            attn_delta = hook_cache["attn_delta"][layer_idx]
            mlp_delta = hook_cache["mlp_delta"][layer_idx]

            layers.append(
                LayerSnapshot(
                    layer_idx=layer_idx,
                    residual_norms=_round_list(
                        torch.linalg.vector_norm(post_hidden, dim=-1).tolist(), precision
                    ),
                    attn_delta_norm=round(
                        float(torch.linalg.vector_norm(attn_delta, dim=-1).mean().item()),
                        precision,
                    ),
                    mlp_delta_norm=round(
                        float(torch.linalg.vector_norm(mlp_delta, dim=-1).mean().item()),
                        precision,
                    ),
                    heads=heads,
                    top_mlp_neurons=top_neurons,
                    logit_lens=logit_lens,
                )
            )

        return layers

    def _run_greedy_continuations(
        self,
        model: Any,
        tokenizer: Any,
        initial_ids: list[int],
        initial_logits: torch.Tensor,
        steps: int,
    ) -> list[AutoregressiveStep]:
        continuations: list[AutoregressiveStep] = []
        running_ids = list(initial_ids)
        next_logits = initial_logits

        for step_idx in range(1, steps + 1):
            next_id = int(torch.argmax(next_logits).item())
            running_ids.append(next_id)
            token_text = tokenizer.decode([next_id])
            appended = TokenInfo(
                position=len(running_ids) - 1,
                token_id=next_id,
                text=token_text,
                display_text=format_display_token(token_text),
                utf8_bytes=list(token_text.encode("utf-8", errors="replace")),
            )

            seq_tensor = torch.tensor([running_ids], device=self.settings.device, dtype=torch.long)
            with torch.no_grad():
                step_out = model(seq_tensor)
            next_logits = step_out.logits[0, -1, :].detach().cpu()
            cands, entropy = self._extract_logits_candidates(
                next_logits,
                tokenizer,
                self.settings.top_final_candidates,
            )
            continuations.append(
                AutoregressiveStep(
                    step_index=step_idx,
                    appended_token=appended,
                    top_candidates=cands,
                    vocab_entropy_bits=entropy,
                )
            )

        return continuations

    def extract_preset_trace(self, preset: PresetSpec) -> LLMTrace:
        """Execute a real GPT-2 forward pass with block hooks and build full trace."""
        model, tokenizer = self.ensure_loaded()
        token_ids = tokenizer.encode(preset.prompt, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Prompt {preset.id!r} produced zero tokens.")

        input_tensor = torch.tensor([token_ids], device=self.settings.device, dtype=torch.long)
        hook_cache: dict[str, dict[int, Any]] = {
            "mlp_act": {},
            "attn_delta": {},
            "mlp_delta": {},
        }
        handles: list[Any] = []

        try:
            for layer_idx, block in enumerate(model.transformer.h):

                def _make_act_hook(idx: int):
                    def _hook(_mod: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
                        hook_cache["mlp_act"][idx] = (
                            inputs[0].squeeze(0).detach().cpu(),
                            output.squeeze(0).detach().cpu(),
                        )

                    return _hook

                def _make_attn_hook(idx: int):
                    def _hook(_mod: Any, _inputs: tuple[Any, ...], output: Any):
                        tensor = output[0] if isinstance(output, tuple) else output
                        hook_cache["attn_delta"][idx] = tensor.squeeze(0).detach().cpu()

                    return _hook

                def _make_mlp_hook(idx: int):
                    def _hook(_mod: Any, _inputs: tuple[Any, ...], output: torch.Tensor):
                        hook_cache["mlp_delta"][idx] = output.squeeze(0).detach().cpu()

                    return _hook

                handles.append(block.mlp.act.register_forward_hook(_make_act_hook(layer_idx)))
                handles.append(block.attn.register_forward_hook(_make_attn_hook(layer_idx)))
                handles.append(block.mlp.register_forward_hook(_make_mlp_hook(layer_idx)))

            with torch.no_grad():
                outputs = model(
                    input_tensor,
                    output_attentions=True,
                    output_hidden_states=True,
                )

            cfg = model.config
            arch = ModelArchitectureInfo(
                model_id=self.settings.model_id,
                vocab_size=int(cfg.vocab_size),
                n_layers=int(cfg.n_layer),
                n_heads=int(cfg.n_head),
                d_model=int(cfg.n_embd),
                d_head=int(cfg.n_embd // cfg.n_head),
                d_mlp=int(cfg.n_inner or (4 * cfg.n_embd)),
                max_position_embeddings=int(cfg.n_positions),
                activation_function=str(getattr(cfg, "activation_function", "gelu_new")),
            )

            embeddings = self._extract_embedding_snapshot(
                model,
                input_tensor,
                outputs.hidden_states[0],
            )
            layers = self._build_layer_snapshots(model, tokenizer, outputs, hook_cache)
            final_hidden = outputs.hidden_states[-1].squeeze(0).detach().cpu()
            final_cosine = _cosine_similarity_matrix(final_hidden, self.settings.float_precision)

            final_logits = outputs.logits[0, -1, :].detach().cpu()
            final_cands, final_entropy = self._extract_logits_candidates(
                final_logits,
                tokenizer,
                self.settings.top_final_candidates,
            )

            continuations = self._run_greedy_continuations(
                model,
                tokenizer,
                token_ids,
                final_logits,
                preset.continuation_steps,
            )

            return LLMTrace(
                id=preset.id,
                title=preset.title,
                category=preset.category,
                prompt=preset.prompt,
                companion_lab=preset.companion_lab,
                companion_url=preset.companion_url,
                highlight_note=preset.highlight_note,
                architecture=arch,
                tokens=self._decode_token_info(tokenizer, token_ids),
                embeddings=embeddings,
                layers=layers,
                final_cosine_matrix=final_cosine,
                final_candidates=final_cands,
                final_vocab_entropy_bits=final_entropy,
                continuation_steps=continuations,
            )
        finally:
            for handle in handles:
                handle.remove()

    def export_presets(self, presets: list[PresetSpec] | None = None) -> TraceDataset:
        """Extract all configured presets and persist JSON + embedded JS assets."""
        specs = presets if presets is not None else self.settings.preset_specs
        traces = [self.extract_preset_trace(spec) for spec in specs]
        dataset = TraceDataset(
            generated_at_utc=datetime.now(UTC).isoformat(),
            model_id=self.settings.model_id,
            presets=traces,
        )

        json_path = self.settings.resolved_json_path()
        js_path = self.settings.resolved_js_path()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        js_path.parent.mkdir(parents=True, exist_ok=True)

        compact = json.dumps(dataset.model_dump(), separators=(",", ":"))
        json_path.write_text(compact, encoding="utf-8")

        js_banner = (
            "// Auto-generated real PyTorch forward-pass traces via `python -m src.cli extract`\n"
            f"window.__EMBEDDED_LLM_TRACES__ = {compact};\n"
        )
        js_path.write_text(js_banner, encoding="utf-8")

        logger.info("Wrote %d real traces to %s and %s", len(traces), json_path, js_path)
        return dataset
