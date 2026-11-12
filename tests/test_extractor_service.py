"""Offline unit tests for TraceExtractorService using a tiny local PyTorch GPT-2 model."""

import json
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.config import AppSettings, PresetSpec
from src.core.extractor_service import TraceExtractorService, format_display_token
from src.schemas import TraceDataset


class StubTokenizer:
    """Deterministic lightweight tokenizer stub compatible with GPT-2 trace extraction."""

    def __init__(self, vocab_size: int = 128) -> None:
        self._vocab = [f"tok{i}" for i in range(vocab_size)]
        self._vocab[1] = " The"
        self._vocab[2] = " capital"
        self._vocab[3] = " of"
        self._vocab[4] = " France"
        self._vocab[5] = " is"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        words = text.strip().split()
        return [((hash(w) % 60) + 10) for w in words] or [1]

    def decode(self, token_ids: list[int]) -> str:
        idx = token_ids[0] % len(self._vocab)
        return self._vocab[idx]


def test_format_display_token_renders_spaces_and_newlines() -> None:
    """Whitespace control markers make leading spaces and newlines visible."""
    assert format_display_token(" Paris") == "·Paris"
    assert format_display_token("\nHello") == "↵Hello"
    assert format_display_token("world") == "world"


def test_extract_trace_from_tiny_gpt2_produces_causal_tensors(tmp_path: Path) -> None:
    """Extractor captures real WTE+WPE slices, causal attention, MLP spikes, and Logit Lens."""
    torch.manual_seed(42)
    gpt_cfg = GPT2Config(
        vocab_size=128,
        n_positions=32,
        n_embd=40,
        n_layer=2,
        n_head=4,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(gpt_cfg)
    model.eval()
    tokenizer = StubTokenizer(vocab_size=128)

    settings = AppSettings(
        project_root=tmp_path,
        model_id="tiny-test-gpt2",
        embedding_slice_dims=16,
        top_mlp_neurons=4,
        top_logit_lens=3,
        top_final_candidates=6,
    )
    preset = PresetSpec(
        id="test-preset",
        title="Test Preset",
        category="Test",
        prompt="The capital of France is",
        companion_lab="Sampling Lab",
        companion_url="https://example.com/sampling",
        highlight_note="Testing note",
        continuation_steps=2,
    )

    service = TraceExtractorService(settings=settings, model=model, tokenizer=tokenizer)
    trace = service.extract_preset_trace(preset)

    n_toks = len(trace.tokens)
    assert n_toks == 5
    assert trace.architecture.n_layers == 2
    assert trace.architecture.n_heads == 4
    assert trace.embeddings.slice_dims == 16
    assert len(trace.embeddings.wte_slices) == n_toks

    # Check causal masking and row normalization on all heads
    for layer in trace.layers:
        assert len(layer.heads) == 4
        assert len(layer.top_mlp_neurons) == 4
        assert len(layer.logit_lens) == 3
        for head in layer.heads:
            assert len(head.matrix) == n_toks
            for q_idx, row in enumerate(head.matrix):
                assert sum(row) == pytest.approx(1.0, abs=0.02)
                for k_idx in range(q_idx + 1, n_toks):
                    assert row[k_idx] == 0.0

    assert len(trace.final_candidates) == 6
    assert len(trace.continuation_steps) == 2

    # Test export writes valid JSON and JS fallback
    dataset = service.export_presets([preset])
    assert isinstance(dataset, TraceDataset)
    json_file = settings.resolved_json_path()
    js_file = settings.resolved_js_path()
    assert json_file.exists()
    assert js_file.exists()

    loaded = json.loads(json_file.read_text(encoding="utf-8"))
    assert loaded["model_id"] == "tiny-test-gpt2"
    assert len(loaded["presets"]) == 1
    assert "window.__EMBEDDED_LLM_TRACES__" in js_file.read_text(encoding="utf-8")
