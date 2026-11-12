"""Pydantic v2 domain contracts for transformer forward-pass traces and sampling."""

from pydantic import BaseModel, Field


class SamplingConfig(BaseModel):
    """Decoding hyperparameters applied to final LM head logits."""

    temperature: float = Field(default=0.8, ge=0.0, le=3.0)
    top_k: int = Field(default=40, ge=0, le=50000)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.5)


class TokenInfo(BaseModel):
    """Single BPE token extracted from the tokenizer."""

    position: int
    token_id: int
    text: str
    display_text: str
    utf8_bytes: list[int]


class EmbeddingSnapshot(BaseModel):
    """Stage 2 embedding vectors and pairwise cosine similarities."""

    hidden_size: int
    slice_dims: int
    wte_slices: list[list[float]]
    wpe_slices: list[list[float]]
    combined_slices: list[list[float]]
    wte_norms: list[float]
    wpe_norms: list[float]
    combined_norms: list[float]
    wte_cosine_matrix: list[list[float]]
    combined_cosine_matrix: list[list[float]]


class HeadAttention(BaseModel):
    """Causal attention matrix and entropy summary for a single attention head."""

    head_idx: int
    matrix: list[list[float]]
    last_token_entropy: float
    last_token_top_key_idx: int


class NeuronActivation(BaseModel):
    """Top activated MLP neuron in a transformer block."""

    neuron_idx: int
    activation: float
    pre_activation: float


class LogitCandidate(BaseModel):
    """Candidate token predicted from logits at a layer or final LM head."""

    token: str
    display_token: str
    token_id: int
    logit: float
    prob: float


class LayerSnapshot(BaseModel):
    """Internal activations extracted from a single transformer decoder block."""

    layer_idx: int
    residual_norms: list[float]
    attn_delta_norm: float
    mlp_delta_norm: float
    heads: list[HeadAttention]
    top_mlp_neurons: list[NeuronActivation]
    logit_lens: list[LogitCandidate]


class AutoregressiveStep(BaseModel):
    """Single real greedy continuation step after the initial prompt."""

    step_index: int
    appended_token: TokenInfo
    top_candidates: list[LogitCandidate]
    vocab_entropy_bits: float


class ModelArchitectureInfo(BaseModel):
    """Metadata describing the inspected causal language model architecture."""

    model_id: str
    vocab_size: int
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int
    max_position_embeddings: int
    activation_function: str = "gelu_new"


class LLMTrace(BaseModel):
    """Complete end-to-end transformer forward pass trace for one prompt preset."""

    id: str
    title: str
    category: str
    prompt: str
    companion_lab: str
    companion_url: str
    highlight_note: str
    architecture: ModelArchitectureInfo
    tokens: list[TokenInfo]
    embeddings: EmbeddingSnapshot
    layers: list[LayerSnapshot]
    final_cosine_matrix: list[list[float]]
    final_candidates: list[LogitCandidate]
    final_vocab_entropy_bits: float
    continuation_steps: list[AutoregressiveStep] = Field(default_factory=list)


class TraceDataset(BaseModel):
    """Root bundle serialized to JSON and embedded JS for the web UI."""

    schema_version: str = "1.0.0"
    generated_at_utc: str
    model_id: str
    presets: list[LLMTrace]
