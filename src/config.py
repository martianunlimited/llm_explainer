"""Application configuration injected via pydantic-settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PresetSpec(BaseSettings):
    """Specification for a curated trace preset linked to companion labs."""

    id: str
    title: str
    category: str
    prompt: str
    companion_lab: str
    companion_url: str
    highlight_note: str
    continuation_steps: int = 4


DEFAULT_PRESETS: list[dict[str, str | int]] = [
    {
        "id": "capital-france",
        "title": "Capital of France (Factual Recall & Logit Lens)",
        "category": "Factual Decoding",
        "prompt": "The capital of France is",
        "companion_lab": "LLM Sampling Lab",
        "companion_url": "https://martianunlimited.github.io/llm_sampling/",
        "highlight_note": (
            "Tracks how ' Paris' emerges in late MLP + Attention blocks and dominates"
            " the 50,257-token softmax distribution."
        ),
        "continuation_steps": 4,
    },
    {
        "id": "winograd-tired",
        "title": "Winograd Ambiguity: '...because it was too tired'",
        "category": "Coreference Attention",
        "prompt": "The trophy wouldn't fit in the brown suitcase because it was too tired",
        "companion_lab": "LLM Attention Lab",
        "companion_url": "https://martianunlimited.github.io/llm_attention/",
        "highlight_note": (
            "Observe middle-to-late attention heads routing ' it' toward animate"
            " subjects or trophy/suitcase syntax."
        ),
        "continuation_steps": 4,
    },
    {
        "id": "winograd-wide",
        "title": "Winograd Flip: '...because it was too wide'",
        "category": "Coreference Attention",
        "prompt": "The trophy wouldn't fit in the brown suitcase because it was too wide",
        "companion_lab": "LLM Attention Lab",
        "companion_url": "https://martianunlimited.github.io/llm_attention/",
        "highlight_note": (
            "Compare directly with the 'tired' trace: changing a single adjective"
            " shifts head attention weights and downstream next-token probabilities."
        ),
        "continuation_steps": 4,
    },
    {
        "id": "strawberry-bpe",
        "title": "BPE Subword Blindness: 'strawberry'",
        "category": "Tokenization Trap",
        "prompt": "How many r's are in strawberry? Count:",
        "companion_lab": "LLM Tokenizer Lab",
        "companion_url": "https://martianunlimited.github.io/llm_tokenizer/",
        "highlight_note": (
            "GPT-2's BPE tokenizer merges 'strawberry' into opaque subword IDs instead"
            " of characters, forcing the model to guess character counts."
        ),
        "continuation_steps": 4,
    },
    {
        "id": "king-woman-analogy",
        "title": "Vector Analogy: 'King is to man as queen is to'",
        "category": "Embedding Geometry",
        "prompt": "King is to man as queen is to",
        "companion_lab": "LLM Embedding Lab",
        "companion_url": "https://martianunlimited.github.io/llm_embedding/",
        "highlight_note": (
            "Watch static WTE token embeddings evolve through contextual layers to"
            " resolve the relational analogy onto ' woman'."
        ),
        "continuation_steps": 4,
    },
    {
        "id": "induction-harry",
        "title": "In-Context Induction: 'Harry Potter ... Harry'",
        "category": "Induction Heads & KV Cache",
        "prompt": "Harry Potter opened the heavy oak door. Harry",
        "companion_lab": "LLM KV-Cache & Attention",
        "companion_url": "https://martianunlimited.github.io/llm_kvcache/",
        "highlight_note": (
            "Mid-layer induction heads match the second 'Harry' against the earlier"
            " 'Harry' key, copying the cached value after it (' Potter')."
        ),
        "continuation_steps": 4,
    },
]


class AppSettings(BaseSettings):
    """Runtime settings for model extraction and artifact generation."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_EXPLAINER_",
        env_file=".env",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    model_id: str = "openai-community/gpt2"
    device: Literal["cpu", "cuda"] = "cpu"
    embedding_slice_dims: int = 32
    top_mlp_neurons: int = 8
    top_logit_lens: int = 5
    top_final_candidates: int = 24
    float_precision: int = 4
    output_json_path: Path = Field(default=Path("data/traces.json"))
    output_js_path: Path = Field(default=Path("js/embeddedTraces.js"))
    preset_specs: list[PresetSpec] = Field(
        default_factory=lambda: [PresetSpec(**item) for item in DEFAULT_PRESETS]
    )

    def resolved_json_path(self) -> Path:
        """Resolve JSON artifact path relative to project root."""
        return (
            self.output_json_path
            if self.output_json_path.is_absolute()
            else self.project_root / self.output_json_path
        )

    def resolved_js_path(self) -> Path:
        """Resolve embedded JS fallback path relative to project root."""
        return (
            self.output_js_path
            if self.output_js_path.is_absolute()
            else self.project_root / self.output_js_path
        )


settings = AppSettings()
