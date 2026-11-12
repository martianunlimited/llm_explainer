"""Thin Typer CLI wrapper around core LLM trace extraction and sampling services."""

import logging

import typer

from src.config import settings
from src.core.extractor_service import TraceExtractorService
from src.core.sampling_math import apply_sampling_pipeline
from src.schemas import SamplingConfig

app = typer.Typer(
    name="llm-explainer",
    help="Extract genuine GPT-2 transformer forward-pass traces and inspect decoding math.",
)


@app.command()
def presets() -> None:
    """List all configured prompt presets and their linked companion labs."""
    for idx, spec in enumerate(settings.preset_specs, start=1):
        typer.echo(f"{idx}. [{spec.id}] ({spec.category}) -> {spec.prompt!r}")
        typer.echo(f"   Companion: {spec.companion_lab} ({spec.companion_url})")


@app.command()
def extract(
    model_id: str = typer.Option(
        settings.model_id,
        "--model-id",
        help="HuggingFace causal LM identifier (default from LLM_EXPLAINER_MODEL_ID).",
    ),
    only_preset: str | None = typer.Option(
        None,
        "--preset",
        help="Optional single preset ID to extract instead of the full suite.",
    ),
) -> None:
    """Run real PyTorch forward passes and write data/traces.json + js/embeddedTraces.js."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runtime_settings = settings.model_copy(update={"model_id": model_id})
    selected = runtime_settings.preset_specs
    if only_preset:
        selected = [p for p in runtime_settings.preset_specs if p.id == only_preset]
        if not selected:
            raise typer.BadParameter(f"Unknown preset id: {only_preset}")

    service = TraceExtractorService(runtime_settings)
    dataset = service.export_presets(selected)
    typer.echo(
        f"Exported {len(dataset.presets)} real traces ({dataset.model_id}) to "
        f"{runtime_settings.resolved_json_path()}"
    )


@app.command()
def inspect(prompt: str = typer.Argument(..., help="Prompt to inspect end-to-end.")) -> None:
    """Inspect a custom prompt's tokenization, final layer top-5 predictions, and entropy."""
    service = TraceExtractorService(settings)
    spec = settings.preset_specs[0].model_copy(
        update={
            "id": "custom-cli",
            "title": "CLI Prompt",
            "category": "Ad-hoc",
            "prompt": prompt,
            "continuation_steps": 1,
        }
    )
    trace = service.extract_preset_trace(spec)
    typer.echo(f"Model: {trace.architecture.model_id} | Tokens ({len(trace.tokens)}):")
    typer.echo("  " + " | ".join(f"{t.display_text}[{t.token_id}]" for t in trace.tokens))
    typer.echo(f"Final Entropy: {trace.final_vocab_entropy_bits:.3f} bits | Top predictions:")
    for cand in trace.final_candidates[:5]:
        typer.echo(
            f"  {cand.display_token:<14} logit={cand.logit:7.3f}  prob={cand.prob * 100:5.2f}%"
        )


@app.command()
def sample(
    prompt: str = typer.Argument(
        ..., help="Prompt to decode under custom sampling hyperparameters."
    ),
    temperature: float = typer.Option(0.8, min=0.0, max=3.0),
    top_k: int = typer.Option(40, min=0, max=50000),
    top_p: float = typer.Option(0.9, min=0.05, max=1.0),
    repetition_penalty: float = typer.Option(1.1, min=1.0, max=2.5),
) -> None:
    """Preview how sampling hyperparameters reshape the extracted candidate distribution."""
    service = TraceExtractorService(settings)
    spec = settings.preset_specs[0].model_copy(
        update={
            "id": "custom-sample",
            "title": "CLI Sample",
            "category": "Ad-hoc",
            "prompt": prompt,
            "continuation_steps": 0,
        }
    )
    trace = service.extract_preset_trace(spec)
    seen_ids = {tok.token_id for tok in trace.tokens}
    cfg = SamplingConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    res = apply_sampling_pipeline(trace.final_candidates, cfg, seen_token_ids=seen_ids)
    typer.echo(
        f"Surviving={res.surviving_count} | Entropy={res.entropy_bits:.3f} bits | "
        f"Clipped mass={res.clipped_mass * 100:.2f}%"
    )
    for cand in res.filtered_candidates[:8]:
        if cand.prob > 0.0:
            typer.echo(
                f"  {cand.display_token:<14} adj_logit={cand.logit:7.3f} p={cand.prob * 100:5.2f}%"
            )


if __name__ == "__main__":
    app()
