"""Static frontend integrity tests ensuring GitHub Pages assets stay cohesive."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_required_static_files_exist() -> None:
    """Verify static site deliverables, workflow, and data trace bundle exist."""
    expected = [
        ROOT / "index.html",
        ROOT / "style.css",
        ROOT / "README.md",
        ROOT / ".nojekyll",
        ROOT / ".github" / "workflows" / "static.yml",
        ROOT / "data" / "traces.json",
        ROOT / "js" / "embeddedTraces.js",
        ROOT / "js" / "mathUtils.js",
        ROOT / "js" / "archExplainer.js",
        ROOT / "js" / "liveWebGPUEngine.js",
        ROOT / "js" / "liveStudio.js",
        ROOT / "js" / "app.js",
    ]
    for path in expected:
        assert path.exists(), f"Missing static asset: {path}"


def test_index_html_links_all_companion_demos_and_dual_tabs() -> None:
    """Ensure companion lab links and both Explainer + Live WebGPU tabs exist."""
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    for companion in ("llm_tokenizer", "llm_embedding", "llm_attention", "llm_sampling"):
        assert companion in html, f"Missing link to companion repo {companion}"
    assert "tab-explainer" in html
    assert "tab-live" in html


def test_traces_json_has_valid_schema_and_presets() -> None:
    """Verify generated real GPT-2 trace bundle is non-empty and schema-consistent."""
    payload = json.loads((ROOT / "data" / "traces.json").read_text(encoding="utf-8"))
    assert payload["model_id"] == "openai-community/gpt2"
    assert len(payload["presets"]) >= 6
    for preset in payload["presets"]:
        assert len(preset["layers"]) == 12
        assert len(preset["layers"][0]["heads"]) == 12
        assert len(preset["final_candidates"]) >= 20


def test_all_js_get_element_by_id_targets_exist_in_html() -> None:
    """Ensure every DOM id queried via getElementById in static JS modules exists in index.html."""
    import re

    html = (ROOT / "index.html").read_text(encoding="utf-8")
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    js_files = ("js/app.js", "js/archExplainer.js", "js/liveStudio.js")
    referenced: set[str] = set()
    for rel in js_files:
        content = (ROOT / rel).read_text(encoding="utf-8")
        referenced.update(re.findall(r'getElementById\("([^"]+)"\)', content))
    missing = referenced - html_ids
    assert not missing, f"DOM IDs queried in JS missing from index.html: {missing}"
