/**
 * Main entry point initializing dual tabs, trace dataset loading, and KaTeX math rendering.
 */

import { ArchExplainerController } from "./archExplainer.js";
import { LiveStudioController } from "./liveStudio.js";

async function loadTraceDataset() {
  try {
    const response = await fetch("data/traces.json", { cache: "no-store" });
    if (response.ok) {
      return await response.json();
    }
  } catch {
    // file:// fallback handled cleanly via embedded window.__EMBEDDED_LLM_TRACES__
  }
  if (window.__EMBEDDED_LLM_TRACES__) {
    return window.__EMBEDDED_LLM_TRACES__;
  }
  throw new Error("Could not load traces.json or embedded fallback trace bundle.");
}

function setupModeTabs() {
  const tabExplainer = document.getElementById("tab-explainer");
  const tabLive = document.getElementById("tab-live");
  const panelExplainer = document.getElementById("panel-explainer");
  const panelLive = document.getElementById("panel-live");

  const switchTo = (mode) => {
    const isExplainer = mode === "explainer";
    tabExplainer.classList.toggle("active", isExplainer);
    tabLive.classList.toggle("active", !isExplainer);
    tabExplainer.setAttribute("aria-selected", String(isExplainer));
    tabLive.setAttribute("aria-selected", String(!isExplainer));
    panelExplainer.classList.toggle("active", isExplainer);
    panelLive.classList.toggle("active", !isExplainer);
  };

  tabExplainer?.addEventListener("click", () => switchTo("explainer"));
  tabLive?.addEventListener("click", () => switchTo("live"));
}

function renderMathWithKaTeX() {
  if (typeof window.renderMathInElement === "function") {
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }
}

async function bootstrap() {
  setupModeTabs();
  const liveStudio = new LiveStudioController();
  liveStudio.init();

  try {
    const dataset = await loadTraceDataset();
    const explainer = new ArchExplainerController(dataset);
    explainer.init();
  } catch (err) {
    const note = document.getElementById("preset-note");
    if (note) {
      note.classList.add("warn");
      note.textContent = `Trace dataset unavailable: ${err.message}`;
    }
  }

  window.addEventListener("load", renderMathWithKaTeX);
  renderMathWithKaTeX();
}

bootstrap();
