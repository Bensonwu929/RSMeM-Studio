"""Presentation assets for the Studio; illustrative content never enters a run."""

import base64
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
CSS = (ASSETS / "studio.css").read_text(encoding="utf-8")
WALKTHROUGH_JS = (ASSETS / "studio.js").read_text(encoding="utf-8")
RUN_TIMER_JS = (ASSETS / "run-timer.js").read_text(encoding="utf-8")
BRAND_MARK = (ASSETS / "rsmem-mark.svg").read_text(encoding="utf-8")

# Gradio otherwise localizes these controls to the visitor's Chinese browser.
# Flat translation keys override the built-in nested messages through its I18n API.
ENGLISH_CONTROLS = {
    "upload_text.drop_file": "Drop imagery here",
    "upload_text.click_to_upload": "Browse files",
    "common.or": "or",
    "common.clear": "Clear",
    "common.download": "Download",
    "common.share": "Share",
    "common.remove": "Remove",
    "common.upload": "Add files",
    "common.empty": "Empty",
    "common.error": "Error",
    "file.uploading": "Uploading…",
    "dropdown.dropdown": "Select a case",
    "radio.radio": "Connection mode",
}


def _image(name: str) -> str:
    return "data:image/webp;base64," + base64.b64encode(
        (ASSETS / name).read_bytes()
    ).decode("ascii")


def language_script() -> str:
    from studio_i18n import TRANSLATIONS

    return "const translations = " + json.dumps(TRANSLATIONS, ensure_ascii=False) + ";\n" + (
        ASSETS / "language.js"
    ).read_text(encoding="utf-8")


NAV = f"""
<header class="studio-nav">
  <a class="wordmark" href="#" aria-label="RSMeM Studio home"><span class="brand-mark" aria-hidden="true">{BRAND_MARK}</span> RSMeM <span class="studio-tag">STUDIO</span></a>
  <span class="nav-caption">EARTH OBSERVATION · KNOWLEDGE · MEMORY</span>
  <div class="nav-actions">
    <a class="nav-source" href="https://github.com/AI9Stars/RSMeM" target="_blank" rel="noopener noreferrer">GitHub ↗</a>
    <div class="language-switch" role="group" aria-label="Interface language" data-no-translate>
      <button type="button" data-ui-language="zh" aria-label="中文" aria-pressed="false">中</button>
      <button type="button" data-ui-language="en" aria-label="English" aria-pressed="true">EN</button>
    </div>
  </div>
</header>
"""


# Only the first CHIPS_SHOWN tasks get a chip; the rest are summarized as a count.
INTERPRETATION_TASKS = (
    "Object detection",
    "Instance segmentation",
    "Change detection",
    "Object counting",
    "Scene classification",
    "Damage assessment",
    "Thematic extraction",
    "Captioning and visual QA",
    "Text recognition",
)
QUANTITATIVE_TASKS = (
    "Land surface temperature",
    "Soil moisture",
    "Water turbidity",
    "Thermal inertia",
    "Vegetation water content and LAI",
    "Sea-ice concentration",
    "Vegetation cover",
    "Precipitable water vapour",
    "Spectral indices",
    "Spatiotemporal change analysis",
)
CHIPS_SHOWN = 4


def _capability_column(index: str, title: str, tasks: tuple[str, ...]) -> str:
    chips = "".join(f"<span>{task}</span>" for task in tasks[:CHIPS_SHOWN])
    hidden = len(tasks) - CHIPS_SHOWN
    more = f'<span class="more-count">+{hidden} more</span>' if hidden > 0 else ""
    return (
        f'<div><div class="capability-title"><span class="principle-index">{index}</span>'
        f'<strong>{title}</strong></div><div class="capability-chips">{chips}{more}</div></div>'
    )


def _capability_strip() -> str:
    return f"""
<div class="capability-strip" aria-label="Supported tasks">
  <div class="capability-columns">
    {_capability_column("01", "Image interpretation", INTERPRETATION_TASKS)}
    {_capability_column("02", "Quantitative parameter estimation", QUANTITATIVE_TASKS)}
  </div>
</div>
"""


def hero_html(case_count: int) -> str:
    return f"""
<section class="studio-hero" aria-labelledby="hero-title">
  <div class="hero-copy">
    <p class="eyebrow"><span class="signal-dot"></span> KNOWLEDGE-ENHANCED REMOTE SENSING AGENT</p>
    <h1 id="hero-title">From imagery<br><span>to grounded answers.</span></h1>
    <p class="hero-description">Ask a question about your imagery. RSMeM plans with geoscience knowledge, runs real analysis tools, and brings back answers you can trace to the evidence.</p>
    <div class="hero-actions"><a class="pill primary" href="#input-card">Start an analysis <span>↗</span></a><a class="pill secondary" href="#how-it-works">See how it works <span>↓</span></a></div>
  </div>
  <div class="earth-observation" aria-label="False-color preview of real Black Forest NDVI example data">
    <div class="observation-grid"></div>
    <div class="scene-heading"><span><i class="signal-dot"></i> OBSERVATION FIELD</span><span>01 / {case_count:02d}</span></div>
    <div class="raster-plane"><img src="{_image('blackforest-ndvi.webp')}" alt="Black Forest NDVI raster in blue-green display colors" /><span class="scan-line" aria-hidden="true"></span><span class="map-crosshair" aria-hidden="true">+</span></div>
    <div class="scene-coordinate">BLACK FOREST<br><span>2021.07.29 · NDVI</span></div>
    <div class="scene-caption"><span class="scene-icon">◫</span><span>Every insight starts with an observation.</span><span class="scene-arrow">↗</span></div>
  </div>
</section>
{_capability_strip()}"""


def walkthrough_html() -> str:
    steps = [
        ("Ask a question", "Start with your query. Use a case if you need inspiration."),
        ("Connect and plan", "Let geoscience knowledge guide the tools."),
        ("Watch the analysis", "Follow tool calls, observations, and progress."),
        ("Review the evidence", "Read the answer and download available outputs."),
        ("Refine and retry", "Choose what the next attempt should learn."),
    ]
    tabs = "".join(
        f'<button type="button" class="route-step" id="route-tab-{i}" role="tab" '
        f'aria-selected="{"true" if i == 0 else "false"}" aria-controls="route-panel-{i}" '
        f'tabindex="{0 if i == 0 else -1}" data-step="{i}"><span class="route-number">0{i + 1}</span>'
        f'<span><strong>{title}</strong><small>{description}</small></span><span class="route-arrow">↗</span></button>'
        for i, (title, description) in enumerate(steps)
    )
    return f"""
<section class="walkthrough" id="how-it-works" aria-labelledby="route-title">
  <div class="section-heading"><div><p class="eyebrow">THE PATH FROM PIXELS TO ANSWERS</p><h2 id="route-title">Your first analysis, step by step.</h2></div><p>Explore each step before you run.<br>Know what to do and what to expect.</p></div>
  <div class="route-layout">
    <div class="route-tabs" role="tablist" aria-label="RSMeM walkthrough" aria-orientation="vertical">{tabs}</div>
    <div class="route-demo">
      <div class="demo-toolbar"><span><i></i><i></i><i></i></span><span>WALKTHROUGH · NOT A LIVE RUN</span><span class="route-counter" aria-live="polite">01 / 05</span></div>
      <div class="route-panel" id="route-panel-0" role="tabpanel" aria-labelledby="route-tab-0" tabindex="0">
        <p class="demo-kicker">INPUT / START WITH A QUESTION</p><h3>“Landsat 8 Band 10, Red, and NIR over Germany's Black Forest, July 2021 — how much cooler are forest pixels (NDVI &gt; 0.7) than non-vegetated pixels (NDVI &lt; 0.2)?”</h3>
        <div class="demo-images"><figure><img src="{_image('blackforest-ndvi.webp')}" alt="Black Forest NDVI example raster" /><figcaption>NDVI <span>Vegetation index</span></figcaption></figure><figure><img src="{_image('blackforest-lst.webp')}" alt="Black Forest LST example raster" /><figcaption>LST <span>Surface temperature</span></figcaption></figure></div>
        <div class="demo-outcome"><span>YOU GET</span><p>Your question and a preview of the first image. If you need a starting point, a prepared case supplies both.</p></div>
        <a class="text-action" href="#input-card">Choose your data <span>↗</span></a>
      </div>
      <div class="route-panel" id="route-panel-1" role="tabpanel" aria-labelledby="route-tab-1" tabindex="0" hidden>
        <p class="demo-kicker">KNOWLEDGE / A PLAN WITH CONTEXT</p><h3>The right analysis starts with domain knowledge.</h3>
        <div class="knowledge-tree"><span>Geoscience knowledge</span><div>Forest cooling effect <span>→</span> Vegetation + temperature <span>→</span> NDVI-masked LST contrast</div><small>HKG retrieves knowledge relevant to your task.</small></div>
        <div class="mini-flow"><span>Retrieve</span><b>→</b><span>Plan</span><b>→</b><span>Act + observe</span></div>
        <div class="demo-outcome"><span>YOUR NEXT STEP</span><p>Connect ModelScope or your own model API, then select Analyze with RSMeM. Live progress appears below.</p></div>
        <a class="text-action" href="#model-service">Connect a model <span>↗</span></a>
      </div>
      <div class="route-panel" id="route-panel-2" role="tabpanel" aria-labelledby="route-tab-2" tabindex="0" hidden>
        <p class="demo-kicker">EXECUTION / OPEN ANY STEP</p><h3>See how tools turn imagery into evidence.</h3>
        <div class="example-trace"><div><span>01</span><code>inspect_rasters</code><small>Inspect inputs</small></div><div><span>02</span><code>lst_single_channel</code><small>Derive surface temperature</small></div><div><span>03</span><code>mean_lst_by_ndvi</code><small>Compare by vegetation class</small></div></div>
        <p class="demo-note">Illustrative Black Forest workflow. The agent chooses its actual tools from your data and each observation.</p>
        <div class="demo-outcome"><span>YOU GET</span><p>An expandable Agent trace: inspect each tool’s inputs, observations, and any errors.</p></div>
        <a class="text-action" href="#run-row">Start your own run <span>↗</span></a>
      </div>
      <div class="route-panel" id="route-panel-3" role="tabpanel" aria-labelledby="route-tab-3" tabindex="0" hidden>
        <p class="demo-kicker">EVIDENCE / MORE THAN AN ANSWER</p><h3>An answer you can check. Evidence you can keep.</h3>
        <div class="deliverables"><div><span>≡</span><strong>Grounded answer</strong><small>Summary · findings supported by tool evidence</small></div><div><span>◧</span><strong>Analysis outputs</strong><small>Derived rasters and other files, when generated</small></div><div><span>⌘</span><strong>Run record</strong><small>JSON · a downloadable execution trace</small></div></div>
        <div class="demo-outcome"><span>NEXT</span><p>Review Summary and Evidence &amp; artifacts below. A statistics-only task may return numbers without creating an image.</p></div>
        <a class="text-action" href="#run-row">Go to the analysis controls <span>↗</span></a>
      </div>
      <div class="route-panel" id="route-panel-4" role="tabpanel" aria-labelledby="route-tab-4" tabindex="0" hidden>
        <p class="demo-kicker">MEMORY / A BETTER NEXT ATTEMPT</p><h3>Turn a failed attempt into useful experience.</h3>
        <div class="memory-loop"><span>Find the issue</span><b>→</b><span>Compare repairs</span><b>→</b><strong>You confirm</strong><div>↳ Save confirmed memory → Retry with new constraints ↗</div></div>
        <div class="demo-outcome"><span>YOUR NEXT STEP</span><p>Accept an upload result or request a repair. Compare three suggestions or write your own. Only the reflection you confirm enters the next attempt.</p></div>
        <p class="demo-note">HKG supplies prior knowledge. FAR refines failure experience. Up to 3 attempts for graded cases, or 10 for uploads and open-ended cases.</p>
      </div>
      <div class="route-controls"><button type="button" data-route-prev aria-label="Previous step" disabled>←</button><span>Pixels → Answers → Experience</span><button type="button" data-route-next aria-label="Next step">Next step →</button></div>
    </div>
  </div>
</section>
"""


WORKSPACE_HEADING = """
<div class="section-heading workspace-heading"><div><p class="eyebrow">YOUR EARTH OBSERVATION WORKSPACE</p><h2>What would you like to discover?</h2></div><a class="text-action" href="#how-it-works">Revisit the walkthrough ↑</a></div>
"""

FOOTER = f"""
<div class="studio-footer"><span class="footer-brand"><span class="brand-mark" aria-hidden="true">{BRAND_MARK}</span>RSMeM Studio</span><span>Grounded in knowledge. Backed by evidence. Improved through experience.</span><a href="https://github.com/AI9Stars/RSMeM" target="_blank" rel="noopener noreferrer">Explore the source ↗</a></div>
"""
