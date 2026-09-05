# RSMeM Studio interaction redesign

## Direction and implementation plan

The old page leads with nested forms before explaining the product or the next
action. Borrow the spacious typography, atmospheric blue, pill actions and
paired explanation/demo rhythm of https://www.deepseek.com/harness/en/ while
keeping RSMeM's own scientific workflow and imagery.

1. Introduce the promise: original imagery → geoscience knowledge → real tool
   execution → supported answers → user-confirmed failure memory.
2. Add a keyboard-accessible walkthrough. Each step explains the action, expected
   output and next step. Label illustrative content separately from live results.
3. Offer prepared cases as optional inspiration; retain the existing upload, authentication,
   streaming, artifact and reflection mechanisms. Remove redundant outer frames.
4. Fix source selection so the preview and the next run refer to the same data.
5. Verify existing workflow and connection regressions, source switching,
   navigation, desktop/mobile layout and reduced-motion behavior.

No new production dependencies. No provider calls or public deployment are
needed for this local frontend change. Live model/GPU output still depends on
the user's configured service; scalar tasks do not always produce image files.

## Delivered experience

The interface is now English throughout the authored content and the relevant
Gradio controls on English and Chinese browser locales. The headline is
**From pixels to grounded answers.** Main actions use short verbs: **Start an
analysis**, **See how it works**, **Choose your data**, and **Connect a model**.
Regular guide text uses 14px type, with stronger contrast for secondary copy.
Small uppercase text is reserved for metadata, never the primary instruction.

The five-step walkthrough explains input, knowledge-guided planning, execution,
evidence, and confirmed failure memory. It supports pointer and keyboard
navigation, explicitly labels its content as illustrative, and links directly
to the real controls. The live tool trace now exposes escaped input parameters
alongside observations.

Prepared-case selection clears previous uploads. Upload, individual deletion,
and clear actions refresh the preview and the selected source together. The
populated upload state reserves space for its toolbar so the first file's delete
button remains usable. Input previews retain fullscreen viewing without
unnecessary sharing controls.

## Changed files

| File | Purpose |
| --- | --- |
| `app.py` | Integrates the new presentation and source-switching handlers; preserves the original 21-output workflow contract. |
| `studio_ui.py` | English page content, walkthrough, control translations, and asset loading. |
| `assets/studio.css` | Dark blue observation theme, open layout, readable controls, responsive rules and reduced-motion support. |
| `assets/studio.js` | Local walkthrough tabs, arrow/Home/End navigation and previous/next controls. |
| `assets/sichuan-ndvi.webp`, `assets/sichuan-lst.webp` | Lightweight display assets derived from bundled Sichuan GeoTIFFs, retaining the source data's Apache-2.0 attribution in `DATA_LICENSE.md`. |
| `scripts/build_studio_previews.py` | Rebuilds the display assets with percentile scaling and documented false colors; original rasters remain untouched. |
| `tests/test_studio_ui.py` | Source-switching/event regressions and HTML escaping of tool inputs/observations. |
| `docs/UI_REDESIGN.md` | Design rationale, changed files and validation record. |

## Validation

- Before frontend edits: 70 existing workflow/connection tests passed.
- Final Python suite: **129 passed, 1 skipped**. The skipped test needs an
  external EarthBench question catalog not available on this machine.
- New Python files pass Ruff; JavaScript syntax and Python compilation pass.
  `git diff --check` passes. `app.py` has 11 pre-existing Ruff findings versus
  12 in HEAD, with no introduced lint findings. No separate type checker is
  configured in this repository.
- Browser verified: all five guide panels, previous/next boundaries, Home-key
  navigation, prepared-case loading, uploading multiple GeoTIFFs, clearing the
  selected case on upload, deleting the first uploaded file with preview change,
  clearing all uploads, switching back to a case and switching model modes.
- Layout checked at 1440px, intermediate widths, and 390px. No page-level
  horizontal overflow at the checked widths. The final visual review passes.
- No live provider/GPU inference was performed and nothing was published.
  Existing browser-session storage warnings may appear after restarting the
  development process; authentication and production deployment are unchanged.

## Follow-up: aligned imagery panels and orbital identity

The imagery controls now share grid rows, allowing both headings to grow together
when their text wraps. Both content windows have a 240px border-box height in
empty, prepared-case, and uploaded-file states. Supporting format restrictions
sit below the pair. Mobile layouts stack the two windows at the same height.

The diamond has been replaced by `assets/rsmem-mark.svg`, an editable vector mark
combining a longitude/latitude globe, an observation orbit, and linked Agent
nodes. Navigation and footer use the same asset. This follow-up changes `app.py`,
`studio_ui.py`, `assets/studio.css`, and the new SVG.

Validation: 38 workflow/presentation regression tests passed; Ruff and
`git diff --check` passed for the relevant changes. Browser measurements confirm
equal top/bottom edges and 580×240px windows at 1440px, 407.4×240px windows at
900px, and stacked 350×240px windows at 390px without horizontal overflow.

## Follow-up: query-first onboarding and language selection

The workspace now leads with **01 / Ask your question**, followed immediately
by the input. The explanatory sentence and duplicate visible Query label have
been removed; the field retains an accessible name. Prepared cases are optional
and collapsed beneath the query under **Need an idea?**

A quiet **中 / EN** control sits beside the navigation links. `studio_i18n.py`
contains the translation catalog and `assets/language.js` applies it only to
marked interface copy. Switching languages does not reload the page or change
the query, files, credentials, active guide step, model output, or evidence.
Only the language preference is stored locally. Newly rendered native controls
are localized through a scoped observer; user names and filenames are excluded.

Input previews now use the first image at its original aspect ratio, fitting
inside the aligned window. Single-band imagery is shown in grayscale; a
multiband image uses its RGB channels only when RGB metadata is explicit.
Separate files are no longer combined into an arbitrary color composite.
The original data used for analysis is unchanged.

Validation: 134 tests passed with one external-data skip after the language and
preview changes; 43 focused tests passed after the final copy simplification.
Browser checks verified bilingual switching, persistence after reload,
unchanged query/preview/guide state, optional-case loading, and an original
640×213 preview rendered at 495.8×165.0 without distortion. Relevant new Python
files pass Ruff and both JavaScript files pass syntax checks.

Push instructions for the existing remote branch are in `docs/FRONTEND_PUSH.md`.
The assistant has not committed or pushed these changes.
