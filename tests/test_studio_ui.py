"""Source-selection regressions for the guided Studio interface."""

from pathlib import Path

import gradio as gr
import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

import app


def test_preview_does_not_mix_separate_rasters_or_change_first_image_proportions(tmp_path):
    paths = []
    for name, shape in (("first.tif", (16, 64)), ("second.tif", (64, 16))):
        path = tmp_path / name
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1],
                           count=1, dtype="float32", transform=from_origin(100, 30, .01, .01)) as src:
            src.write(values, 1)
        paths.append(str(path))

    preview = app.raster_preview(paths)
    assert preview.shape == (16, 64, 3)
    np.testing.assert_array_equal(preview, app.raster_preview(paths[:1]))


def test_preview_uses_rgb_metadata_from_one_multiband_image(tmp_path):
    path = tmp_path / "rgb.tif"
    base = np.arange(96, dtype=np.uint8).reshape(8, 12)
    channels = [base, np.flipud(base), np.fliplr(base)]
    with rasterio.open(path, "w", driver="GTiff", height=8, width=12, count=3,
                       dtype="uint8", transform=from_origin(100, 30, .01, .01)) as src:
        src.write(np.stack(channels))
        src.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)

    expected = np.stack([app._normalize_preview_band(band) for band in channels], axis=-1)
    np.testing.assert_array_equal(app.raster_preview([str(path)]), expected)


@pytest.fixture
def built_in_case():
    name, case = next(
        (name, case) for name, case in app.EXAMPLES.items() if case["directory"]
    )
    paths = sorted((app.ROOT / "examples" / case["directory"]).glob("*.tif"))
    assert paths, "The guided case requires its bundled imagery."
    return name, paths


def test_loading_case_clears_upload_and_matches_the_case_that_will_run(
    built_in_case, tmp_path
):
    name, bundled_paths = built_in_case
    expected_query, expected_preview = app.load_example(name)

    query, preview, uploaded_files = app.load_example_for_ui(name)
    paths, source = app.prepare_rasters(uploaded_files, tmp_path, name)

    assert query == expected_query
    np.testing.assert_array_equal(preview, expected_preview)
    assert uploaded_files is None
    assert paths == [str(path.resolve()) for path in bundled_paths]
    assert name in source


def test_upload_replaces_the_selected_case_and_previews_the_executed_file(
    built_in_case, tmp_path
):
    name, bundled_paths = built_in_case
    uploaded_files = [str(bundled_paths[0])]

    preview, selected_case = app.select_uploaded_source(uploaded_files, name)
    paths, source = app.prepare_rasters(
        uploaded_files, tmp_path, selected_case["value"]
    )

    assert selected_case["value"] is None
    np.testing.assert_array_equal(preview, app.raster_preview(uploaded_files))
    assert paths == [str(Path(uploaded_files[0]).resolve())]
    assert source == "visitor-uploaded remote-sensing imagery"


@pytest.mark.parametrize("cleared_files", [None, []])
def test_clearing_upload_preserves_an_explicitly_selected_case_preview(
    built_in_case, cleared_files
):
    name, _ = built_in_case

    preview, selected_case = app.select_uploaded_source(cleared_files, name)

    np.testing.assert_array_equal(preview, app.load_example(name)[1])
    assert selected_case == gr.skip()


@pytest.mark.parametrize("cleared_files", [None, []])
def test_clearing_own_files_never_silently_selects_a_default_case(
    cleared_files, monkeypatch
):
    preview, selected_case = app.select_uploaded_source(cleared_files, None)

    assert preview is None
    assert selected_case == gr.skip()
    monkeypatch.setattr(app, "validate_base_url", lambda value: value)
    with pytest.raises(Exception, match="Attach GeoTIFF"):
        app._initialize_interactive_workflow(
            "Inspect my imagery.", None, cleared_files,
            app.DEFAULT_BASE_URL, "test-model", "private-test-key",
        )


def test_case_then_upload_then_clear_leaves_an_empty_source(built_in_case):
    name, paths = built_in_case
    _, _, cleared_upload = app.load_example_for_ui(name)
    assert cleared_upload is None

    _, selected_case = app.select_uploaded_source([str(paths[0])], name)
    preview, unchanged_case = app.select_uploaded_source([], selected_case["value"])

    assert selected_case["value"] is None
    assert preview is None
    assert unchanged_case == gr.skip()


def test_source_controls_use_handlers_that_keep_preview_and_input_in_sync():
    dependencies = app.demo.get_config_file()["dependencies"]
    case_events = [
        event for event in dependencies
        if app.demo.fns[event["id"]].fn is app.load_example_for_ui
    ]
    upload_events = [
        event for event in dependencies
        if app.demo.fns[event["id"]].fn is app.select_uploaded_source
    ]

    assert any(
        event["inputs"] == [app.example_input._id]
        and event["outputs"] == [
            app.query_input._id, app.raster_preview_output._id, app.raster_input._id
        ]
        for event in case_events
    )
    assert {
        (app.raster_input._id, "upload"),
        (app.raster_input._id, "clear"),
        (app.raster_input._id, "delete"),
    }.issubset({tuple(target) for event in upload_events for target in event["targets"]})
    for event in upload_events:
        assert event["inputs"] == [app.raster_input._id, app.example_input._id]
        assert event["outputs"] == [app.raster_preview_output._id, app.example_input._id]


def test_agent_trace_exposes_parameters_and_observations_as_escaped_text():
    state = {
        "attempts": [{
            "attempt": 1,
            "tool_plan": [{"tool": "inspect_rasters"}],
            "trajectory": [{
                "step": 1,
                "tool": "inspect_rasters",
                "status": "success",
                "parameters": {
                    "paths": ["/tmp/scene.tif"],
                    "note": "<script>alert(1)</script>",
                },
                "observation": '<img src=x onerror="alert(2)"> Raster is readable.',
            }],
        }],
    }

    rendered = app._attempts_interactive_html(state)

    assert "Inputs" in rendered
    assert "Observation" in rendered
    assert "/tmp/scene.tif" in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "&lt;img src=x onerror=&quot;alert(2)&quot;&gt;" in rendered
    assert "Raster is readable." in rendered
    assert "<script" not in rendered
    assert "<img" not in rendered
