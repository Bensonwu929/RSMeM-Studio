from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import run_storage


def test_archive_root_defaults_and_empty_disables(monkeypatch):
    monkeypatch.delenv("RSMEM_RUN_ARCHIVE_ROOT", raising=False)
    assert run_storage.archive_root() == Path("/mnt/workspace/runs")
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", "")
    assert run_storage.archive_root() is None
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", "   ")
    assert run_storage.archive_root() is None


def test_new_run_dir_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", str(tmp_path / "runs"))
    archive = run_storage.new_run_dir("120000-abcd")
    assert archive is not None
    assert (archive / "inputs").is_dir()
    assert (archive / "outputs").is_dir()
    assert archive.parent.parent == tmp_path / "runs"
    assert archive.name == "120000-abcd"


def test_new_run_dir_returns_none_without_a_root(monkeypatch):
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", "")
    assert run_storage.new_run_dir("x") is None


def test_new_run_dir_returns_none_on_oserror(tmp_path, monkeypatch):
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", str(tmp_path / "runs"))

    def boom(self, *args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(Path, "mkdir", boom)
    assert run_storage.new_run_dir("x") is None


def test_archive_inputs_and_outputs_and_same_name_collision(tmp_path, monkeypatch):
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", str(tmp_path / "runs"))
    archive = run_storage.new_run_dir("r1")
    src = tmp_path / "plane.png"
    src.write_bytes(b"image-bytes")
    other = tmp_path / "more" / "plane.png"
    other.parent.mkdir()
    other.write_bytes(b"second")

    inputs = run_storage.archive_inputs(archive, [str(src)])
    assert len(inputs) == 1
    assert inputs[0].startswith(str(archive / "inputs"))
    assert Path(inputs[0]).read_bytes() == b"image-bytes"

    first = run_storage.archive_outputs(archive, [str(src)])
    second = run_storage.archive_outputs(archive, [str(other)])
    assert Path(first[0]).name == "plane.png"
    assert Path(second[0]).name == "plane_2.png"
    assert Path(first[0]).read_bytes() == b"image-bytes"
    assert Path(second[0]).read_bytes() == b"second"


def test_archive_outputs_does_not_recopy_a_file_already_in_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", str(tmp_path / "runs"))
    archive = run_storage.new_run_dir("r1")
    src = tmp_path / "out.png"
    src.write_bytes(b"once")
    first = run_storage.archive_outputs(archive, [str(src)])
    again = run_storage.archive_outputs(archive, first)
    assert again == first
    assert [path.name for path in (archive / "outputs").iterdir()] == [Path(first[0]).name]


def test_disabled_archive_returns_original_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", "")
    src = tmp_path / "a.png"
    src.write_bytes(b"x")
    assert run_storage.archive_inputs(None, [str(src)]) == [str(src)]
    assert run_storage.archive_outputs(None, [str(src)]) == [str(src)]
    assert run_storage.write_trace(None, {"a": 1}) is None


def test_copy_failure_keeps_the_local_path(tmp_path, monkeypatch):
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", str(tmp_path / "runs"))
    archive = run_storage.new_run_dir("r1")
    src = tmp_path / "a.png"
    src.write_bytes(b"x")

    def boom(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(run_storage.shutil, "copy2", boom)
    result = run_storage.archive_inputs(archive, [str(src)])
    assert result == [str(src)]


def test_write_trace_does_not_create_a_query_index(tmp_path, monkeypatch):
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", str(tmp_path / "runs"))
    archive = run_storage.new_run_dir("r1")
    path = run_storage.write_trace(archive, {"query": "find the red plane", "steps": [1]})
    assert path and path.endswith("trace.json")
    assert not (archive / "manifest.json").exists()


def test_write_trace_returns_none_on_oserror(tmp_path, monkeypatch):
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", str(tmp_path / "runs"))
    archive = run_storage.new_run_dir("r1")

    def boom(path, payload):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(run_storage, "_write_json", boom)
    assert run_storage.write_trace(archive, {"a": 1}) is None


def test_sweep_local_runs_only_deletes_aged_matching_prefixes(tmp_path):
    old = tmp_path / "rsmem_interactive_old"
    old.mkdir()
    (old / "file").write_text("x", encoding="utf-8")
    recent = tmp_path / "rsmem_interactive_recent"
    recent.mkdir()
    cpu = tmp_path / "rsmem_cpu_old"
    cpu.mkdir()
    keep = tmp_path / "rsmem_cpu_keep"
    keep.mkdir()
    other = tmp_path / "gradio_cache"
    other.mkdir()
    forbidden_name = tmp_path / "not_a_run"
    forbidden_name.mkdir()
    aged = time.time() - 9 * 3600
    for path in (old, cpu, keep, other, forbidden_name):
        os.utime(path, (aged, aged))

    deleted = run_storage.sweep_local_runs(
        max_age_s=8 * 3600,
        keep={str(keep)},
        parent=tmp_path,
    )
    assert deleted == 2
    assert not old.exists()
    assert not cpu.exists()
    assert recent.exists()
    assert keep.exists()
    assert other.exists()
    assert forbidden_name.exists()


def test_sweep_refuses_symlinks_and_rsmem_local(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "secret").write_text("no", encoding="utf-8")
    staging = tmp_path / "tmp"
    staging.mkdir()
    link = staging / "rsmem_interactive_link"
    link.symlink_to(victim)
    aged = time.time() - 9 * 3600
    os.utime(victim, (aged, aged))

    deleted = run_storage.sweep_local_runs(max_age_s=8 * 3600, parent=staging)
    assert deleted == 0
    assert (victim / "secret").is_file()

    local_root = tmp_path / "rsmem-local"
    local_root.mkdir()
    assert run_storage.sweep_local_runs(parent=local_root) == 0


def test_sanitize_trace_rewrites_archive_paths_to_filenames(tmp_path, monkeypatch):
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", str(tmp_path / "runs"))
    archive = run_storage.new_run_dir("r1")
    run_dir = tmp_path / "rsmem_interactive_abc"
    run_dir.mkdir()
    payload = {
        "run_id": "r1",
        "archive_dir": str(archive),
        "run_dir": str(run_dir),
        "checkpoint_path": str(run_dir / "ck.sqlite"),
        "raster_paths": [str(archive / "inputs" / "a.tif")],
        "attempts": [{"observation": f"Result save at {archive / 'outputs' / 'out.tif'}"}],
    }
    clean = run_storage.sanitize_trace_for_download(
        payload, archive_dir=archive, run_dir=run_dir,
    )
    dumped = json.dumps(clean)
    assert str(archive.resolve()) not in dumped
    assert str(run_dir) not in dumped
    assert "archive_dir" not in clean
    assert "run_dir" not in clean
    assert clean["run_id"] == "r1"
    assert clean["raster_paths"] == ["a.tif"]
    assert "out.tif" in dumped


def test_publish_downloads_lands_outside_the_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(run_storage, "downloads_root", lambda: tmp_path / "rsmem_downloads")
    src = tmp_path / "out.tif"
    src.write_bytes(b"abc")
    published = run_storage.publish_downloads("rid", [str(src)])
    dest = Path(published[0])
    assert dest.parent == (tmp_path / "rsmem_downloads" / "rid").resolve()
    assert dest.read_bytes() == b"abc"


def test_sweep_downloads_deletes_only_aged_entries(tmp_path, monkeypatch):
    root = tmp_path / "rsmem_downloads"
    monkeypatch.setattr(run_storage, "downloads_root", lambda: root)
    old_file = root / "old.json"
    recent_file = root / "recent.json"
    old_dir = root / "old-run"
    root.mkdir()
    old_file.write_text("{}", encoding="utf-8")
    recent_file.write_text("{}", encoding="utf-8")
    old_dir.mkdir()
    (old_dir / "a.tif").write_bytes(b"x")
    aged = time.time() - 9 * 3600
    os.utime(old_file, (aged, aged))
    os.utime(old_dir, (aged, aged))
    deleted = run_storage.sweep_downloads(max_age_s=8 * 3600)
    assert deleted == 2
    assert not old_file.exists()
    assert not old_dir.exists()
    assert recent_file.exists()


def test_runs_endpoint_is_absent_even_with_the_former_token(tmp_path, monkeypatch):
    import app as studio_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RSMEM_GPU_AUTOSTART", "0")
    monkeypatch.setenv("RSMEM_RUN_ARCHIVE_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("RSMEM_RUNS_API_TOKEN", "owner-secret")
    client = TestClient(studio_app.create_app())
    assert client.get("/rsmem/runs").status_code == 404
    assert client.get("/rsmem/runs", headers={"X-RSMeM-Runs-Token": "wrong"}).status_code == 404
    assert client.get(
        "/rsmem/runs", headers={"Authorization": "Bearer owner-secret"}
    ).status_code == 404
    response = client.get("/rsmem/runs", headers={"X-RSMeM-Runs-Token": "owner-secret"})
    assert response.status_code == 404
