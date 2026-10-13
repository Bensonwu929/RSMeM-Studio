from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tools import code_executor as executor


def test_successful_run_reports_streams_and_exit_code(tmp_path):
    result = executor.run_python(
        "import sys\nprint('to stdout')\nprint('to stderr', file=sys.stderr)\n",
        tmp_path,
    )

    assert result.exit_code == 0
    assert result.stdout == "to stdout\n"
    assert "to stderr" in result.stderr
    assert result.timed_out is False
    assert Path(result.script_path).is_file()


def test_failing_code_returns_normally_with_full_traceback(tmp_path):
    """A crashing script is a reported outcome, not a tool failure.

    Every surveyed harness reports the failing exit code as ordinary output.
    Raising here would turn each normal debugging iteration into a trajectory
    step the deterministic evaluator treats as a real execution error.
    """
    result = executor.run_python("raise KeyError('missing band')", tmp_path)

    assert result.exit_code != 0
    assert result.timed_out is False
    assert "KeyError" in result.stderr
    assert "missing band" in result.stderr


def test_scripts_and_logs_live_in_the_hidden_scratch_directory(tmp_path):
    result = executor.run_python("print('x')", tmp_path)

    scratch = tmp_path / executor.CODE_SUBDIR
    assert scratch.is_dir()
    assert Path(result.script_path).parent == scratch
    assert Path(result.stdout_path).parent == scratch
    # The harness excludes dot-directories from the artifact listing, so these
    # never reach the user's download list.
    assert executor.CODE_SUBDIR.startswith(".")


def test_runs_are_numbered_so_the_audit_trail_is_readable(tmp_path):
    first = executor.run_python("print(1)", tmp_path)
    second = executor.run_python("print(2)", tmp_path)

    assert Path(first.script_path).name == "step_001.py"
    assert Path(second.script_path).name == "step_002.py"


def test_empty_code_is_rejected(tmp_path):
    with pytest.raises(executor.CodeExecutionError):
        executor.run_python("   \n  ", tmp_path)


def test_missing_workspace_is_rejected(tmp_path):
    with pytest.raises(executor.CodeExecutionError):
        executor.run_python("print(1)", tmp_path / "absent")


# --- timeout behaviour -------------------------------------------------------


def test_timeout_preserves_output_collected_before_the_kill(tmp_path):
    """LangChain's shell middleware discards everything on timeout.

    Losing the partial output leaves the model unable to tell how far its code
    got, so it rewrites blindly. Unbuffered output plus file capture keeps it.
    """
    result = executor.run_python(
        "print('checkpoint reached')\nimport time\ntime.sleep(30)\n",
        tmp_path,
        timeout_s=2,
    )

    assert result.timed_out is True
    assert "checkpoint reached" in result.stdout
    assert "exceeded the 2s limit" in result.stderr


def test_timeout_is_clamped_to_the_server_ceiling(tmp_path):
    result = executor.run_python("print('ok')", tmp_path, timeout_s=10**9)

    assert result.exit_code == 0


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX only")
def test_timeout_kills_backgrounded_grandchildren(tmp_path):
    """Killing only the direct child leaks the process tree.

    Both deepagents and LangChain still have this hole: a backgrounded
    grandchild outlives the timeout and keeps writing. The marker file must
    never appear.
    """
    marker = tmp_path / "grandchild_survived.txt"
    code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', "
        f"\"import time; time.sleep(3); open({str(marker)!r}, 'w').write('alive')\"])\n"
        "time.sleep(30)\n"
    )

    result = executor.run_python(code, tmp_path, timeout_s=2)
    assert result.timed_out is True

    time.sleep(5)
    assert not marker.exists(), "a backgrounded grandchild outlived the timeout"


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX only")
def test_clean_exit_also_reaps_backgrounded_grandchildren(tmp_path):
    """The graceful path needs the same cleanup as the timeout path.

    LangChain only kills the group when a command times out, so a script that
    exits normally after backgrounding work leaves that work running.
    """
    marker = tmp_path / "survived_clean_exit.txt"
    code = (
        "import subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', "
        f"\"import time; time.sleep(3); open({str(marker)!r}, 'w').write('alive')\"])\n"
        "print('parent done')\n"
    )

    result = executor.run_python(code, tmp_path)
    assert result.exit_code == 0

    time.sleep(5)
    assert not marker.exists(), "a backgrounded grandchild outlived a clean exit"


# --- output governance -------------------------------------------------------


def test_long_stdout_keeps_the_head_and_points_at_the_full_log(tmp_path):
    result = executor.run_python(
        f"print('A' * {executor.MAX_STDOUT_BYTES * 3})", tmp_path
    )

    assert result.stdout_truncated is True
    assert result.stdout.startswith("AAA")
    assert "read_text_file" in result.stdout
    assert Path(result.stdout_path).stat().st_size > executor.MAX_STDOUT_BYTES


def test_long_stderr_keeps_the_tail_so_the_exception_survives(tmp_path):
    """A Python traceback ends with the line that matters.

    Head-biased truncation, which is what deepagents and LangChain both do,
    would cut off the exception type entirely.
    """
    code = (
        "import sys\n"
        f"sys.stderr.write('noise\\n' * {executor.MAX_STDERR_BYTES})\n"
        "raise ValueError('the actual problem')\n"
    )
    result = executor.run_python(code, tmp_path)

    assert result.stderr_truncated is True
    assert "ValueError: the actual problem" in result.stderr


def test_a_single_enormous_line_is_still_capped(tmp_path):
    """Line-based caps do not bound memory or context.

    LangChain's 100-line cap lets a single two-million-character line through
    untouched, because a line-oriented reader never yields until the newline.
    """
    result = executor.run_python(f"print('y' * {2_000_000}, end='')", tmp_path)

    assert result.stdout_truncated is True
    assert len(result.stdout) < executor.MAX_STDOUT_BYTES * 2


def test_result_dict_only_advertises_paths_when_truncated(tmp_path):
    small = executor.run_python("print('tiny')", tmp_path).as_dict()
    assert "stdout_path" not in small
    assert small["truncated"] is False

    large = executor.run_python(
        f"print('B' * {executor.MAX_STDOUT_BYTES * 3})", tmp_path
    ).as_dict()
    assert large["truncated"] is True
    assert Path(large["stdout_path"]).is_file()


# --- environment and resource limits ----------------------------------------


def test_credentials_never_reach_the_child(tmp_path):
    base = {
        "PATH": os.environ.get("PATH", ""),
        "RSMEM_API_KEY": "sk-should-not-leak",
        "OAUTH_CLIENT_SECRET": "shh",
        "MODELSCOPE_API_TOKEN": "tok",
        "HARMLESS": "keep-me",
    }
    env = executor.child_environment(executor.ResourceLimits(), tmp_path, base=base)

    assert "RSMEM_API_KEY" not in env
    assert "OAUTH_CLIENT_SECRET" not in env
    assert "MODELSCOPE_API_TOKEN" not in env
    assert env["HARMLESS"] == "keep-me"


def test_workspace_is_importable_so_written_modules_can_be_reused(tmp_path):
    (tmp_path / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")

    result = executor.run_python("import helper\nprint(helper.VALUE)", tmp_path)

    assert result.exit_code == 0
    assert result.stdout.strip() == "7"


def test_relative_paths_resolve_inside_the_attempt_directory(tmp_path):
    result = executor.run_python(
        "open('derived.txt', 'w').write('written')", tmp_path
    )

    assert result.exit_code == 0
    assert (tmp_path / "derived.txt").read_text(encoding="utf-8") == "written"


@pytest.fixture
def restore_rlimits():
    """Undo any limit the test applies to this process.

    ``apply_limits`` normally runs inside ``preexec_fn`` in a child that is
    about to be replaced. Calling it here would otherwise cap the test runner
    itself for the rest of the session.
    """
    if executor.resource is None:  # pragma: no cover - Windows
        yield
        return
    names = ["RLIMIT_CPU", "RLIMIT_AS", "RLIMIT_FSIZE", "RLIMIT_NPROC"]
    saved = {}
    for name in names:
        constant = getattr(executor.resource, name, None)
        if constant is not None:
            saved[constant] = executor.resource.getrlimit(constant)
    try:
        yield
    finally:
        for constant, limit in saved.items():
            try:
                executor.resource.setrlimit(constant, limit)
            except (ValueError, OSError):  # pragma: no cover - defensive
                pass


def test_apply_limits_never_raises_on_a_rejecting_platform(restore_rlimits):
    """macOS refuses RLIMIT_AS with EINVAL.

    An exception escaping preexec_fn aborts the spawn, which is why LangChain
    cannot start a session at all once a memory cap is configured.
    """
    executor.apply_limits(executor.ResourceLimits(memory_bytes=64 * 1024 * 1024))


def test_probe_reports_platform_support_without_touching_the_caller():
    before = None
    if executor.resource is not None:
        before = executor.resource.getrlimit(executor.resource.RLIMIT_CPU)

    applied, rejected = executor.probe_limits()

    assert set(applied) | set(rejected)
    if executor.resource is not None:
        assert executor.resource.getrlimit(executor.resource.RLIMIT_CPU) == before


@pytest.mark.skipif(os.name != "posix", reason="rlimits are POSIX only")
def test_cpu_limit_stops_a_tight_loop(tmp_path):
    limits = executor.ResourceLimits(cpu_seconds=1, memory_bytes=None)
    result = executor.run_python(
        "while True:\n    pass\n", tmp_path, timeout_s=60, limits=limits
    )

    assert result.timed_out is False, "the CPU limit should fire before the timeout"
    assert result.exit_code != 0


@pytest.mark.skipif(os.name != "posix", reason="rlimits are POSIX only")
def test_file_size_limit_bounds_one_run(tmp_path):
    limits = executor.ResourceLimits(file_size_bytes=64 * 1024, memory_bytes=None)
    result = executor.run_python(
        "open('big.bin', 'wb').write(b'0' * (5 * 1024 * 1024))",
        tmp_path,
        limits=limits,
    )

    assert result.exit_code != 0
