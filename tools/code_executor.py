"""One-shot sandboxed Python execution for the RSMeM Studio code tools.

Every run starts a fresh interpreter inside a short-lived grandchild process.
That shape is deliberate:

* The MCP server owns a stdio JSONRPC channel, so anything the model's code
  prints on stdout would corrupt message framing and deadlock the server.
  Redirecting a grandchild's streams to files makes that impossible by
  construction rather than by discipline.
* A runaway script can only lose its own tool call.  The MCP server survives,
  so the remaining tool calls in the attempt still work.
* Resource limits and a process-group kill have somewhere to attach.

Interpreter state does not survive between runs.  Intermediate results travel
through files in the attempt directory, which is also where the ReAct harness
collects downloadable artifacts from.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - Windows has no resource module
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


#: Directory (relative to the attempt workspace) holding generated scripts and
#: captured output.  The leading dot keeps it out of the artifact listing that
#: the harness offers the user for download.
CODE_SUBDIR = ".rsmem_code"

DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 600

#: stdout keeps its head because a script prints progressively; stderr keeps its
#: tail because a Python traceback ends with the exception that matters.
MAX_STDOUT_BYTES = 30_000
MAX_STDERR_BYTES = 10_000

TRUNCATION_NOTE = (
    "\n[... truncated. The complete stream is at {path}; read it with "
    "read_text_file if you need the rest. ...]"
)

_SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


class CodeExecutionError(RuntimeError):
    """Raised when a run could not be attempted at all."""


@dataclass(frozen=True)
class ResourceLimits:
    """Per-run guards.

    Every limit is applied independently and a platform that rejects one must
    not prevent the run.  macOS returns EINVAL for ``RLIMIT_AS``, and a library
    that lets that exception escape ``preexec_fn`` cannot spawn at all.
    """

    #: CPU seconds.  Backstop for a tight loop that the wall-clock timeout would
    #: also catch; counted per process, so it does not bound a process tree.
    cpu_seconds: int | None = 180
    #: Address space.  Effective on Linux, rejected by the macOS kernel.
    memory_bytes: int | None = 2 * 1024 * 1024 * 1024
    #: Single-file write cap, which is what keeps one run from filling the disk.
    file_size_bytes: int | None = 512 * 1024 * 1024
    #: Task count.  Counted per real UID and threads count too, so this has to
    #: stay well clear of what NumPy/OpenBLAS need while still stopping a fork
    #: bomb, which reaches thousands immediately.
    max_tasks: int | None = 512
    #: Thread caps for the numeric stack.  Not a security control -- it keeps a
    #: single run from taking every core on a shared Studio.
    thread_env_value: str | None = "4"


@dataclass
class ExecutionResult:
    """What one run reports back to the model."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    script_path: str = ""
    stdout_path: str = ""
    stderr_path: str = ""

    def as_dict(self) -> dict:
        payload = {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "truncated": self.stdout_truncated or self.stderr_truncated,
        }
        if self.stdout_truncated:
            payload["stdout_path"] = self.stdout_path
        if self.stderr_truncated:
            payload["stderr_path"] = self.stderr_path
        return payload


def code_dir(workspace: Path) -> Path:
    """Return (and create) the per-attempt scratch directory for code runs."""
    target = Path(workspace).resolve() / CODE_SUBDIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _next_label(directory: Path) -> str:
    return f"step_{len(list(directory.glob('step_*.py'))) + 1:03d}"


def child_environment(
    limits: ResourceLimits,
    workspace: Path,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the grandchild environment.

    The MCP server is already started without credentials, but this strips them
    again so the executor stays correct even when it is exercised directly.
    ``PYTHONPATH`` points at the workspace so a module written by
    ``write_script`` can be imported by a later ``run_python`` call.
    """
    source = os.environ if base is None else base
    env = {
        key: value
        for key, value in source.items()
        if not any(marker in key.upper() for marker in _SECRET_MARKERS)
    }
    env["PYTHONPATH"] = str(Path(workspace).resolve())
    # Unbuffered output matters for timeouts: a block-buffered stream loses
    # everything the script printed before it was killed.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if limits.thread_env_value:
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = limits.thread_env_value
    return env


def _limit_plan(limits: ResourceLimits) -> list[tuple[str, int]]:
    plan: list[tuple[str, int]] = []
    for attribute, value in (
        ("RLIMIT_CPU", limits.cpu_seconds),
        ("RLIMIT_AS", limits.memory_bytes),
        ("RLIMIT_FSIZE", limits.file_size_bytes),
        ("RLIMIT_NPROC", limits.max_tasks),
    ):
        if value is not None:
            plan.append((attribute, value))
    return plan


def apply_limits(limits: ResourceLimits) -> None:
    """Apply resource limits, skipping any the platform refuses.

    Never raises.  This runs inside ``preexec_fn``, where a single escaping
    exception aborts the spawn outright -- the exact failure mode that makes
    LangChain's shell middleware unusable on macOS once a memory cap is set.
    """
    if resource is None:  # pragma: no cover - Windows
        return
    for attribute, value in _limit_plan(limits):
        constant = getattr(resource, attribute, None)
        if constant is None:
            continue
        try:
            _, hard = resource.getrlimit(constant)
            ceiling = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(constant, (ceiling, hard))
        except (ValueError, OSError):
            continue


_PROBE_SOURCE = """
import json, resource, sys
plan = json.loads(sys.argv[1])
applied, rejected = [], []
for attribute, value in plan:
    constant = getattr(resource, attribute, None)
    if constant is None:
        rejected.append(attribute)
        continue
    try:
        _, hard = resource.getrlimit(constant)
        ceiling = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(constant, (ceiling, hard))
    except (ValueError, OSError):
        rejected.append(attribute)
    else:
        applied.append(attribute)
print(json.dumps({"applied": applied, "rejected": rejected}))
"""


def probe_limits(limits: ResourceLimits | None = None) -> tuple[list[str], list[str]]:
    """Report which limits this platform accepts.

    Runs in a throwaway child so the caller's own limits are never touched.
    Exists because the platform differences here are silent otherwise: Linux
    honours ``RLIMIT_AS`` while the macOS kernel rejects it with EINVAL, and a
    developer should be able to see that rather than assume memory is capped.
    """
    import json

    limits = limits or ResourceLimits()
    plan = _limit_plan(limits)
    if resource is None:  # pragma: no cover - Windows
        return [], [name for name, _ in plan]
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE_SOURCE, json.dumps(plan)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:  # pragma: no cover - defensive
        return [], [name for name, _ in plan]
    report = json.loads(completed.stdout)
    return report["applied"], report["rejected"]


def _terminate_group(process: subprocess.Popen) -> None:
    """Kill the whole process group, on every exit path.

    Killing only the direct child leaves backgrounded grandchildren running --
    a leak both LangChain's shell middleware and deepagents' local backend still
    have.  The group is only killed when the child actually leads its own group,
    so this can never signal the MCP server itself.
    """
    if os.name != "posix":  # pragma: no cover - deployment is POSIX
        if process.poll() is None:
            process.kill()
        return
    pgid = process.pid  # start_new_session=True makes the child a group leader
    if pgid == os.getpgrp():  # pragma: no cover - defensive
        if process.poll() is None:
            process.kill()
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _read_capped(path: Path, max_bytes: int, keep_tail: bool) -> tuple[str, bool]:
    if not path.exists():
        return "", False
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size <= max_bytes:
            return handle.read().decode("utf-8", "replace"), False
        if keep_tail:
            handle.seek(size - max_bytes)
        chunk = handle.read(max_bytes)
    text = chunk.decode("utf-8", "replace")
    note = TRUNCATION_NOTE.format(path=path)
    return (note.strip() + "\n" + text) if keep_tail else (text + note), True


def run_python(
    code: str,
    workspace: Path,
    timeout_s: int | None = None,
    limits: ResourceLimits | None = None,
) -> ExecutionResult:
    """Run ``code`` in a fresh interpreter rooted at ``workspace``."""
    if not code or not code.strip():
        raise CodeExecutionError("No code was supplied.")
    workspace = Path(workspace).resolve()
    if not workspace.is_dir():
        raise CodeExecutionError(f"Workspace directory does not exist: {workspace}")

    limits = limits or ResourceLimits()
    budget = DEFAULT_TIMEOUT_S if timeout_s is None else int(timeout_s)
    budget = max(1, min(budget, MAX_TIMEOUT_S))

    scratch = code_dir(workspace)
    label = _next_label(scratch)
    script = scratch / f"{label}.py"
    script.write_text(code, encoding="utf-8")
    out_path = scratch / f"{label}.stdout.log"
    err_path = scratch / f"{label}.stderr.log"

    def _preexec() -> None:  # pragma: no cover - runs in the child
        apply_limits(limits)

    timed_out = False
    exit_code: int | None = None
    with out_path.open("wb") as out_handle, err_path.open("wb") as err_handle:
        process = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            cwd=str(workspace),
            env=child_environment(limits, workspace),
            stdin=subprocess.DEVNULL,
            stdout=out_handle,
            stderr=err_handle,
            start_new_session=(os.name == "posix"),
            preexec_fn=_preexec if os.name == "posix" else None,
        )
        try:
            exit_code = process.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Runs on both paths: a clean exit can still leave backgrounded
            # grandchildren behind.
            _terminate_group(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                pass

    stdout, stdout_truncated = _read_capped(out_path, MAX_STDOUT_BYTES, keep_tail=False)
    stderr, stderr_truncated = _read_capped(err_path, MAX_STDERR_BYTES, keep_tail=True)

    if timed_out:
        stderr = (
            f"[Execution exceeded the {budget}s limit and was terminated. "
            "Output collected before the timeout is preserved above.]\n" + stderr
        )

    return ExecutionResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        script_path=str(script),
        stdout_path=str(out_path),
        stderr_path=str(err_path),
    )
