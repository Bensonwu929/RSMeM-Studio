"""Code tools exposed to the RSMeM ReAct agent as peers of the geospatial tools.

The released catalog covers raster and numeric operations, but nothing in it
reads a JSON or CSV file, lists a directory, or performs an operation the
catalog does not already name. These four tools fill that gap so the agent can
solve a task by writing code when no tool matches, without leaving the MCP
boundary that the rest of the Studio executes through.

Execution itself lives in :mod:`code_executor`, which runs each script in a
short-lived grandchild process. Nothing the model writes executes inside this
server, so a print statement can never corrupt the stdio JSONRPC framing and a
runaway script can never take the server down mid-attempt.

None of these tools raise. LangGraph's default tool-error handling re-raises
anything that is not a ``ToolException``, which ends the whole attempt -- a
mistyped path would cost the run rather than a turn. Refusals and failures come
back as ordinary results the agent can read and correct, the same way a crashed
script comes back as a non-zero exit code.
"""

from __future__ import annotations

import argparse
import functools as _functools
import sys as _sys
from pathlib import Path

from fastmcp import FastMCP

# MCP stdio transport uses stdout as the JSONRPC channel; stray print() output
# on stdout corrupts message framing and deadlocks tool calls. Route all
# diagnostic prints in this module to stderr (does not affect tool return values).
print = _functools.partial(print, file=_sys.stderr)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from code_executor import (  # noqa: E402
    CODE_SUBDIR,
    DEFAULT_TIMEOUT_S,
    MAX_TIMEOUT_S,
    CodeExecutionError,
    run_python as _run_python,
)

mcp = FastMCP()
parser = argparse.ArgumentParser()
parser.add_argument("--temp_dir", type=str)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir).resolve()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_READ_BYTES = 30_000
MAX_LISTED_ENTRIES = 200


class _ToolRefusal(Exception):
    """A recoverable refusal, reported to the model instead of raised at it."""


def _refusal(message: str) -> dict:
    return {"error": message}


def _resolve_inside(path: str) -> Path:
    """Resolve a model-supplied path, refusing anything outside the workspace.

    The harness path policy only inspects arguments whose names look like
    paths, and it cannot inspect the body of a script at all, so containment is
    enforced here rather than assumed upstream.
    """
    if not path or not str(path).strip():
        raise _ToolRefusal("A path is required.")
    candidate = Path(str(path)).expanduser()
    if not candidate.is_absolute():
        candidate = TEMP_DIR / candidate
    candidate = candidate.resolve()
    if candidate != TEMP_DIR and TEMP_DIR not in candidate.parents:
        raise _ToolRefusal(
            f"Path is outside this request's working directory: {path}"
        )
    return candidate


@mcp.tool(
    name="run_python",
    description=f"""
Run Python in this request's working directory and return its output.

Use this for any operation you want to perform directly -- a statistic nobody
exposes, a custom mask rule, reshaping results that several tools produced, or
a calculation you would rather express as code. It is a peer of the released
geospatial tools, not a last resort.

Execution model:
* Every call starts a FRESH interpreter. Variables, imports and state DO NOT
  carry over between calls. Pass intermediate results through files: write
  `numpy.save('mean.npy', arr)` in one call and `numpy.load('mean.npy')` in
  the next.
* The working directory is this request's directory, so relative paths are
  the natural way to read inputs and write outputs. Files you write there
  become downloadable artifacts for the user.
* If an import or a network call fails, that failure is the answer: read it
  from `stderr` and adapt, rather than assuming what is or is not available.
* Default timeout {DEFAULT_TIMEOUT_S}s, maximum {MAX_TIMEOUT_S}s.
* stdout is truncated from the end and stderr from the beginning when large;
  the untruncated stream is written to a file whose path is returned, and you
  can retrieve it with read_text_file.

Print whatever you need to observe: only what you print comes back.

A non-zero exit code is returned normally rather than raised, so read
`exit_code` and `stderr` and correct your code. Never state a numeric result
from a run whose `exit_code` is non-zero.

Do NOT reimplement perception models here. Object detection, segmentation,
change detection, captioning and OCR come from the perception tools; if those
are unavailable, say so instead of approximating them in code.
""",
)
def run_python(code: str, timeout_s: int | None = None) -> dict:
    try:
        return _run_python(code, TEMP_DIR, timeout_s=timeout_s).as_dict()
    except CodeExecutionError as exc:
        return _refusal(str(exc))


@mcp.tool(
    name="read_text_file",
    description=f"""
Read a UTF-8 text file from this request's working directory.

Use it to bring results your code produced -- JSON, CSV, plain text, a saved
log -- back into the conversation, and to retrieve the complete stream when
run_python reports truncated output. It is not for rasters; inspect those with
the raster tools.

Returns at most {MAX_READ_BYTES} bytes; pass `offset` to continue past that.
""",
)
def read_text_file(path: str, offset: int = 0) -> dict:
    try:
        target = _resolve_inside(path)
    except _ToolRefusal as refusal:
        return _refusal(str(refusal))
    if not target.is_file():
        return _refusal(f"File does not exist: {path}")
    size = target.stat().st_size
    start = max(0, int(offset))
    with target.open("rb") as handle:
        handle.seek(start)
        chunk = handle.read(MAX_READ_BYTES)
    end = start + len(chunk)
    return {
        "path": str(target),
        "content": chunk.decode("utf-8", "replace"),
        "bytes_total": size,
        "bytes_returned": len(chunk),
        "next_offset": end if end < size else None,
    }


@mcp.tool(
    name="list_workspace",
    description="""
List filenames and sizes in this request's working directory.

This exists so you can see what your OWN code has written, when you would
otherwise be guessing at a filename you produced earlier. It reports nothing
but names and byte counts.

Do NOT open with this tool. To find and inspect the rasters supplied with the
task, use the raster inspection tool: it reports bands, shape, dtype and CRS,
which is what any subsequent geospatial operation depends on and which this
tool cannot tell you.

Generated scripts and captured output streams are kept in a hidden directory
and are not listed.
""",
)
def list_workspace(subdir: str | None = None) -> dict:
    try:
        root = _resolve_inside(subdir) if subdir else TEMP_DIR
    except _ToolRefusal as refusal:
        return _refusal(str(refusal))
    if not root.is_dir():
        return _refusal(f"Not a directory: {subdir}")
    entries = []
    for item in sorted(root.rglob("*")):
        if any(part.startswith(".") for part in item.relative_to(root).parts):
            continue
        if not item.is_file():
            continue
        entries.append(
            {
                "path": str(item.relative_to(TEMP_DIR)),
                "bytes": item.stat().st_size,
            }
        )
        if len(entries) >= MAX_LISTED_ENTRIES:
            break
    return {"root": str(root), "count": len(entries), "files": entries}


@mcp.tool(
    name="write_script",
    description=f"""
Write a UTF-8 text file into this request's working directory.

Its purpose is to hold a longer analysis as a reusable module so you do not
resend it on every run: write `helper.py` here, then `import helper` from
run_python, which already has this directory on its import path. Fixing one
line then means rewriting one file rather than repeating the whole script.

Also usable for small data or configuration files. Generated scripts are kept
out of the hidden `{CODE_SUBDIR}` directory that run_python manages.
""",
)
def write_script(path: str, content: str) -> dict:
    try:
        target = _resolve_inside(path)
    except _ToolRefusal as refusal:
        return _refusal(str(refusal))
    if target.is_dir():
        return _refusal(f"Path is a directory: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"path": str(target), "bytes": target.stat().st_size}


if __name__ == "__main__":
    mcp.run()
