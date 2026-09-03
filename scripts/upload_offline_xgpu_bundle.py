#!/usr/bin/env python3
"""Upload completed offline artifacts from QY_11 directly to ModelScope.

The ModelScope token is read once from stdin and is never written or printed.
Run one or more explicit filenames so an archive still being built cannot be
uploaded accidentally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modelscope.hub.api import HubApi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--repo-id", default="wbx929/RSMeM-Perception-Models")
    parser.add_argument(
        "--root",
        default="/slow_share2/ultrafast_share/gzh_data/wubingxian/xgpu-offline-build/output",
    )
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if not token:
        raise SystemExit("A ModelScope token is required on stdin")
    root = Path(args.root)
    api = HubApi()
    for filename in args.files:
        source = root / filename
        if not source.is_file() or source.stat().st_size == 0:
            raise SystemExit(f"Missing or empty artifact: {source}")
        print(f"[upload] {filename} ({source.stat().st_size} bytes)", flush=True)
        api.upload_file(
            path_or_fileobj=source,
            path_in_repo=f"offline/{filename}",
            repo_id=args.repo_id,
            repo_type="model",
            revision="master",
            token=token,
            commit_message=f"Upload offline xGPU artifact {filename}",
            buffer_size_mb=16,
        )
        print(f"[upload] completed {filename}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
