#!/usr/bin/env bash
set -euo pipefail

# Build relocatable Linux/xGPU source and runtime archives from the environments
# that are already serving the validated QY_11 perception workers.  Run this on
# QY_11; the resulting archives are uploaded to the private ModelScope model
# repository and are never committed to the Studio Git repository.

BUILD_ROOT="${RSMEM_OFFLINE_BUILD_ROOT:-/slow_share2/ultrafast_share/gzh_data/wubingxian/xgpu-offline-build}"
RS_ROOT="${RSMEM_QY_RS_ROOT:-/slow_share2/ultrafast_share/gzh_data/wubingxian/RSMeM-public}"
OEA_ROOT="${RSMEM_QY_OEA_ROOT:-/slow_share2/ultrafast_share/gzh_data/baijin/OpenEarthAgent}"
SAM2_ENV="${RSMEM_QY_SAM2_ENV:-/slow_share2/ultrafast_share/gzh_data/yuegengxin/miniconda3/envs/sam2-deploy}"
REMOTECLIP_ENV="${RSMEM_QY_REMOTECLIP_ENV:-/slow_share2/ultrafast_share/gzh_data/yuegengxin/miniconda3/envs/remoteclip-deploy}"
CHANGEOS_ENV="${RSMEM_QY_CHANGEOS_ENV:-/slow_share2/ultrafast_share/gzh_data/yuegengxin/miniconda3/envs/changeos-deploy}"
BRIGHT_ENV="${RSMEM_QY_BRIGHT_ENV:-/slow_share2/ultrafast_share/gzh_data/yuegengxin/miniconda3/envs/bright-deploy}"
OEA_E1_ENV="${RSMEM_QY_OEA_E1_ENV:-/slow_share2/ultrafast_share/gzh_data/baijin/envs/tool-server-e1}"
OEA_E3_ENV="${RSMEM_QY_OEA_E3_ENV:-/slow_share2/ultrafast_share/gzh_data/baijin/envs/tool-server-e3}"
PACK_PYTHONPATH="$BUILD_ROOT/build-tools"
OUTPUT="$BUILD_ROOT/output"
STAGE="$BUILD_ROOT/stage"

mkdir -p "$OUTPUT" "$STAGE"

echo "[offline-build] staging minimal source trees"
rm -rf "$STAGE/sources"
mkdir -p \
  "$STAGE/sources/RSMeM-public/tools" \
  "$STAGE/sources/OpenEarthAgent/tool_server/tool_workers" \
  "$STAGE/sources/models" \
  "$STAGE/sources/easyocr"

rsync -a \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.git/' \
  --exclude='logs/' --exclude='benchmark/' --exclude='question*/' \
  --exclude='sam2_server_results/' \
  "$RS_ROOT/tools/deploy_tools/" \
  "$STAGE/sources/RSMeM-public/tools/deploy_tools/"

rsync -a \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.git/' \
  --exclude='logs/' --exclude='tools_output*/' --exclude='cache*/' \
  --exclude='~/' --exclude='videollava/' \
  "$OEA_ROOT/tool_server/tool_workers/" \
  "$STAGE/sources/OpenEarthAgent/tool_server/tool_workers/"

# Every selected OEA worker imports logging, request, and registration helpers
# from tool_server.utils.  Package them explicitly so runtime never falls back
# to a Git checkout or an undeclared server path.
rsync -a \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.git/' \
  "$OEA_ROOT/tool_server/utils/" \
  "$STAGE/sources/OpenEarthAgent/tool_server/utils/"

rsync -a --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
  "$OEA_ROOT/models/GroundingDINO/" "$STAGE/sources/models/GroundingDINO/"
rsync -a --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' \
  "$OEA_ROOT/models/LAE-DINO/" "$STAGE/sources/models/LAE-DINO/"
rsync -a "$OEA_ROOT/../easyocr/model/" "$STAGE/sources/easyocr/model/"

tar -C "$STAGE/sources" -czf "$OUTPUT/rsmem-perception-sources-v1.tar.gz" .

pack_env() {
  local prefix="$1"
  local output="$2"
  if [[ -s "$output" ]] && gzip -t "$output"; then
    echo "[offline-build] reusing completed archive $output"
    return
  fi
  echo "[offline-build] packing $prefix -> $output"
  PYTHONPATH="$PACK_PYTHONPATH" python3 - "$prefix" "$output" <<'PY'
import sys
from conda_pack import pack

pack(
    prefix=sys.argv[1],
    output=sys.argv[2],
    format="tar.gz",
    ignore_editable_packages=True,
    ignore_missing_files=True,
    force=True,
)
PY
}

pack_env "$SAM2_ENV" "$OUTPUT/sam2-py310-cu121-v1.tar.gz"
pack_env "$REMOTECLIP_ENV" "$OUTPUT/remoteclip-py310-cu121-v1.tar.gz"
pack_env "$CHANGEOS_ENV" "$OUTPUT/changeos-py310-cu121-v1.tar.gz"
pack_env "$BRIGHT_ENV" "$OUTPUT/bright-py310-cu121-v1.tar.gz"
pack_env "$OEA_E1_ENV" "$OUTPUT/oea-e1-py310-cu124-v1.tar.gz"

# The shared E3 prefix contains one zero-byte nvJitLink library.  Never mutate
# another user's live environment: copy it into our staging area and repair the
# staged copy with the matching healthy CUDA 12 library from E1.  A completed,
# gzip-valid archive is immutable and can be reused without restaging ~5 GB.
OEA_E3_ARCHIVE="$OUTPUT/oea-e3-py310-cu124-v1.tar.gz"
if [[ -s "$OEA_E3_ARCHIVE" ]] && gzip -t "$OEA_E3_ARCHIVE"; then
  echo "[offline-build] reusing completed archive $OEA_E3_ARCHIVE"
else
  echo "[offline-build] staging and validating repaired OEA E3 environment"
  rm -rf "$STAGE/oea-e3"
  rsync -a "$OEA_E3_ENV/" "$STAGE/oea-e3/"
  cp -f \
    "$OEA_E1_ENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12" \
    "$STAGE/oea-e3/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12"
  "$STAGE/oea-e3/bin/python" -c \
    'import torch, mmcv, mmdet; print(torch.__version__, torch.version.cuda, mmcv.__version__, mmdet.__version__)'
  pack_env "$STAGE/oea-e3" "$OEA_E3_ARCHIVE"
fi

echo "[offline-build] checksums"
(cd "$OUTPUT" && sha256sum \
  rsmem-perception-sources-v1.tar.gz \
  sam2-py310-cu121-v1.tar.gz \
  remoteclip-py310-cu121-v1.tar.gz \
  changeos-py310-cu121-v1.tar.gz \
  bright-py310-cu121-v1.tar.gz \
  oea-e1-py310-cu124-v1.tar.gz \
  oea-e3-py310-cu124-v1.tar.gz > SHA256SUMS)
cat "$OUTPUT/SHA256SUMS"
