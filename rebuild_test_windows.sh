#!/usr/bin/env bash
set -euo pipefail
# 重新切割 test 資料集產生 slipce 視窗 (short/long) 並輸出 NPZ
# 預設使用核心特徵 (36) 與長視窗 75/stride 40, 短視窗 30/stride 15
# 若要使用全部特徵加上 --all_features
# 可自訂: --short_win --short_stride --long_win --long_stride
# 會備份既有 output_dir/windows_npz.npz (加時間戳) 才覆蓋

EIGEN_DIR="/home/user/projects/train/test_data/Eigenvalue"
LABEL_DIR="/home/user/projects/train/test_data/VIA_smoke_nosmoke"
OUTPUT_DIR="/home/user/projects/train/test_data/slipce"
PY=${PYTHON:-python3}

SHORT_WIN=30
SHORT_STRIDE=15
LONG_WIN=75
LONG_STRIDE=40
SMOKE_THRESH=0.5
NORM_SCOPE=video  # video | dataset
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [options]
Options:
  --eigen DIR          Eigenvalue 目錄 (default: $EIGEN_DIR)
  --label DIR          VIA 標記目錄 (default: $LABEL_DIR)
  --out DIR            輸出目錄 (default: $OUTPUT_DIR)
  --short_win N        短視窗長度 (default: $SHORT_WIN)
  --short_stride N     短視窗步長 (default: $SHORT_STRIDE)
  --long_win N         長視窗長度 (default: $LONG_WIN)
  --long_stride N      長視窗步長 (default: $LONG_STRIDE)
  --smoke_thresh F     smoke 視窗比例閾值 (default: $SMOKE_THRESH)
  --norm_scope S       長視窗正規化 video|dataset (default: $NORM_SCOPE)
  --all_features       使用全部特徵
  --phase              啟用六階段 phase 序列
  --seed N             隨機種子 (default: 42)
  --no_short_bg        不啟用短視窗背景補樣 (預設: 關閉; 加上此旗標代表顯式關閉, 僅為語意清楚)
  --bg_ratio F         短視窗背景 no_smoke:smoke 目標比
  --hard_nearby        啟用難負樣本 (nearby)
  --fps F              指定 FPS (全部影片強制)
  --help               顯示此說明
範例:
  $0 --out /home/user/projects/train/test_data/slipce --export_dense
  $0 --norm_scope dataset --all_features --phase
EOF
}

SEED=42
BG_RATIO=1.0
ENABLE_BG=false
ENABLE_HARD=false
FPS_FORCE=0
EXPORT_DENSE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eigen) EIGEN_DIR="$2"; shift 2;;
    --label) LABEL_DIR="$2"; shift 2;;
    --out) OUTPUT_DIR="$2"; shift 2;;
    --short_win) SHORT_WIN="$2"; shift 2;;
    --short_stride) SHORT_STRIDE="$2"; shift 2;;
    --long_win) LONG_WIN="$2"; shift 2;;
    --long_stride) LONG_STRIDE="$2"; shift 2;;
    --smoke_thresh) SMOKE_THRESH="$2"; shift 2;;
    --norm_scope) NORM_SCOPE="$2"; shift 2;;
    --all_features) EXTRA_ARGS+=(--all_features); shift 1;;
    --phase) EXTRA_ARGS+=(--enable_phase); shift 1;;
    --seed) SEED="$2"; shift 2;;
    --no_short_bg) ENABLE_BG=false; shift 1;;
    --bg_ratio) BG_RATIO="$2"; ENABLE_BG=true; shift 2;;
    --hard_nearby) ENABLE_HARD=true; shift 1;;
    --fps) FPS_FORCE="$2"; shift 2;;
    --export_dense) EXPORT_DENSE=true; shift 1;;
    --help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

set -x
mkdir -p "$OUTPUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
if [[ -f "$OUTPUT_DIR/windows_npz.npz" ]]; then
  mv "$OUTPUT_DIR/windows_npz.npz" "$OUTPUT_DIR/windows_npz.backup_${TS}.npz"
fi
if [[ -f "$OUTPUT_DIR/windows_dense_npz.npz" ]]; then
  mv "$OUTPUT_DIR/windows_dense_npz.npz" "$OUTPUT_DIR/windows_dense_npz.backup_${TS}.npz"
fi

CMD=("$PY" Tool/slipce.py \
  --eigen_dir "$EIGEN_DIR" \
  --label_dir "$LABEL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --short_win "$SHORT_WIN" --short_stride "$SHORT_STRIDE" \
  --long_win "$LONG_WIN" --long_stride "$LONG_STRIDE" \
  --smoke_ratio_thresh "$SMOKE_THRESH" \
  --long_norm_scope "$NORM_SCOPE" \
  --export_npz)

if $EXPORT_DENSE; then
  CMD+=(--export_npz_dense)
fi
if $ENABLE_BG; then
  CMD+=(--short_bg_sample --short_bg_ratio "$BG_RATIO")
fi
if $ENABLE_HARD; then
  CMD+=(--hard_negative_nearby)
fi
if [[ "$FPS_FORCE" != "0" ]]; then
  CMD+=(--fps "$FPS_FORCE")
fi
CMD+=(--seed "$SEED" "${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}" | tee "$OUTPUT_DIR/rebuild_cmd_${TS}.txt"
"${CMD[@]}"

# 簡單檢查 feature_list
python3 - <<'PY'
import numpy as np, os, sys
out_dir = os.environ.get('OUTPUT_DIR','/home/user/projects/train/test_data/slipce')
npz_path = os.path.join(out_dir,'windows_npz.npz')
if not os.path.isfile(npz_path):
    print('NPZ not found:', npz_path); sys.exit(1)
D = np.load(npz_path, allow_pickle=True)
feat = D['feature_list'].tolist()
print('feature_list len =', len(feat))
print('features =', feat)
for k in ['long_raw','short_raw']:
    if k in D:
        arr = D[k]
        try:
            print(k, 'shape', arr.shape, 'dtype', arr.dtype)
        except Exception:
            print(k, 'loaded')
PY

echo "Done."
