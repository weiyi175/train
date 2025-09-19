#!/usr/bin/env bash
set -euo pipefail
# 全流程：重新以 pose -> eigen 特徵 -> slipce 視窗 (short/long) -> NPZ
# 目標：確保 test_data/slipce 具有 36 個核心特徵 (含 norm_dist_leftHand_mouth / norm_dist_rightHand_mouth)
# 使用：
#   ./full_rebuild_test_slipce.sh \
#       --pose_dir /home/user/projects/train/test_data/extract_pose \
#       --eigen_out /home/user/projects/train/test_data/Eigenvalue_new \
#       --slipce_out /home/user/projects/train/test_data/slipce \
#       --keep_old
# 可選：--all_features / --phase / --norm_scope dataset / --export_dense

POSE_DIR="/home/user/projects/train/test_data/extract_pose"
EIG_OUT="/home/user/projects/train/test_data/Eigenvalue_new"
SLIPCE_OUT="/home/user/projects/train/test_data/slipce"
PY=${PYTHON:-python3}
SHORT_WIN=30
SHORT_STRIDE=15
LONG_WIN=75
LONG_STRIDE=40
SMOKE_THRESH=0.5
NORM_SCOPE=video
SEED=123
EXTRA_SLIPCE=()
KEEP_OLD=false

usage(){
cat <<EOF
Usage: $0 [options]
  --pose_dir DIR       pose csv 來源資料夾 (含 *_pose.csv)
  --eigen_out DIR      產出 eigen 特徵目錄
  --slipce_out DIR     視窗輸出目錄 (最終 windows_npz.npz)
  --short_win N        短視窗長度 (default $SHORT_WIN)
  --short_stride N     短視窗步長 (default $SHORT_STRIDE)
  --long_win N         長視窗長度 (default $LONG_WIN)
  --long_stride N      長視窗步長 (default $LONG_STRIDE)
  --smoke_thresh F     視窗 smoke 比例閾值 (default $SMOKE_THRESH)
  --norm_scope S       長視窗正規化 (video|dataset) (default $NORM_SCOPE)
  --seed N             隨機種子 (default $SEED)
  --all_features       使用全部特徵 (非僅 CORE)
  --phase              啟用 phase
  --export_dense       同時輸出 dense npz
  --keep_old           不移除原 slipce_out 內既有 *windows_npz*.npz (只做 .backup)
  --help               顯示此說明
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pose_dir) POSE_DIR="$2"; shift 2;;
    --eigen_out) EIG_OUT="$2"; shift 2;;
    --slipce_out) SLIPCE_OUT="$2"; shift 2;;
    --short_win) SHORT_WIN="$2"; shift 2;;
    --short_stride) SHORT_STRIDE="$2"; shift 2;;
    --long_win) LONG_WIN="$2"; shift 2;;
    --long_stride) LONG_STRIDE="$2"; shift 2;;
    --smoke_thresh) SMOKE_THRESH="$2"; shift 2;;
    --norm_scope) NORM_SCOPE="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --all_features) EXTRA_SLIPCE+=(--all_features); shift 1;;
    --phase) EXTRA_SLIPCE+=(--enable_phase); shift 1;;
    --export_dense) EXTRA_SLIPCE+=(--export_npz_dense); shift 1;;
    --keep_old) KEEP_OLD=true; shift 1;;
    --help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

TS=$(date +%Y%m%d_%H%M%S)

echo "[1/3] Recompute eigen features -> $EIG_OUT"
mkdir -p "$EIG_OUT"
$PY Tool/Eigenvalue.py --pose_dir "$POSE_DIR" --out_dir "$EIG_OUT"

echo "[2/3] Backup old slipce outputs (if exist)"
mkdir -p "$SLIPCE_OUT"
if [[ -f "$SLIPCE_OUT/windows_npz.npz" ]]; then
  mv "$SLIPCE_OUT/windows_npz.npz" "$SLIPCE_OUT/windows_npz.backup_${TS}.npz"
fi
if [[ -f "$SLIPCE_OUT/windows_dense_npz.npz" ]]; then
  mv "$SLIPCE_OUT/windows_dense_npz.npz" "$SLIPCE_OUT/windows_dense_npz.backup_${TS}.npz"
fi

# 若不 keep_old, 清除 CSV 結果避免混淆
if ! $KEEP_OLD; then
  rm -f "$SLIPCE_OUT/short_windows.csv" "$SLIPCE_OUT/long_windows.csv" || true
fi

echo "[3/3] Run slipce slicing -> $SLIPCE_OUT"
CMD=("$PY" Tool/slipce.py \
  --eigen_dir "$EIG_OUT" \
  --label_dir /home/user/projects/train/test_data/VIA_smoke_nosmoke \
  --output_dir "$SLIPCE_OUT" \
  --short_win "$SHORT_WIN" --short_stride "$SHORT_STRIDE" \
  --long_win "$LONG_WIN" --long_stride "$LONG_STRIDE" \
  --smoke_ratio_thresh "$SMOKE_THRESH" \
  --long_norm_scope "$NORM_SCOPE" \
  --export_npz \
  --seed "$SEED")

# 附加額外參數
CMD+=("${EXTRA_SLIPCE[@]}")

echo "Running slipce: ${CMD[*]}" | tee "$SLIPCE_OUT/full_rebuild_cmd_${TS}.txt"
"${CMD[@]}"

python3 - <<'PY'
import numpy as np, os
slipce_out = os.environ.get('SLIPCE_OUT','/home/user/projects/train/test_data/slipce')
npz = os.path.join(slipce_out,'windows_npz.npz')
if os.path.isfile(npz):
    D = np.load(npz, allow_pickle=True)
    feats = D['feature_list'].tolist()
    print('[CHECK] feature_list len =', len(feats))
    print('[CHECK] features =', feats)
else:
    print('[CHECK] missing NPZ:', npz)
PY

echo "Done full rebuild."
