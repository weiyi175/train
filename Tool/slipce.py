"""Multi-scale sliding window slicing for smoking action dataset.

功能概要:
1. 讀取 Eigenvalue 特徵檔 ( *_eig.csv ) 以及對應 VIA 標記 ( *_full.csv )。
2. 依時間對每一幀配置 frame-level label (smoke / no_smoke)。
3. 產生兩種時間尺度視窗:
   - 短視窗: 針對快速手部靠近/離開嘴部的動態 (預設 win=30, stride=15)。
	 正規化: 視窗內 z-score (window-level)。
   - 長視窗: 捕捉完整動作循環 (預設 win=75, stride=40)。
	 正規化: 影片層 (video-level) 或資料集層 (dataset-level) z-score，可選。
4. 於輸出中同時保留 raw 與 normalized 序列 (利於模型同時學幅度與形狀)。
5. 標籤策略: 視窗內 smoke 幀比例 >= threshold (預設 0.5) 則視窗標記 smoke，否則 no_smoke；亦輸出 smoke_ratio 供後處理。
6. 精簡特徵: 預設取核心行為相關欄位 (可用 --all_features 改為全欄位)。

輸出:
 output_dir/
   short_windows.csv
   long_windows.csv
   stats_short.json
   stats_long.json

每行一個視窗，欄位格式 (範例):
 video_id,scale,window_index,start_frame,end_frame,start_time,end_time,label,smoke_ratio,n_frames,
   featName_raw_t0,...,featName_raw_t{W-1},featName_norm_t0,... (對長視窗亦然)

注意:
 - 為避免記憶體爆增，直接串流 append 到 CSV。
 - 若資料集很大且想改成 .npz，可再擴充。

使用:  python Tool/slipce.py --eigen_dir path --label_dir path --output_dir path
"""

from __future__ import annotations

import os
import csv
import math
import argparse
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable, Optional

###############################################################################
# 預設路徑 (可用 CLI 參數覆寫)
###############################################################################
DEFAULT_EIGEN_DIR = "/home/user/projects/train/test_data/Eigenvalue"
DEFAULT_LABEL_DIR = "/home/user/projects/train/test_data/VIA_smoke_nosmoke"
DEFAULT_OUTPUT_DIR = "/home/user/projects/train/test_data/slipce"

###############################################################################
# 核心特徵欄位 (簡化版)。若 --all_features 則使用全部 (排除 meta)。
###############################################################################
CORE_FEATURES = [
	# 距離/正規化距離 (雙手-嘴、鼻-手)
	"dist_leftHand_mouth", "dist_rightHand_mouth",
	"norm_dist_leftHand_mouth", "norm_dist_rightHand_mouth",
	"dist_nose_leftHand", "dist_nose_rightHand",
	# 嘴部 / 置信度 / 遮擋
	"mouth_conf_adj", "occlusion_flag", "mouth_vx", "mouth_vy", "mouth_vz",
	# 手部速度加速度 (腕與食指替代: 使用 landmarks 15,16=腕, 19,20=食指)
	"l15_vx", "l15_vy", "l15_vz", "l15_ax", "l15_ay", "l15_az",
	"l16_vx", "l16_vy", "l16_vz", "l16_ax", "l16_ay", "l16_az",
	"l19_vx", "l19_vy", "l19_vz", "l19_ax", "l19_ay", "l19_az",
	"l20_vx", "l20_vy", "l20_vz", "l20_ax", "l20_ay", "l20_az",
	# 跳變旗標
	"velocity_jump_flag"
]

META_COLUMNS = {"frame", "time_sec"}

###############################################################################
# 讀取函式
###############################################################################


def read_label_intervals(csv_path: str) -> List[Tuple[float, float, str]]:
	intervals: List[Tuple[float, float, str]] = []
	if not os.path.isfile(csv_path):
		return intervals
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			try:
				s = float(row.get("start", "0") or 0)
				e = float(row.get("end", "0") or 0)
				label = (row.get("label") or "").strip()
			except ValueError:
				continue
			if e > s:
				intervals.append((s, e, label))
	return intervals


def load_eigen_csv(path: str) -> Tuple[List[Dict[str, float]], List[str]]:
	"""讀取 eigenvalue 特徵 CSV -> list of dict 及 header。
	數值轉 float, 空字串 -> NaN。
	"""
	rows: List[Dict[str, float]] = []
	with open(path, "r", encoding="utf-8") as f:
		reader = csv.reader(f)
		header = next(reader, None)
		if not header:
			return rows, []
		for r in reader:
			if not r:
				continue
			d: Dict[str, float] = {}
			for i, name in enumerate(header):
				val = r[i] if i < len(r) else ""
				if name in ("frame",):
					try:
						d[name] = int(float(val))
					except Exception:
						d[name] = math.nan
				else:
					if val == "":
						d[name] = math.nan
					else:
						try:
							d[name] = float(val)
						except Exception:
							d[name] = math.nan
			rows.append(d)
	return rows, header


###############################################################################
# 標籤與視窗工具
###############################################################################


def build_frame_labels(frames: List[Dict[str, float]], intervals: List[Tuple[float, float, str]], default_label: str = "no_smoke") -> List[str]:
	labels: List[str] = []
	for fr in frames:
		t = float(fr.get("time_sec", 0.0) or 0.0)
		lab = default_label
		for s, e, L in intervals:
			if s <= t < e:
				lab = L
				break
		labels.append(lab)
	return labels


def iter_sliding_windows(n: int, win: int, stride: int) -> Iterable[Tuple[int, int]]:
	if n <= 0 or win <= 0:
		return []
	i = 0
	while i + win <= n:
		yield i, i + win
		i += stride


def select_feature_columns(header: List[str], use_all: bool) -> List[str]:
	if use_all:
		# 排除 meta + 可能的門檻紀錄欄 (同樣允許, 僅去除 frame/time_sec)
		return [h for h in header if h not in META_COLUMNS]
	# 僅使用 CORE_FEATURES (存在才保留)
	return [h for h in CORE_FEATURES if h in header]


def compute_video_stats(frames: List[Dict[str, float]], feat_cols: List[str]) -> Dict[str, Tuple[float, float]]:
	stats: Dict[str, Tuple[float, float]] = {}
	for c in feat_cols:
		vals = [f[c] for f in frames if not math.isnan(f.get(c, math.nan))]
		if not vals:
			stats[c] = (0.0, 0.0)
			continue
		mean = sum(vals) / len(vals)
		var = sum((v - mean) ** 2 for v in vals) / max(1, (len(vals) - 1))
		std = math.sqrt(var)
		stats[c] = (mean, std)
	return stats


def compute_dataset_stats_welford(video_frames: Dict[str, List[Dict[str, float]]], feat_cols: List[str]) -> Dict[str, Tuple[float, float]]:
	"""使用 Welford 演算法精確聚合全資料集 mean/std (忽略 NaN)。"""
	acc = {c: {"n": 0, "mean": 0.0, "M2": 0.0} for c in feat_cols}
	for frames in video_frames.values():
		for fr in frames:
			for c in feat_cols:
				v = fr.get(c, math.nan)
				if math.isnan(v):
					continue
				st = acc[c]
				st["n"] += 1
				delta = v - st["mean"]
				st["mean"] += delta / st["n"]
				delta2 = v - st["mean"]
				st["M2"] += delta * delta2
	out: Dict[str, Tuple[float, float]] = {}
	for c, st in acc.items():
		n = st["n"]
		if n < 2:
			out[c] = (st["mean"], 0.0)
		else:
			var = st["M2"] / (n - 1)
			out[c] = (st["mean"], math.sqrt(max(var, 0.0)))
	return out


def zscore(value: float, mean: float, std: float) -> float:
	if std <= 1e-8:
		return 0.0
	return (value - mean) / std


###############################################################################
# 視窗處理主流程
###############################################################################


def process_video(
	video_id: str,
	frames: List[Dict[str, float]],
	frame_labels: List[str],
	feat_cols: List[str],
	short_win: int,
	short_stride: int,
	long_win: int,
	long_stride: int,
	smoke_ratio_thresh: float,
	long_norm_stats: Dict[str, Tuple[float, float]],
	short_writer,
	long_writer,
	short_header_ref: List[str],
	long_header_ref: List[str],
	phases: Optional[List[int]] = None,
	dist_min_series: Optional[List[float]] = None,
	dist_speed_series: Optional[List[float]] = None,
	phase_enable: bool = False,
	npz_collect: Optional[Dict[str, List]] = None,
	merge_config: Optional[Dict] = None,
):
	n_frames = len(frames)

	def short_ranges():
		if not merge_config or not merge_config.get("enable"):
			for rng in iter_sliding_windows(n_frames, short_win, short_stride):
				yield rng
			return
		# only inside smoke segments
		for (ss, ee) in merge_config.get("smoke_segments", []):
			inner_stride_factor = max(1, int(merge_config.get("inner_stride_factor", 1)))
			stride_inner = max(1, short_win // inner_stride_factor)
			i = ss
			while i + short_win <= ee:
				yield (i, i + short_win)
				i += stride_inner

	short_total = short_smoke = 0
	long_total = long_smoke = 0

	# short windows
	short_index = 0
	for s, e in short_ranges():
		sub = frames[s:e]
		labs = frame_labels[s:e]
		smoke_frames = sum(1 for L in labs if L == "smoke")
		ratio = smoke_frames / len(labs)
		label = "smoke" if ratio >= smoke_ratio_thresh else "no_smoke"
		short_total += 1
		if label == "smoke":
			short_smoke += 1
		per_stats = compute_video_stats(sub, feat_cols)
		row_meta = [video_id, "short", short_index, sub[0].get("frame", 0), sub[-1].get("frame", 0),
					f"{sub[0].get('time_sec',0.0):.6f}", f"{sub[-1].get('time_sec',0.0):.6f}", label, f"{ratio:.4f}", len(sub)]
		seq_raw = []
		seq_norm = []
		for fcol in feat_cols:
			mean, std = per_stats[fcol]
			for fr in sub:
				v = fr.get(fcol, math.nan)
				seq_raw.append("" if math.isnan(v) else f"{v:.6f}")
				nv = 0.0 if math.isnan(v) else zscore(v, mean, std)
				seq_norm.append(f"{nv:.6f}")
		phase_seq = []
		if phase_enable and phases is not None:
			phase_seq = [str(p) for p in phases[s:e]]
		row = row_meta + seq_raw + seq_norm + phase_seq
		if not short_header_ref:
			header = ["video_id","scale","window_index","start_frame","end_frame","start_time","end_time","label","smoke_ratio","n_frames"]
			for fcol in feat_cols:
				header.extend([f"{fcol}_raw_t{i}" for i in range(short_win)])
			for fcol in feat_cols:
				header.extend([f"{fcol}_norm_t{i}" for i in range(short_win)])
			if phase_enable:
				header.extend([f"phase_t{i}" for i in range(short_win)])
			short_writer.writerow(header)
			short_header_ref.extend(header)
		short_writer.writerow(row)
		if npz_collect is not None:
			# collect (F,W)
			W = short_win
			F = len(feat_cols)
			raw_vals = []
			norm_vals = []
			for fi in range(F):
				start = fi * W
				raw_vals.append([float(x) if x != "" else math.nan for x in seq_raw[start:start+W]])
			for fi in range(F):
				start = fi * W
				norm_vals.append([float(x) for x in seq_norm[start:start+W]])
			npz_collect.setdefault("short_raw", []).append(raw_vals)
			npz_collect.setdefault("short_norm", []).append(norm_vals)
			npz_collect.setdefault("short_label", []).append(1 if label=="smoke" else 0)
			npz_collect.setdefault("short_smoke_ratio", []).append(ratio)
			npz_collect.setdefault("short_video_id", []).append(video_id)
			npz_collect.setdefault("short_start_frame", []).append(sub[0].get("frame",0))
			npz_collect.setdefault("short_end_frame", []).append(sub[-1].get("frame",0))
			if phase_seq:
				npz_collect.setdefault("short_phase", []).append([int(p) for p in phase_seq])
		short_index += 1

	# long windows
	long_index = 0
	for s, e in iter_sliding_windows(n_frames, long_win, long_stride):
		sub = frames[s:e]
		if len(sub) < long_win:
			continue
		labs = frame_labels[s:e]
		smoke_frames = sum(1 for L in labs if L == "smoke")
		ratio = smoke_frames / len(labs)
		label = "smoke" if ratio >= smoke_ratio_thresh else "no_smoke"
		long_total += 1
		if label == "smoke":
			long_smoke += 1
		row_meta = [video_id, "long", long_index, sub[0].get("frame",0), sub[-1].get("frame",0),
					f"{sub[0].get('time_sec',0.0):.6f}", f"{sub[-1].get('time_sec',0.0):.6f}", label, f"{ratio:.4f}", len(sub)]
		seq_raw = []
		seq_norm = []
		for fcol in feat_cols:
			mean, std = long_norm_stats.get(fcol, (0.0,0.0))
			for fr in sub:
				v = fr.get(fcol, math.nan)
				seq_raw.append("" if math.isnan(v) else f"{v:.6f}")
				nv = 0.0 if math.isnan(v) else zscore(v, mean, std)
				seq_norm.append(f"{nv:.6f}")
		phase_seq = []
		approach_speed_peak = hold_duration = leave_speed_peak = 0.0
		if phase_enable and phases is not None and dist_min_series is not None and dist_speed_series is not None:
			phase_seq = [str(p) for p in phases[s:e]]
			# compute summary metrics
			idxs = list(range(s,e))
			local_d = [dist_min_series[i] if dist_min_series[i] is not None else math.nan for i in idxs]
			local_spd = [dist_speed_series[i] for i in idxs]
			valid_pairs = [(i,d) for i,d in enumerate(local_d) if not math.isnan(d)]
			if valid_pairs:
				min_i = min(valid_pairs, key=lambda x: x[1])[0]
				before = local_spd[:min_i]
				after = local_spd[min_i+1:]
				if before:
					approach_speed_peak = min(before)
				if after:
					leave_speed_peak = max(after)
				d_sorted = sorted([d for d in local_d if not math.isnan(d)])
				if d_sorted:
					near_thr = d_sorted[max(0,int(0.25*len(d_sorted))-1)]
					spd_abs = [abs(v) for v in local_spd]
					spd_sorted = sorted(spd_abs)
					spd_thr = spd_sorted[max(0,int(0.3*len(spd_sorted))-1)] if spd_sorted else 0.0
					cur=best=0
					for dval,sval in zip(local_d,spd_abs):
						if (not math.isnan(dval)) and dval<=near_thr and sval<=spd_thr:
							cur+=1
							best=max(best,cur)
						else:
							cur=0
					hold_duration = float(best)
		row = row_meta + seq_raw + seq_norm
		if phase_enable:
			row += phase_seq
			row += [f"{approach_speed_peak:.6f}", f"{hold_duration:.0f}", f"{leave_speed_peak:.6f}"]
		if not long_header_ref:
			header = ["video_id","scale","window_index","start_frame","end_frame","start_time","end_time","label","smoke_ratio","n_frames"]
			for fcol in feat_cols:
				header.extend([f"{fcol}_raw_t{i}" for i in range(long_win)])
			for fcol in feat_cols:
				header.extend([f"{fcol}_norm_t{i}" for i in range(long_win)])
			if phase_enable:
				header.extend([f"phase_t{i}" for i in range(long_win)])
				header.extend(["approach_speed_peak","hold_duration","leave_speed_peak"])
			long_writer.writerow(header)
			long_header_ref.extend(header)
		long_writer.writerow(row)
		if npz_collect is not None:
			W = long_win
			F = len(feat_cols)
			raw_vals=[]; norm_vals=[]
			for fi in range(F):
				start = fi*W
				raw_vals.append([float(x) if x!="" else math.nan for x in seq_raw[start:start+W]])
			for fi in range(F):
				start = fi*W
				norm_vals.append([float(x) for x in seq_norm[start:start+W]])
			npz_collect.setdefault("long_raw", []).append(raw_vals)
			npz_collect.setdefault("long_norm", []).append(norm_vals)
			npz_collect.setdefault("long_label", []).append(1 if label=="smoke" else 0)
			npz_collect.setdefault("long_smoke_ratio", []).append(ratio)
			npz_collect.setdefault("long_video_id", []).append(video_id)
			npz_collect.setdefault("long_start_frame", []).append(sub[0].get("frame",0))
			npz_collect.setdefault("long_end_frame", []).append(sub[-1].get("frame",0))
			if phase_seq:
				npz_collect.setdefault("long_phase", []).append([int(p) for p in phase_seq])
				npz_collect.setdefault("long_approach_speed_peak", []).append(approach_speed_peak)
				npz_collect.setdefault("long_hold_duration", []).append(hold_duration)
				npz_collect.setdefault("long_leave_speed_peak", []).append(leave_speed_peak)
		long_index += 1

	return short_total, short_smoke, long_total, long_smoke


###############################################################################
# 主執行流程
###############################################################################


def main():
	ap = argparse.ArgumentParser(description="Multi-scale window slicing for smoking action (short + long windows)")
	ap.add_argument("--eigen_dir", default=DEFAULT_EIGEN_DIR)
	ap.add_argument("--label_dir", default=DEFAULT_LABEL_DIR)
	ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
	ap.add_argument("--short_win", type=int, default=30)
	ap.add_argument("--short_stride", type=int, default=15)
	ap.add_argument("--long_win", type=int, default=75)
	ap.add_argument("--long_stride", type=int, default=40)
	ap.add_argument("--smoke_ratio_thresh", type=float, default=0.5, help="視窗內 smoke 幀比例達阈值則標記 smoke")
	ap.add_argument("--long_norm_scope", choices=["video", "dataset"], default="video", help="長視窗 z-score 正規化範圍")
	ap.add_argument("--all_features", action="store_true", help="使用全部特徵欄位 (排除 frame/time_sec)")
	ap.add_argument("--video_filter", nargs="*", help="只處理指定 video_id (以檔名前綴, 例如 1 2 15)")
	ap.add_argument("--export_npz", action="store_true", help="額外輸出 npz 壓縮檔")
	ap.add_argument("--enable_phase", action="store_true", help="啟用六階段 proxy 狀態機")
	ap.add_argument("--merge_smoke_segments", action="store_true", help="僅於 smoke 片段內重取短視窗 (減少冗餘)")
	ap.add_argument("--segment_pad", type=int, default=5, help="smoke 片段上下文 padding 幀數")
	ap.add_argument("--segment_inner_stride_factor", type=int, default=1, help="片段內 stride = win/因子")
	# phase 參數
	ap.add_argument("--phase_near_q", type=float, default=0.2, help="距離分位: near 閾值")
	ap.add_argument("--phase_far_q", type=float, default=0.6, help="距離分位: far 閾值")
	ap.add_argument("--phase_speed_z", type=float, default=0.8, help="速度 z-score 係數")
	ap.add_argument("--phase_min_frames", type=int, default=5, help="低於此幀數不進行 phase 推斷")
	args = ap.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	short_csv_path = os.path.join(args.output_dir, "short_windows.csv")
	long_csv_path = os.path.join(args.output_dir, "long_windows.csv")

	eigen_files = [f for f in os.listdir(args.eigen_dir) if f.endswith("_eig.csv")]
	if args.video_filter:
		filt_set = set(args.video_filter)
		eigen_files = [f for f in eigen_files if any(f.startswith(x + "_") or f.startswith(x + ".") or f.startswith(x) for x in filt_set)]
	if not eigen_files:
		print("⚠️ 沒有找到 eigenvalue 檔案")
		return

	# 第一輪: 載入影片並預備 phase
	per_video_stats = {}
	header_ref: Optional[List[str]] = None
	feat_cols: Optional[List[str]] = None
	video_frames_cache: Dict[str, List[Dict[str, float]]] = {}
	video_frame_labels_cache: Dict[str, List[str]] = {}
	video_phase_cache: Dict[str, List[int]] = {}
	video_dist_cache: Dict[str, List[float]] = {}
	video_speed_cache: Dict[str, List[float]] = {}
	print(f"[INFO] 準備統計 (long_norm_scope={args.long_norm_scope}) 影片數: {len(eigen_files)}")
	for fname in eigen_files:
		video_id = fname.split("_eig.csv")[0].split("_pose")[0]
		path = os.path.join(args.eigen_dir, fname)
		frames, header = load_eigen_csv(path)
		if not frames:
			continue
		if header_ref is None:
			header_ref = header
			feat_cols = select_feature_columns(header_ref, args.all_features)
			print(f"[INFO] 特徵欄位數: {len(feat_cols)} (all={args.all_features})")
		intervals = read_label_intervals(os.path.join(args.label_dir, f"{video_id}_full.csv"))
		frame_labels = build_frame_labels(frames, intervals)
		video_frames_cache[video_id] = frames
		video_frame_labels_cache[video_id] = frame_labels
		per_video_stats[video_id] = compute_video_stats(frames, feat_cols)  # type: ignore
		# Phase state machine (簡易)
		if args.enable_phase and len(frames) >= args.phase_min_frames:
			# 距離 (取左右手最小)
			d_series = []
			for fr in frames:
				dl = fr.get("dist_leftHand_mouth", math.nan)
				dr = fr.get("dist_rightHand_mouth", math.nan)
				best = dl
				if math.isnan(best) or (not math.isnan(dr) and dr < best):
					best = dr
				d_series.append(best)
			ts = [fr.get("time_sec", 0.0) for fr in frames]
			spd = [0.0]
			for i in range(1, len(d_series)):
				dt = (ts[i]-ts[i-1]) or 1e-6
				if math.isnan(d_series[i]) or math.isnan(d_series[i-1]):
					spd.append(0.0)
				else:
					spd.append((d_series[i]-d_series[i-1])/dt)
			valid_d = sorted([d for d in d_series if not math.isnan(d)])
			if valid_d:
				near_thr = valid_d[max(0,int(args.phase_near_q*len(valid_d))-1)]
				far_thr = valid_d[max(0,int(args.phase_far_q*len(valid_d))-1)]
			else:
				near_thr = far_thr = 0.0
			valid_s = [v for v in spd if not math.isnan(v)]
			if valid_s:
				mean_s = sum(valid_s)/len(valid_s)
				std_s = math.sqrt(sum((v-mean_s)**2 for v in valid_s)/max(1,len(valid_s)-1))
			else:
				std_s = 0.0
			approach_thr = -args.phase_speed_z * std_s
			leave_thr = args.phase_speed_z * std_s
			hold_thr = 0.3 * std_s
			phases = []
			state = 1
			for dval, v in zip(d_series, spd):
				if math.isnan(dval):
					phases.append(state)
					continue
				if state in (1,6):
					if v < approach_thr and dval > near_thr:
						state = 2
				elif state == 2:
					if dval <= near_thr:
						state = 3
				elif state == 3:
					if abs(v) <= hold_thr:
						state = 4
					elif v > leave_thr:
						state = 5
				elif state == 4:
					if v > leave_thr and dval > near_thr:
						state = 5
				elif state == 5:
					if dval >= far_thr and abs(v) <= hold_thr:
						state = 6
				phases.append(state)
			video_phase_cache[video_id] = phases
			video_dist_cache[video_id] = d_series
			video_speed_cache[video_id] = spd
		else:
			video_phase_cache[video_id] = [1]*len(frames)
			video_dist_cache[video_id] = [math.nan]*len(frames)
			video_speed_cache[video_id] = [0.0]*len(frames)

	if feat_cols is None:
		print("⚠️ 無有效特徵欄位, 終止")
		return

	if args.long_norm_scope == "dataset":
		dataset_stats = compute_dataset_stats_welford(video_frames_cache, feat_cols)
	else:
		dataset_stats = {}  # 每影片獨立, 使用 per_video_stats 內的

	# 準備寫出 CSV
	short_header_written: List[str] = []
	long_header_written: List[str] = []
	short_count = long_count = smoke_short = smoke_long = 0

	# 若啟用 merge smoke segments
	segment_map = {}
	if args.merge_smoke_segments:
		for vid, labs in video_frame_labels_cache.items():
			segments = []
			start=None
			for i,lab in enumerate(labs):
				if lab=='smoke' and start is None:
					start=i
				elif lab!='smoke' and start is not None:
					segments.append((start,i))
					start=None
			if start is not None:
				segments.append((start,len(labs)))
			pad=args.segment_pad
			padded=[]
			for s,e in segments:
				padded.append((max(0,s-pad), min(len(labs), e+pad)))
			segment_map[vid]=padded
	with open(short_csv_path, "w", newline="", encoding="utf-8") as fshort, \
		 open(long_csv_path, "w", newline="", encoding="utf-8") as flong:
		short_writer = csv.writer(fshort)
		long_writer = csv.writer(flong)
		# npz collect
		npz_collect = defaultdict(list) if args.export_npz else None
		for video_id in sorted(video_frames_cache.keys()):
			frames = video_frames_cache[video_id]
			labels = video_frame_labels_cache[video_id]
			long_stats = dataset_stats if args.long_norm_scope == "dataset" else per_video_stats[video_id]
			merge_cfg = None
			if args.merge_smoke_segments:
				merge_cfg = {"enable": True, "smoke_segments": segment_map.get(video_id, []), "inner_stride_factor": args.segment_inner_stride_factor}
			st, ss, lt, ls = process_video(
				video_id, frames, labels, feat_cols,
				args.short_win, args.short_stride, args.long_win, args.long_stride,
				args.smoke_ratio_thresh, long_stats,
				short_writer, long_writer, short_header_written, long_header_written,
				phases=video_phase_cache.get(video_id),
				dist_min_series=video_dist_cache.get(video_id),
				dist_speed_series=video_speed_cache.get(video_id),
				phase_enable=args.enable_phase,
				npz_collect=npz_collect,
				merge_config=merge_cfg,
			)
			short_count += st
			smoke_short += ss
			long_count += lt
			smoke_long += ls
		if args.export_npz and npz_collect is not None:
			import numpy as np
			npz_path = os.path.join(args.output_dir, "windows_npz.npz")
			npz_collect['feature_list'] = list(feat_cols)
			np.savez_compressed(npz_path, **{k: np.array(v, dtype=object) for k,v in npz_collect.items()})
			print(f"🗜️ NPZ 輸出: {npz_path}")

	# (已即時計數)

	stats_out_short = {
		"total_windows": short_count,
		"smoke_windows": smoke_short,
		"no_smoke_windows": short_count - smoke_short,
		"smoke_ratio_thresh": args.smoke_ratio_thresh,
		"win_size": args.short_win,
		"stride": args.short_stride,
		"norm_type": "per_window",
		"feature_count": len(feat_cols),
	}
	stats_out_long = {
		"total_windows": long_count,
		"smoke_windows": smoke_long,
		"no_smoke_windows": long_count - smoke_long,
		"smoke_ratio_thresh": args.smoke_ratio_thresh,
		"win_size": args.long_win,
		"stride": args.long_stride,
		"norm_type": f"{args.long_norm_scope}_zscore",
		"feature_count": len(feat_cols),
	}
	with open(os.path.join(args.output_dir, "stats_short.json"), "w", encoding="utf-8") as f:
		json.dump(stats_out_short, f, ensure_ascii=False, indent=2)
	with open(os.path.join(args.output_dir, "stats_long.json"), "w", encoding="utf-8") as f:
		json.dump(stats_out_long, f, ensure_ascii=False, indent=2)
	print("✅ 完成多尺度切片")
	print(f"[SHORT] windows={stats_out_short['total_windows']} smoke={stats_out_short['smoke_windows']} no_smoke={stats_out_short['no_smoke_windows']}")
	print(f"[LONG ] windows={stats_out_long['total_windows']} smoke={stats_out_long['smoke_windows']} no_smoke={stats_out_long['no_smoke_windows']}")
	print(f"輸出: {short_csv_path}, {long_csv_path}")


if __name__ == "__main__":
	main()
