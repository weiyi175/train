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
"""------------------------------------------------------------------------------------------------------------
功能概覽
多尺度視窗切割
短視窗：抓取快速動態（預設 win=30, stride=15），短視窗內做 z-score（window-level normalization）。
長視窗：捕捉完整動作循環（預設 win=75, stride=40），可選影片層或資料集層 z-score。
標籤與 smoke_ratio
以視窗內 smoke 幀比例與閾值（預設 0.5）決定 label=smoke/no_smoke；另輸出 smoke_ratio。
特徵欄位
預設取行為相關核心特徵（距離、速度、加速度等），可用 --all_features 改為全欄位（排除 frame/time_sec）。
六階段 proxy 狀態機（選用）
以距離分位與速度 z-score 簡易推斷階段（1–6），可輸出 phase 序列；長視窗另產出摘要指標。
長視窗摘要
approach_speed_peak、hold_frames、hold_seconds、leave_speed_peak（需 FPS）。
背景負樣本補齊（短視窗）
片段外抽樣（有保護帶 margin_sec）；支援策略 random/uniform、stride_override（可密集逐幀產生候選）、每影片上限、全資料集二階段 global fill 達成目標類別比例。
難負樣本（近鄰 hard negatives）
在 smoke core 片段附近（不重疊，距離由 margin_sec 控制）抽取負樣本，並以 weight 降低對訓練影響（預設 0.5）。
FPS 與秒數換算
--fps 未提供時，從 time_sec 差分自動估 fps，對長視窗輸出 fps、fps_estimated、fps_source 並換算 hold_seconds。
輸出
CSV：短/長視窗；含 meta、label、smoke_ratio、特徵序列、phase（選用）、長視窗摘要與 fps 欄位。
NPZ：物件版 windows_npz.npz。
Dense NPZ：windows_dense_npz.npz（float32）含 mask、meta 索引（video_id/start/end/…）與短視窗 weight。
可重現性
--seed 控制背景與難負抽樣隨機性。
附工具
analyze_windows.py：統計分析（含 weight 與 fps 統計）。
verify_dense_meta.py：隨機比對 dense NPZ meta 與 CSV 是否一致。
dataset_npz.py：PyTorch Dataset/DataLoader（on-the-fly temporal jitter、加權採樣）。
歷次重點修正（精簡）
背景負樣本補齊
加入背景抽樣（保護帶）、策略 random/uniform、per-video 上限。
新增 --short_bg_stride_override 以密集產生候選。
新增全域二階段補齊以達到資料集層級目標比例（global fill）。
FPS 與長視窗摘要強化
--fps；未提供則自動估（time_sec 差分→就近到 24/25/30/50/60）。
長視窗輸出 fps、fps_estimated、fps_source；hold_seconds 正確換算。
Dense NPZ 強化
由 object NPZ 另輸出稠密 float32（N,W,F）+ mask（N,W,F），另提供 mask_any（N,W）。
加入 meta indices：short/long 的 video_id/start_frame/end_frame 等，短視窗另含 weight。
難負樣本（hard negatives）
近鄰（不重疊）區域內抽取難負，標籤仍為 0，但以 weight 控制訓練影響。
Phase 與摘要
六階段序列與長視窗摘要（approach/hold/leave）。
其他
Welford 聚合（dataset-level z-score 選項）、seed 固定、分析器與驗證器腳本。
執行參數總表（含註解）
通用與 I/O

--eigen_dir：輸入 Eigenvalue 特徵目錄（預設 test_data/Eigenvalue）
--label_dir：VIA 標記目錄（*_full.csv）
--output_dir：輸出目錄
--video_filter [IDs…]：僅處理指定 video_id（前綴）
視窗與標籤

--short_win, --short_stride：短視窗大小與步長（預設 30/15）
--long_win, --long_stride：長視窗大小與步長（預設 75/40）
--smoke_ratio_thresh：判定 smoke 的比例閾值（預設 0.5）
特徵與正規化

--all_features：使用 CSV 全欄位（排除 frame/time_sec）
--long_norm_scope {video,dataset}：長視窗正規化範圍，影片層或資料集層（預設 video）
六階段 proxy（選用）

--enable_phase：啟用 phase 推斷與輸出
--phase_near_q, --phase_far_q：距離分位數，用於 near/far 閾值（預設 0.2/0.6）
--phase_speed_z：速度 z-score 係數（預設 0.8）
--phase_min_frames：低於此幀數不推斷（預設 5）
短視窗背景負樣本（True Negative）補齊

--short_bg_sample：啟用每影片的背景補樣
--short_bg_ratio：目標 no_smoke:smoke 比（短視窗，預設 1.0）
--short_bg_margin_sec：與 smoke 片段間的保護帶（秒，預設 1.0）
--short_bg_strategy {random,uniform}：候選挑選策略（預設 random）
--short_bg_max_per_video：每影片最多補的負樣本數
--short_bg_stride_override：背景候選的 stride 覆寫；1 為逐幀（0=沿用短視窗 stride）
短視窗背景全域補齊（Two-pass Global Fill）

--short_bg_global_fill：啟用全資料集層級的第二階段補齊
--short_bg_global_target_ratio：全資料集 no_smoke:smoke 目標比（預設 1.0）
--short_bg_global_strategy {random,uniform}：全域候選挑選策略
--short_bg_global_max_add：全域最多補的負樣本數
難負樣本（nearby hard negatives）

--hard_negative_nearby：啟用靠近 smoke 但不重疊的短視窗難負抽樣
--hard_negative_margin_sec：近鄰區距離（秒，預設 0.5）
--hard_negative_ratio：新增難負樣本的上限比例，對目前「負樣本數」的比例（預設 0.1 = 10%）
--hard_negative_weight：難負樣本的 sample weight（預設 0.5）
FPS 與秒數換算

--fps：影片 FPS；為 0 時自動以 time_sec 差分估計並四捨五入到常見 fps 值
輸出格式

--export_npz：輸出物件陣列 NPZ（windows_npz.npz）
--export_npz_dense：輸出稠密 float32 NPZ（windows_dense_npz.npz），含 mask 與完整 meta
重現性

--seed：隨機種子（背景/難負抽樣）
輸出檔案與欄位
short_windows.csv
基本：video_id, scale, window_index, start_frame, end_frame, start_time, end_time, label, smoke_ratio, n_frames, weight
序列：各特徵 raw_t0..tW-1、norm_t0..tW-1
phase（若啟用）：phase_t0..tW-1
long_windows.csv
基本：同上，另含 fps, fps_estimated, fps_source
序列：各特徵 raw/norm 時序
phase（若啟用）：phase_t*
摘要：approach_speed_peak, hold_frames, hold_seconds, leave_speed_peak
windows_dense_npz.npz（主要鍵）
共用：feature_list
short：short_raw、short_norm（N,W,F）、short_mask（N,W,F）、short_mask_any（N,W）、short_label、short_smoke_ratio、short_video_id、short_start_frame、short_end_frame、short_weight、short_phase（可選）
long：long_raw、long_norm（N,W,F）、long_mask（N,W,F）、long_mask_any（N,W）、long_label、long_smoke_ratio、long_video_id、long_start_frame、long_end_frame、long_fps、long_fps_estimated、long_fps_source、long_phase（可選）、long_approach_speed_peak、long_hold_frames、long_hold_seconds、long_leave_speed_peak
windows_npz.npz（object 版本）
對應上述資料，型別為 object array（利於檢查）
video_fps_map.json
各影片 fps、estimated 與 source
訓練建議（搭配附工具）
權重使用
短視窗已提供 sample weight（hard negatives 權重較低）；loss 乘上該權重或使用 WeightedRandomSampler。
Dataset/DataLoader
dataset_npz.py 提供 WindowsNPZDataset 與 build_dataloader（需安裝 PyTorch）；支援 on-the-fly temporal jitter（±k 幀）與同時使用類別平衡與權重。
分析與驗證
analyze_windows.py：快速檢查類別分佈、smoke_ratio、phase、長視窗摘要與 fps。
verify_dense_meta.py：核對 dense NPZ meta 與 CSV 清單一致性（回溯與 overlay 需要）。
"""
from __future__ import annotations

import os
import csv
import math
import argparse
import json
from collections import defaultdict
import random
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
	short_bg_cfg: Optional[Dict] = None,
	# hard negative nearby sampling config
	hard_neg_cfg: Optional[Dict] = None,
	# for global fill: expose candidate pools and locally used picks
	bg_pool_map: Optional[Dict[str, List[Tuple[int,int]]]] = None,
	bg_used_map: Optional[Dict[str, List[Tuple[int,int]]]] = None,
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

	# helper: write one short window row and npz collect
	def write_short_window(s: int, e: int, short_index: int, *, weight: float = 1.0) -> Tuple[int, int]:
		sub = frames[s:e]
		labs = frame_labels[s:e]
		smoke_frames = sum(1 for L in labs if L == "smoke")
		ratio = smoke_frames / len(labs)
		label = "smoke" if ratio >= smoke_ratio_thresh else "no_smoke"
		per_stats = compute_video_stats(sub, feat_cols)
		row_meta = [video_id, "short", short_index, sub[0].get("frame", 0), sub[-1].get("frame", 0),
					f"{sub[0].get('time_sec',0.0):.6f}", f"{sub[-1].get('time_sec',0.0):.6f}", label, f"{ratio:.4f}", len(sub), f"{float(weight):.4f}"]
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
			header = ["video_id","scale","window_index","start_frame","end_frame","start_time","end_time","label","smoke_ratio","n_frames","weight"]
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
			npz_collect.setdefault("short_weight", []).append(float(weight))
			if phase_seq:
				npz_collect.setdefault("short_phase", []).append([int(p) for p in phase_seq])
		return (1, 1 if label=="smoke" else 0)

	# short windows
	short_index = 0
	for s, e in short_ranges():
		w_added, w_smoke = write_short_window(s, e, short_index, weight=1.0)
		short_total += w_added
		short_smoke += w_smoke
		short_index += w_added

	# optional: background candidate pool and local per-video fill
	if short_bg_cfg:
		# segments without padding from merge_config if provided; otherwise derive from labels
		base_segments = []
		if merge_config and merge_config.get("smoke_segments_core"):
			base_segments = merge_config["smoke_segments_core"]
		else:
			start=None
			for i,lab in enumerate(frame_labels):
				if lab=='smoke' and start is None:
					start=i
				elif lab!='smoke' and start is not None:
					base_segments.append((start,i))
					start=None
			if start is not None:
				base_segments.append((start,len(frame_labels)))
		margin_sec = float(short_bg_cfg.get("margin_sec", 1.0))
		bg_stride = int(short_bg_cfg.get("stride_override", 0)) or short_stride
		bg_stride = max(1, bg_stride)
		def window_far_from_segments(si:int, ei:int) -> bool:
			ws = frames[si].get('time_sec',0.0)
			we = frames[ei-1].get('time_sec',0.0)
			for ss,ee in base_segments:
				ts = frames[ss].get('time_sec',0.0)
				te = frames[ee-1].get('time_sec',0.0) if ee-1 < len(frames) else frames[-1].get('time_sec',0.0)
				# overlap with margin?
				if not ((we + margin_sec) <= ts or (ws - margin_sec) >= te):
					return False
			return True
		cands: List[Tuple[int,int]] = []
		for s,e in iter_sliding_windows(n_frames, short_win, bg_stride):
			if not window_far_from_segments(s,e):
				continue
			labs = frame_labels[s:e]
			smoke_frames = sum(1 for L in labs if L == 'smoke')
			ratio = smoke_frames/len(labs)
			if ratio < smoke_ratio_thresh:
				cands.append((s,e))
		# expose candidate pool for global fill
		if bg_pool_map is not None:
			bg_pool_map.setdefault(video_id, []).extend(cands)
		# local per-video fill if enabled
		if short_bg_cfg.get("enable"):
			target_ratio = float(short_bg_cfg.get("ratio", 1.0))
			strategy = short_bg_cfg.get("strategy", "random")
			max_add = int(short_bg_cfg.get("max_add", 10**9))
			curr_neg = short_total - short_smoke
			need_neg = int(math.ceil(short_smoke * target_ratio))
			deficit = max(0, need_neg - curr_neg)
			deficit = min(deficit, max_add)
			if deficit > 0 and cands:
				if strategy == 'uniform':
					idxs = list(range(len(cands)))
					if deficit >= len(idxs):
						pick = idxs
					else:
						step = len(idxs)/deficit
						pick = [int(i*step) for i in range(deficit)]
				else:
					pick = random.sample(range(len(cands)), k=min(deficit, len(cands)))
				for pi in pick:
					s,e = cands[pi]
					w_added, w_smoke = write_short_window(s, e, short_index, weight=1.0)
					short_total += w_added
					short_smoke += w_smoke
					short_index += w_added
					if bg_used_map is not None:
						bg_used_map.setdefault(video_id, []).append((s,e))

	# optional: hard negative nearby sampling (short windows only)
	if hard_neg_cfg:
		# build smoke core segments
		base_segments = []
		if merge_config and merge_config.get("smoke_segments_core"):
			base_segments = merge_config["smoke_segments_core"]
		else:
			start=None
			for i,lab in enumerate(frame_labels):
				if lab=='smoke' and start is None:
					start=i
				elif lab!='smoke' and start is not None:
					base_segments.append((start,i))
					start=None
			if start is not None:
				base_segments.append((start,len(frame_labels)))
		margin_sec = float(hard_neg_cfg.get("margin_sec", 0.5))
		# near if overlaps padded segment [ss-margin, ee+margin] but not the core [ss,ee]
		def is_near_not_overlap(si:int, ei:int) -> bool:
			ws = frames[si].get('time_sec',0.0)
			we = frames[ei-1].get('time_sec',0.0)
			for ss,ee in base_segments:
				ts = frames[ss].get('time_sec',0.0)
				te = frames[ee-1].get('time_sec',0.0) if ee-1 < len(frames) else frames[-1].get('time_sec',0.0)
				# padded interval
				ps = max(0.0, ts - margin_sec)
				pe = te + margin_sec
				# overlap with padded?
				over_padded = not ((we) <= ps or (ws) >= pe)
				# overlap with core?
				over_core = not ((we) <= ts or (ws) >= te)
				if over_padded and (not over_core):
					return True
			return False
		near_cands: List[Tuple[int,int]] = []
		for s,e in iter_sliding_windows(n_frames, short_win, max(1, short_stride)):
			if not is_near_not_overlap(s,e):
				continue
			labs = frame_labels[s:e]
			smoke_frames = sum(1 for L in labs if L=='smoke')
			ratio = smoke_frames/len(labs)
			if ratio < smoke_ratio_thresh:  # ensure labeled negative
				near_cands.append((s,e))
		# how many to add
		curr_neg = short_total - short_smoke
		max_to_add = int(math.floor(curr_neg * float(hard_neg_cfg.get("ratio", 0.1))))
		if max_to_add > 0 and near_cands:
			k = min(max_to_add, len(near_cands))
			pick = random.sample(range(len(near_cands)), k=k)
			w = float(hard_neg_cfg.get("weight", 0.5))
			for pi in pick:
				s,e = near_cands[pi]
				w_added, w_smoke = write_short_window(s, e, short_index, weight=w)
				short_total += w_added
				short_smoke += w_smoke
				short_index += w_added

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
		# fps meta for long rows
		fps_val =  float(merge_config.get('fps', 0.0)) if merge_config else 0.0
		fps_estimated = bool(merge_config.get('fps_estimated', False)) if merge_config else False
		fps_source = str(merge_config.get('fps_source', 'explicit')) if merge_config else 'explicit'
		row_meta += [f"{fps_val:.3f}", str(int(fps_estimated)), fps_source]
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
		approach_speed_peak = hold_duration = hold_seconds = leave_speed_peak = 0.0
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
					# seconds via global CLI fps (filled later in main via closure or config)
		row = row_meta + seq_raw + seq_norm
		if phase_enable:
			row += phase_seq
			# hold_seconds will be computed in main via provided fps, passed in via long_norm_stats? No, add via long_meta_fps in merge_config
			row += [f"{approach_speed_peak:.6f}"]
			# place holders; we'll compute using fps_cfg in merge_config
			fps_val =  float(merge_config.get('fps', 0.0)) if merge_config else 0.0
			if fps_val and fps_val > 0:
				hold_seconds = hold_duration / fps_val
			row += [f"{hold_duration:.0f}", f"{hold_seconds:.6f}", f"{leave_speed_peak:.6f}"]
		if not long_header_ref:
			header = ["video_id","scale","window_index","start_frame","end_frame","start_time","end_time","label","smoke_ratio","n_frames","fps","fps_estimated","fps_source"]
			for fcol in feat_cols:
				header.extend([f"{fcol}_raw_t{i}" for i in range(long_win)])
			for fcol in feat_cols:
				header.extend([f"{fcol}_norm_t{i}" for i in range(long_win)])
			if phase_enable:
				header.extend([f"phase_t{i}" for i in range(long_win)])
				header.extend(["approach_speed_peak","hold_frames","hold_seconds","leave_speed_peak"])
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
				npz_collect.setdefault("long_hold_seconds", []).append(hold_seconds)
				npz_collect.setdefault("long_leave_speed_peak", []).append(leave_speed_peak)
			npz_collect.setdefault("long_fps", []).append(float(fps_val))
			npz_collect.setdefault("long_fps_estimated", []).append(1 if fps_estimated else 0)
			npz_collect.setdefault("long_fps_source", []).append(fps_source)
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
	# short background sampling
	ap.add_argument("--short_bg_sample", action="store_true", help="額外抽取短視窗背景 (no_smoke) 以達到目標比例")
	ap.add_argument("--short_bg_ratio", type=float, default=1.0, help="目標 no_smoke:smoke 比例 (短視窗)")
	ap.add_argument("--short_bg_margin_sec", type=float, default=1.0, help="背景視窗與 smoke 片段的時間保護帶 (秒)")
	ap.add_argument("--short_bg_strategy", choices=["random","uniform"], default="random", help="背景抽樣策略")
	ap.add_argument("--short_bg_max_per_video", type=int, default=1000000, help="每支影片最多補多少負樣本")
	ap.add_argument("--short_bg_stride_override", type=int, default=0, help="背景候選 stride 覆寫 (1 表逐幀)，0=沿用短視窗 stride")
	# Global fill options
	ap.add_argument("--short_bg_global_fill", action="store_true", help="啟用全資料集層級第二階段背景補樣")
	ap.add_argument("--short_bg_global_target_ratio", type=float, default=1.0, help="全資料集目標 no_smoke:smoke 比例")
	ap.add_argument("--short_bg_global_strategy", choices=["random","uniform"], default="random", help="全域補樣策略")
	ap.add_argument("--short_bg_global_max_add", type=int, default=1000000000, help="全域最多補多少負樣本")
	# hard negative nearby sampling
	ap.add_argument("--hard_negative_nearby", action="store_true", help="針對 smoke 片段附近 (不重疊) 抽取難負樣本")
	ap.add_argument("--hard_negative_margin_sec", type=float, default=0.5, help="nearby 區域 (秒) 與 smoke 片段的距離")
	ap.add_argument("--hard_negative_ratio", type=float, default=0.1, help="難負樣本占目前負樣本比例上限，如 0.1 表 10%%")
	ap.add_argument("--hard_negative_weight", type=float, default=0.5, help="難負樣本的訓練權重")
	# phase 參數
	ap.add_argument("--phase_near_q", type=float, default=0.2, help="距離分位: near 閾值")
	ap.add_argument("--phase_far_q", type=float, default=0.6, help="距離分位: far 閾值")
	ap.add_argument("--phase_speed_z", type=float, default=0.8, help="速度 z-score 係數")
	ap.add_argument("--phase_min_frames", type=int, default=5, help="低於此幀數不進行 phase 推斷")
	# fps for converting hold frames to seconds
	ap.add_argument("--fps", type=float, default=0.0, help="影片 FPS。若為 0 則嘗試由 time_sec 估算")
	# dense npz export
	ap.add_argument("--export_npz_dense", action="store_true", help="同時輸出稠密 float32 NPZ 與 mask (N,T,F)")
	# reproducibility
	ap.add_argument("--seed", type=int, default=42, help="隨機種子，用於背景抽樣重現")
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
	# seed
	random.seed(args.seed)

	def estimate_fps_for_video(frames: List[Dict[str,float]]) -> Tuple[float, bool, str]:
		"""Estimate FPS from time_sec diffs; return (fps, estimated_flag, source_str)."""
		if args.fps and args.fps > 0:
			return float(args.fps), False, "explicit"
		ts = [fr.get('time_sec', math.nan) for fr in frames]
		diffs = []
		for i in range(1, len(ts)):
			if not math.isnan(ts[i]) and not math.isnan(ts[i-1]):
				dt = ts[i]-ts[i-1]
				if dt > 1e-6:
					diffs.append(dt)
		if not diffs:
			# fallback
			return 30.0, True, "estimated"
		mean_dt = sum(diffs)/len(diffs)
		fps_raw = 1.0/mean_dt if mean_dt>0 else 30.0
		common = [24,25,30,50,60]
		fps = min(common, key=lambda x: abs(x - fps_raw))
		return float(fps), True, "estimated"
	video_fps_map: Dict[str, Dict[str, float|str|bool]] = {}
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
		# estimate fps if needed
		fps_val, fps_est, fps_src = estimate_fps_for_video(frames)
		video_fps_map[video_id] = {"fps": fps_val, "estimated": fps_est, "source": fps_src}
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
	segment_core_map = {}
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
			segment_core_map[vid] = list(segments)
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
		# pools for global fill
		bg_pool_map: Dict[str, List[Tuple[int,int]]] = {}
		bg_used_map: Dict[str, List[Tuple[int,int]]] = {}
		short_written_by_video: Dict[str, int] = {}
		for video_id in sorted(video_frames_cache.keys()):
			frames = video_frames_cache[video_id]
			labels = video_frame_labels_cache[video_id]
			long_stats = dataset_stats if args.long_norm_scope == "dataset" else per_video_stats[video_id]
			merge_cfg = None
			if args.merge_smoke_segments:
				merge_cfg = {"enable": True, "smoke_segments": segment_map.get(video_id, []), "smoke_segments_core": segment_core_map.get(video_id, []), "inner_stride_factor": args.segment_inner_stride_factor, "fps": video_fps_map[video_id]["fps"], "fps_estimated": video_fps_map[video_id]["estimated"], "fps_source": video_fps_map[video_id]["source"]}
			else:
				merge_cfg = {"enable": False, "fps": video_fps_map[video_id]["fps"], "fps_estimated": video_fps_map[video_id]["estimated"], "fps_source": video_fps_map[video_id]["source"]}
			short_bg_cfg = None
			if args.short_bg_sample or args.short_bg_global_fill:
				short_bg_cfg = {"enable": bool(args.short_bg_sample), "ratio": args.short_bg_ratio, "margin_sec": args.short_bg_margin_sec, "strategy": args.short_bg_strategy, "max_add": args.short_bg_max_per_video, "stride_override": args.short_bg_stride_override}
			hard_neg_cfg = None
			if args.hard_negative_nearby:
				hard_neg_cfg = {"margin_sec": args.hard_negative_margin_sec, "ratio": args.hard_negative_ratio, "weight": args.hard_negative_weight}
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
				short_bg_cfg=short_bg_cfg,
				hard_neg_cfg=hard_neg_cfg,
				bg_pool_map=(bg_pool_map if args.short_bg_global_fill else None),
				bg_used_map=(bg_used_map if args.short_bg_global_fill and args.short_bg_sample else None),
			)
			short_count += st
			smoke_short += ss
			short_written_by_video[video_id] = st
			long_count += lt
			smoke_long += ls

		# Global fill (two-pass)
		if args.short_bg_global_fill:
			# gather global pool excluding used
			global_pool: List[Tuple[str,int,int]] = []
			for vid, cands in bg_pool_map.items():
				used = set(bg_used_map.get(vid, []))
				for s,e in cands:
					if (s,e) not in used:
						global_pool.append((vid,s,e))
			current_neg = short_count - smoke_short
			need_neg = int(math.ceil(smoke_short * args.short_bg_global_target_ratio))
			deficit = max(0, need_neg - current_neg)
			deficit = min(deficit, args.short_bg_global_max_add)
			if deficit > 0 and global_pool:
				if args.short_bg_global_strategy == 'uniform':
					idxs = list(range(len(global_pool)))
					if deficit >= len(idxs):
						pick_idxs = idxs
					else:
						step = len(idxs)/deficit
						pick_idxs = [int(i*step) for i in range(deficit)]
				else:
					pick_idxs = random.sample(range(len(global_pool)), k=min(deficit, len(global_pool)))
				# write picks
				for pi in pick_idxs:
					vid,s,e = global_pool[pi]
					frames = video_frames_cache[vid]
					labels = video_frame_labels_cache[vid]
					phases_seq = video_phase_cache.get(vid)
					sub = frames[s:e]
					labs = labels[s:e]
					smoke_frames = sum(1 for L in labs if L == 'smoke')
					ratio = smoke_frames/len(labs)
					label = 'smoke' if ratio >= args.smoke_ratio_thresh else 'no_smoke'
					per_stats = compute_video_stats(sub, feat_cols)
					row_meta = [vid,'short', short_written_by_video.get(vid,0), sub[0].get('frame',0), sub[-1].get('frame',0), f"{sub[0].get('time_sec',0.0):.6f}", f"{sub[-1].get('time_sec',0.0):.6f}", label, f"{ratio:.4f}", len(sub), f"{1.0:.4f}"]
					seq_raw=[]; seq_norm=[]
					for fcol in feat_cols:
						mean,std = per_stats[fcol]
						for fr in sub:
							v = fr.get(fcol, math.nan)
							seq_raw.append('' if math.isnan(v) else f"{v:.6f}")
							nv = 0.0 if math.isnan(v) else zscore(v, mean, std)
							seq_norm.append(f"{nv:.6f}")
					phase_seq=[]
					if args.enable_phase and phases_seq is not None:
						phase_seq=[str(p) for p in phases_seq[s:e]]
					row = row_meta + seq_raw + seq_norm + phase_seq
					if not short_header_written:
						header = ["video_id","scale","window_index","start_frame","end_frame","start_time","end_time","label","smoke_ratio","n_frames","weight"]
						for fcol in feat_cols:
							header.extend([f"{fcol}_raw_t{i}" for i in range(args.short_win)])
						for fcol in feat_cols:
							header.extend([f"{fcol}_norm_t{i}" for i in range(args.short_win)])
						if args.enable_phase:
							header.extend([f"phase_t{i}" for i in range(args.short_win)])
						short_writer.writerow(header)
						short_header_written.extend(header)
					short_writer.writerow(row)
					if npz_collect is not None:
						W = args.short_win
						F = len(feat_cols)
						raw_vals=[]; norm_vals=[]
						for fi in range(F):
							start = fi*W
							raw_vals.append([float(x) if x!='' else math.nan for x in seq_raw[start:start+W]])
						for fi in range(F):
							start = fi*W
							norm_vals.append([float(x) for x in seq_norm[start:start+W]])
						npz_collect.setdefault('short_raw', []).append(raw_vals)
						npz_collect.setdefault('short_norm', []).append(norm_vals)
						npz_collect.setdefault('short_label', []).append(1 if label=='smoke' else 0)
						npz_collect.setdefault('short_smoke_ratio', []).append(ratio)
						npz_collect.setdefault('short_video_id', []).append(vid)
						npz_collect.setdefault('short_start_frame', []).append(sub[0].get('frame',0))
						npz_collect.setdefault('short_end_frame', []).append(sub[-1].get('frame',0))
						npz_collect.setdefault('short_weight', []).append(1.0)
						if phase_seq:
							npz_collect.setdefault('short_phase', []).append([int(p) for p in phase_seq])
					short_count += 1
					if label=='smoke':
						smoke_short += 1
					short_written_by_video[vid] = short_written_by_video.get(vid,0) + 1
		if args.export_npz and npz_collect is not None:
			import numpy as np
			npz_path = os.path.join(args.output_dir, "windows_npz.npz")
			npz_collect['feature_list'] = list(feat_cols)
			np.savez_compressed(npz_path, **{k: np.array(v, dtype=object) for k,v in npz_collect.items()})
			print(f"🗜️ NPZ 輸出: {npz_path}")
			# Optional dense export
			if args.export_npz_dense:
				def to_dense(arr_list):
					if not arr_list:
						return None, None
					A = np.asarray(arr_list, dtype=float)  # (N,F,W)
					# mask from raw (finite)
					mask = np.isfinite(A)
					A = A.transpose(0,2,1).astype(np.float32)  # (N,W,F)
					mask = mask.transpose(0,2,1)
					return A, mask
				short_raw_dense, short_mask = to_dense(npz_collect.get('short_raw', []))
				short_norm_dense, _ = to_dense(npz_collect.get('short_norm', []))
				long_raw_dense, long_mask = to_dense(npz_collect.get('long_raw', []))
				long_norm_dense, _ = to_dense(npz_collect.get('long_norm', []))
				dense_path = os.path.join(args.output_dir, "windows_dense_npz.npz")
				out = {"feature_list": np.array(feat_cols, dtype=object)}
				if short_raw_dense is not None:
					out.update({
						"short_raw": short_raw_dense,
						"short_norm": short_norm_dense,
						"short_mask": short_mask,
						"short_mask_any": short_mask.any(axis=2) if short_mask is not None else None,
						"short_label": np.array(npz_collect.get('short_label', []), dtype=np.int64),
						"short_smoke_ratio": np.array(npz_collect.get('short_smoke_ratio', []), dtype=np.float32),
						"short_video_id": np.array(npz_collect.get('short_video_id', []), dtype=object),
						"short_start_frame": np.array(npz_collect.get('short_start_frame', []), dtype=np.int64),
						"short_end_frame": np.array(npz_collect.get('short_end_frame', []), dtype=np.int64),
						"short_weight": np.array(npz_collect.get('short_weight', []), dtype=np.float32),
					})
				if long_raw_dense is not None:
					out.update({
						"long_raw": long_raw_dense,
						"long_norm": long_norm_dense,
						"long_mask": long_mask,
						"long_mask_any": long_mask.any(axis=2) if long_mask is not None else None,
						"long_label": np.array(npz_collect.get('long_label', []), dtype=np.int64),
						"long_smoke_ratio": np.array(npz_collect.get('long_smoke_ratio', []), dtype=np.float32),
						"long_video_id": np.array(npz_collect.get('long_video_id', []), dtype=object),
						"long_start_frame": np.array(npz_collect.get('long_start_frame', []), dtype=np.int64),
						"long_end_frame": np.array(npz_collect.get('long_end_frame', []), dtype=np.int64),
						"long_fps": np.array(npz_collect.get('long_fps', []), dtype=np.float32),
						"long_fps_estimated": np.array(npz_collect.get('long_fps_estimated', []), dtype=np.int64),
					})
				# phases and summaries if available
				if 'short_phase' in npz_collect:
					out['short_phase'] = np.array(npz_collect['short_phase'], dtype=np.int16)
				if 'long_phase' in npz_collect:
					out['long_phase'] = np.array(npz_collect['long_phase'], dtype=np.int16)
				if 'long_approach_speed_peak' in npz_collect:
					out['long_approach_speed_peak'] = np.array(npz_collect['long_approach_speed_peak'], dtype=np.float32)
				if 'long_hold_duration' in npz_collect:
					out['long_hold_frames'] = np.array(npz_collect['long_hold_duration'], dtype=np.float32)
				if 'long_hold_seconds' in npz_collect:
					out['long_hold_seconds'] = np.array(npz_collect['long_hold_seconds'], dtype=np.float32)
				if 'long_leave_speed_peak' in npz_collect:
					out['long_leave_speed_peak'] = np.array(npz_collect['long_leave_speed_peak'], dtype=np.float32)
				np.savez_compressed(dense_path, **out)
				print(f"🧱 Dense NPZ 輸出: {dense_path}")

	# save fps map for traceability
	with open(os.path.join(args.output_dir, "video_fps_map.json"), "w", encoding="utf-8") as f:
		json.dump({vid: {"fps": float(info["fps"]), "estimated": bool(info["estimated"]), "source": info["source"]} for vid, info in video_fps_map.items()}, f, ensure_ascii=False, indent=2)

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
