"""Feature (Eigenvalue) extraction from pose CSVs.

輸入來源目錄: /home/user/projects/train/test_data/extract_pose
輸出目錄: /home/user/projects/train/test_data/Eigenvalue

對每支 *_pose.csv 逐幀計算:
1. 距離特徵: 手-嘴、手-手、鼻-手、手腕距離等
2. 速度: 已有 (vx,vy,vz) -> 直接複製輸出
3. 加速度: a = Δv (本程式計算)
4. 臉部方向估計: 以眼睛群中心線與鼻子相對位置估 yaw(-1~1)
5. 嘴巴點位補償: 若嘴部可見度不足, 用眼睛中點 + 上一穩定 mouth offset 推估
6. 嘴部置信度調整: nose 低 + 推估 / 手遮擋 等情況修正

使用的關鍵 Mediapipe 索引 (Pose):
0 nose, 1~3 左眼群, 4~6 右眼群, 9 mouth_left, 10 mouth_right,
11 左肩,12 右肩,13 左肘,14 右肘,15 左腕,16 右腕,17~22 手指/拇指 (左 17,19,21; 右 18,20,22)

限制: 僅使用已抽取的子集合 (0~6, 9~22)。
"""

from __future__ import annotations

import os
import csv
import math
from typing import Dict, List, Tuple, Optional

POSE_INPUT_DIR = "/home/user/projects/train/train_data/extract_pose"
FEATURE_OUTPUT_DIR = "/home/user/projects/train/train_data/Eigenvalue"

# 選取骨架點 (與抽取階段一致)
SELECTED_LANDMARKS = list(range(0, 7)) + list(range(9, 23))

# 重要索引
NOSE = 0
LEFT_EYE_GROUP = [1, 2, 3]
RIGHT_EYE_GROUP = [4, 5, 6]
MOUTH_L = 9
MOUTH_R = 10
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_INDEX = 19
RIGHT_INDEX = 20

# 門檻/常數
VIS_THRESH_NOSE = 0.5
VIS_THRESH_MOUTH = 0.5
YAW_LEFT = -0.2   # yaw < -0.2 視為朝左
YAW_RIGHT = 0.2   # yaw > 0.2 視為朝右
HAND_MOUTH_OCCLUDE_DIST = 0.08  # 正規化座標距離
DEFAULT_MOUTH_OFFSET_Y = 0.07   # 眼睛中點到嘴中心大致垂直偏移
VELOCITY_JUMP_V_THRESH = 3.0    # 基礎速度跳變門檻 (單一分量)
ACC_JUMP_A_THRESH = 6.0         # 基礎加速度跳變門檻 (單一分量)
# Robust 門檻 (以全體幀的中位數 + k*IQR)
ROBUST_V_IQR_K = 3.0
ROBUST_A_IQR_K = 3.0
# 一幀中需達到異常的 landmark 數量下限
JUMP_MIN_LANDMARKS = 3

# 距離欄位定義 (成對點) 供 2D/3D 計算
DISTANCE_PAIRS = {
	"leftHand_mouth": (LEFT_INDEX, MOUTH_L),  # mouth 最終會用 center; 這裡僅占位
	"rightHand_mouth": (RIGHT_INDEX, MOUTH_R),
	"hands": (LEFT_INDEX, RIGHT_INDEX),
	"nose_leftHand": (NOSE, LEFT_INDEX),
	"nose_rightHand": (NOSE, RIGHT_INDEX),
	"wrists": (LEFT_WRIST, RIGHT_WRIST),
}


def _safe_float(s: str) -> float:
	if s is None or s == "":
		return math.nan
	try:
		return float(s)
	except Exception:
		return math.nan


class FrameData:
	def __init__(self, frame: int, t: float):
		self.frame = frame
		self.time = t
		# landmark -> (x,y,z,v,vx,vy,vz)
		self.landmarks: Dict[int, Tuple[float, float, float, float, float, float, float]] = {}


def read_pose_csv(path: str) -> List[FrameData]:
	frames: List[FrameData] = []
	with open(path, "r", encoding="utf-8") as f:
		reader = csv.reader(f)
		header = next(reader, None)
		if not header:
			return frames
		# 建立欄位對應: l{idx}_x ...
		col_map = {name: i for i, name in enumerate(header)}
		for row in reader:
			if not row:
				continue
			frame_idx = int(float(row[col_map.get("frame", 0)]))
			t = float(row[col_map.get("time_sec", 1)])
			fd = FrameData(frame_idx, t)
			for idx in SELECTED_LANDMARKS:
				base = f"l{idx}_"
				x = _safe_float(row[col_map.get(base + "x", -1)]) if (base + "x") in col_map else math.nan
				y = _safe_float(row[col_map.get(base + "y", -1)]) if (base + "y") in col_map else math.nan
				z = _safe_float(row[col_map.get(base + "z", -1)]) if (base + "z") in col_map else math.nan
				v = _safe_float(row[col_map.get(base + "v", -1)]) if (base + "v") in col_map else math.nan
				vx = _safe_float(row[col_map.get(base + "vx", -1)]) if (base + "vx") in col_map else math.nan
				vy = _safe_float(row[col_map.get(base + "vy", -1)]) if (base + "vy") in col_map else math.nan
				vz = _safe_float(row[col_map.get(base + "vz", -1)]) if (base + "vz") in col_map else math.nan
				fd.landmarks[idx] = (x, y, z, v, vx, vy, vz)
			frames.append(fd)
	return frames


def _avg_points(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
	xs = [p[0] for p in points if not any(math.isnan(c) for c in p)]
	ys = [p[1] for p in points if not any(math.isnan(c) for c in p)]
	zs = [p[2] for p in points if not any(math.isnan(c) for c in p)]
	if not xs or not ys or not zs:
		return (math.nan, math.nan, math.nan)
	return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


def _dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
	if any(math.isnan(c) for c in a + b):
		return math.nan
	return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _dist2(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
	if any(math.isnan(c) for c in (a[0], a[1], b[0], b[1])):
		return math.nan
	return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def compute_features(frames: List[FrameData]) -> List[Dict[str, float]]:
	"""兩段流程:
	1. 初步計算所有特徵 (不決定 jump flag)
	2. 基於所有幀的速度/加速度 robust 統計建立門檻, 再計算 jump flag
	"""
	results: List[Dict[str, float]] = []
	last_vel: Dict[int, Tuple[float, float, float]] = {}
	last_mouth_offset: Optional[Tuple[float, float, float]] = None
	last_mouth_pos: Optional[Tuple[float, float, float]] = None
	last_mouth_vel: Optional[Tuple[float, float, float]] = None

	for i, fd in enumerate(frames):
		feat: Dict[str, float] = {"frame": fd.frame, "time_sec": fd.time}

		# 基本取點
		def get_point(idx: int):
			if idx not in fd.landmarks:
				return (math.nan, math.nan, math.nan, math.nan)
			x, y, z, v, _, _, _ = fd.landmarks[idx]
			return (x, y, z, v)

		def get_vel(idx: int):
			if idx not in fd.landmarks:
				return (math.nan, math.nan, math.nan)
			_, _, _, _, vx, vy, vz = fd.landmarks[idx]
			return (vx, vy, vz)

		# 眼睛中心
		l_eye_pts = [get_point(idx)[:3] for idx in LEFT_EYE_GROUP]
		r_eye_pts = [get_point(idx)[:3] for idx in RIGHT_EYE_GROUP]
		left_eye_center = _avg_points(l_eye_pts)
		right_eye_center = _avg_points(r_eye_pts)
		nose_xyzv = get_point(NOSE)
		nose_pos = nose_xyzv[:3]
		nose_vis = nose_xyzv[3]

		# yaw 估計
		yaw_raw_ratio = math.nan  # nose 在眼睛線段上的原始比例 (0~1 期望, 可能超界)
		yaw = math.nan
		face_class = 0
		if not any(math.isnan(c) for c in left_eye_center + right_eye_center) and not any(math.isnan(c) for c in nose_pos):
			# 用 x 座標比例
			lx = left_eye_center[0]
			rx = right_eye_center[0]
			if rx != lx:
				ratio = (nose_pos[0] - lx) / (rx - lx)
				yaw_raw_ratio = ratio
				# ratio ~ 0 nose 靠左眼, ~1 nose 靠右眼; 轉換為 -1~1 後再 clamp
				yaw = (ratio - 0.5) * 2.0
				yaw = max(-1.0, min(1.0, yaw))
				if yaw < YAW_LEFT:
					face_class = -1
				elif yaw > YAW_RIGHT:
					face_class = 1
				else:
					face_class = 0
		yaw_valid_flag = 0 if math.isnan(yaw) else 1
		feat["yaw_est"] = yaw
		feat["yaw_raw_ratio"] = yaw_raw_ratio
		feat["yaw_valid_flag"] = yaw_valid_flag
		feat["face_orient_class"] = face_class

		# 嘴部原始位置/可見度
		mouth_l = get_point(MOUTH_L)
		mouth_r = get_point(MOUTH_R)
		mouth_vis_vals = [mouth_l[3], mouth_r[3]]
		mouth_valid_vals = [v for v in mouth_vis_vals if not math.isnan(v)]
		raw_mouth_conf = sum(mouth_valid_vals)/len(mouth_valid_vals) if mouth_valid_vals else math.nan

		mouth_center = (math.nan, math.nan, math.nan)
		use_estimation = 0
		if not any(math.isnan(c) for c in mouth_l[:3]) and not any(math.isnan(c) for c in mouth_r[:3]):
			mouth_center = ((mouth_l[0]+mouth_r[0])/2.0, (mouth_l[1]+mouth_r[1])/2.0, (mouth_l[2]+mouth_r[2])/2.0)
			if nose_vis >= VIS_THRESH_NOSE and not any(math.isnan(c) for c in nose_pos + mouth_center):
				last_mouth_offset = (mouth_center[0]-nose_pos[0], mouth_center[1]-nose_pos[1], mouth_center[2]-nose_pos[2])
		else:
			# 嘴部點缺失, 根據 nose + offset 推估
			if (raw_mouth_conf < VIS_THRESH_MOUTH or math.isnan(raw_mouth_conf)) and not any(math.isnan(c) for c in nose_pos):
				if last_mouth_offset is None:
					# 建立預設 offset: 垂直向下 DEFAULT_MOUTH_OFFSET_Y
					last_mouth_offset = (0.0, DEFAULT_MOUTH_OFFSET_Y, 0.0)
				mouth_center = (nose_pos[0] + last_mouth_offset[0], nose_pos[1] + last_mouth_offset[1], nose_pos[2] + last_mouth_offset[2])
				use_estimation = 1
		# 置信度調整
		mouth_conf_adj = raw_mouth_conf
		if (raw_mouth_conf < VIS_THRESH_MOUTH or math.isnan(raw_mouth_conf)) and not math.isnan(nose_vis):
			mouth_conf_adj = max(raw_mouth_conf if not math.isnan(raw_mouth_conf) else 0.0, nose_vis * 0.6)

		# 手遮擋 (粗略): 手在嘴附近
		left_hand_pt = get_point(LEFT_INDEX)
		if any(math.isnan(c) for c in left_hand_pt[:3]):  # 用腕替代
			left_hand_pt = get_point(LEFT_WRIST)
		right_hand_pt = get_point(RIGHT_INDEX)
		if any(math.isnan(c) for c in right_hand_pt[:3]):
			right_hand_pt = get_point(RIGHT_WRIST)

		if not any(math.isnan(c) for c in mouth_center + left_hand_pt[:3]):
			d_l = _dist3(mouth_center, left_hand_pt[:3])
		else:
			d_l = math.nan
		if not any(math.isnan(c) for c in mouth_center + right_hand_pt[:3]):
			d_r = _dist3(mouth_center, right_hand_pt[:3])
		else:
			d_r = math.nan

		occlusion = False
		if (not math.isnan(d_l) and d_l < HAND_MOUTH_OCCLUDE_DIST) or (not math.isnan(d_r) and d_r < HAND_MOUTH_OCCLUDE_DIST):
			occlusion = True
		if occlusion and mouth_conf_adj < 0.3:
			# 將因遮擋導致的置信度適度抬升
			mouth_conf_adj = 0.3

		# 嘴部速度 / 加速度 (自行計算)
		mouth_v = (math.nan, math.nan, math.nan)
		mouth_a = (math.nan, math.nan, math.nan)
		if last_mouth_pos and not any(math.isnan(c) for c in last_mouth_pos + mouth_center):
			mouth_v = (mouth_center[0]-last_mouth_pos[0], mouth_center[1]-last_mouth_pos[1], mouth_center[2]-last_mouth_pos[2])
			if last_mouth_vel and not any(math.isnan(c) for c in last_mouth_vel + mouth_v):
				mouth_a = (mouth_v[0]-last_mouth_vel[0], mouth_v[1]-last_mouth_vel[1], mouth_v[2]-last_mouth_vel[2])
		last_mouth_vel = mouth_v
		last_mouth_pos = mouth_center

		# 距離特徵 (3D 與 2D) 及嘴寬
		nose_left_hand = _dist3(nose_pos, left_hand_pt[:3])
		nose_right_hand = _dist3(nose_pos, right_hand_pt[:3])
		hands_dist = _dist3(left_hand_pt[:3], right_hand_pt[:3])
		wrists_dist = _dist3(get_point(LEFT_WRIST)[:3], get_point(RIGHT_WRIST)[:3])
		mouth_left_hand = d_l
		mouth_right_hand = d_r
		# 2D 視圖距離 (XY)
		nose_left_hand_2d = _dist2(nose_pos, left_hand_pt[:3])
		nose_right_hand_2d = _dist2(nose_pos, right_hand_pt[:3])
		hands_dist_2d = _dist2(left_hand_pt[:3], right_hand_pt[:3])
		wrists_dist_2d = _dist2(get_point(LEFT_WRIST)[:3], get_point(RIGHT_WRIST)[:3])
		mouth_left_hand_2d = _dist2(mouth_center, left_hand_pt[:3]) if not any(math.isnan(c) for c in mouth_center) else math.nan
		mouth_right_hand_2d = _dist2(mouth_center, right_hand_pt[:3]) if not any(math.isnan(c) for c in mouth_center) else math.nan
		# 嘴巴寬度
		mouth_width_3d = math.nan
		mouth_width_2d = math.nan
		if not any(math.isnan(c) for c in mouth_l[:3] + mouth_r[:3]):
			mouth_width_3d = _dist3(mouth_l[:3], mouth_r[:3])
			mouth_width_2d = _dist2(mouth_l[:3], mouth_r[:3])

		# 肩寬 (正規化尺度)
		shoulder_l = get_point(11)
		shoulder_r = get_point(12)
		shoulder_width_3d = _dist3(shoulder_l[:3], shoulder_r[:3]) if not any(math.isnan(c) for c in shoulder_l[:3] + shoulder_r[:3]) else math.nan
		shoulder_width_2d = _dist2(shoulder_l[:3], shoulder_r[:3]) if not any(math.isnan(c) for c in shoulder_l[:3] + shoulder_r[:3]) else math.nan

		# Normalized mouth width
		normalized_mouth_width = math.nan
		if not math.isnan(mouth_width_2d) and not math.isnan(shoulder_width_2d) and shoulder_width_2d > 1e-6:
			normalized_mouth_width = mouth_width_2d / shoulder_width_2d

		# Torso-normalized 距離 (以肩寬規格化)
		def _norm(val, denom):
			if math.isnan(val) or math.isnan(denom) or denom <= 1e-6:
				return math.nan
			return val / denom
		feat.update({
			"mouth_x": mouth_center[0],
			"mouth_y": mouth_center[1],
			"mouth_z": mouth_center[2],
			"mouth_conf_raw": raw_mouth_conf,
			"mouth_conf_adj": mouth_conf_adj,
			"mouth_from_estimation": use_estimation,
			"mouth_width_3d": mouth_width_3d,
			"mouth_width_2d": mouth_width_2d,
			"normalized_mouth_width": normalized_mouth_width,
			"mouth_vx": mouth_v[0],
			"mouth_vy": mouth_v[1],
			"mouth_vz": mouth_v[2],
			"mouth_ax": mouth_a[0],
			"mouth_ay": mouth_a[1],
			"mouth_az": mouth_a[2],
			"dist_leftHand_mouth": mouth_left_hand,
			"dist_rightHand_mouth": mouth_right_hand,
			"dist_hands": hands_dist,
			"dist_nose_leftHand": nose_left_hand,
			"dist_nose_rightHand": nose_right_hand,
			"dist_wrists": wrists_dist,
			"dist2d_leftHand_mouth": mouth_left_hand_2d,
			"dist2d_rightHand_mouth": mouth_right_hand_2d,
			"dist2d_hands": hands_dist_2d,
			"dist2d_nose_leftHand": nose_left_hand_2d,
			"dist2d_nose_rightHand": nose_right_hand_2d,
			"dist2d_wrists": wrists_dist_2d,
			# Normalized 3D
			"norm_dist_leftHand_mouth": _norm(mouth_left_hand, shoulder_width_3d),
			"norm_dist_rightHand_mouth": _norm(mouth_right_hand, shoulder_width_3d),
			"norm_dist_hands": _norm(hands_dist, shoulder_width_3d),
			"norm_dist_nose_leftHand": _norm(nose_left_hand, shoulder_width_3d),
			"norm_dist_nose_rightHand": _norm(nose_right_hand, shoulder_width_3d),
			"norm_dist_wrists": _norm(wrists_dist, shoulder_width_3d),
			# Normalized 2D
			"norm2d_dist_leftHand_mouth": _norm(mouth_left_hand_2d, shoulder_width_2d),
			"norm2d_dist_rightHand_mouth": _norm(mouth_right_hand_2d, shoulder_width_2d),
			"norm2d_dist_hands": _norm(hands_dist_2d, shoulder_width_2d),
			"norm2d_dist_nose_leftHand": _norm(nose_left_hand_2d, shoulder_width_2d),
			"norm2d_dist_nose_rightHand": _norm(nose_right_hand_2d, shoulder_width_2d),
			"norm2d_dist_wrists": _norm(wrists_dist_2d, shoulder_width_2d),
			"shoulder_width_3d": shoulder_width_3d,
			"shoulder_width_2d": shoulder_width_2d,
			"occlusion_flag": 1 if occlusion else 0,
		})

		# 複製速度 + 計算加速度 (針對所有選取點) 先不判定 jump
		for idx in SELECTED_LANDMARKS:
			x, y, z, v, vx, vy, vz = fd.landmarks.get(idx, (math.nan,)*7)
			prev_v = last_vel.get(idx)
			if prev_v and not any(math.isnan(c) for c in (vx, vy, vz) + prev_v):
				ax = vx - prev_v[0]
				ay = vy - prev_v[1]
				az = vz - prev_v[2]
			else:
				ax = ay = az = math.nan
			feat[f"l{idx}_vx"] = vx
			feat[f"l{idx}_vy"] = vy
			feat[f"l{idx}_vz"] = vz
			feat[f"l{idx}_ax"] = ax
			feat[f"l{idx}_ay"] = ay
			feat[f"l{idx}_az"] = az
			if not any(math.isnan(c) for c in (vx, vy, vz)):
				last_vel[idx] = (vx, vy, vz)
		# 先占位, 第二階段再填
		feat["velocity_jump_flag"] = 0
		feat["velocity_jump_count"] = 0
		results.append(feat)
	# --- 第二階段: robust 門檻計算 ---
	def _collect_magnitudes(kind: str) -> List[float]:  # kind in {"v","a"}
		mags = []
		for feat in results:
			for idx in SELECTED_LANDMARKS:
				vx = feat.get(f"l{idx}_vx", math.nan)
				vy = feat.get(f"l{idx}_vy", math.nan)
				vz = feat.get(f"l{idx}_vz", math.nan)
				if kind == "v" and not any(math.isnan(c) for c in (vx, vy, vz)):
					mags.append(math.sqrt(vx*vx + vy*vy + vz*vz))
				if kind == "a":
					ax = feat.get(f"l{idx}_ax", math.nan)
					ay = feat.get(f"l{idx}_ay", math.nan)
					az = feat.get(f"l{idx}_az", math.nan)
					if not any(math.isnan(c) for c in (ax, ay, az)):
						mags.append(math.sqrt(ax*ax + ay*ay + az*az))
		return mags

	def _median_iqr(values: List[float]) -> Tuple[float, float, float, float]:
		vals = sorted([v for v in values if not math.isnan(v)])
		n = len(vals)
		if n == 0:
			return math.nan, math.nan, math.nan, math.nan
		def _percentile(p: float) -> float:
			if n == 1:
				return vals[0]
			k = (n - 1) * p
			f = math.floor(k)
			c = math.ceil(k)
			if f == c:
				return vals[int(k)]
			return vals[f] + (vals[c] - vals[f]) * (k - f)
		med = _percentile(0.5)
		q1 = _percentile(0.25)
		q3 = _percentile(0.75)
		return med, q1, q3, q3 - q1

	v_mags = _collect_magnitudes("v")
	a_mags = _collect_magnitudes("a")
	v_med, v_q1, v_q3, v_iqr = _median_iqr(v_mags)
	a_med, a_q1, a_q3, a_iqr = _median_iqr(a_mags)
	robust_v_thresh = (v_med + ROBUST_V_IQR_K * v_iqr) if not math.isnan(v_med) and not math.isnan(v_iqr) else VELOCITY_JUMP_V_THRESH
	robust_a_thresh = (a_med + ROBUST_A_IQR_K * a_iqr) if not math.isnan(a_med) and not math.isnan(a_iqr) else ACC_JUMP_A_THRESH
	# fallback: 至少不低於基礎門檻
	robust_v_thresh = max(robust_v_thresh, VELOCITY_JUMP_V_THRESH)
	robust_a_thresh = max(robust_a_thresh, ACC_JUMP_A_THRESH)

	for feat in results:
		count = 0
		for idx in SELECTED_LANDMARKS:
			vx = feat.get(f"l{idx}_vx", math.nan)
			vy = feat.get(f"l{idx}_vy", math.nan)
			vz = feat.get(f"l{idx}_vz", math.nan)
			ax = feat.get(f"l{idx}_ax", math.nan)
			ay = feat.get(f"l{idx}_ay", math.nan)
			az = feat.get(f"l{idx}_az", math.nan)
			if not any(math.isnan(c) for c in (vx, vy, vz)):
				vmag = math.sqrt(vx*vx + vy*vy + vz*vz)
			else:
				vmag = math.nan
			if not any(math.isnan(c) for c in (ax, ay, az)):
				amag = math.sqrt(ax*ax + ay*ay + az*az)
			else:
				amag = math.nan
			if (not math.isnan(vmag) and vmag > robust_v_thresh) or (not math.isnan(amag) and amag > robust_a_thresh):
				count += 1
		feat["velocity_jump_count"] = count
		feat["velocity_jump_flag"] = 1 if count >= JUMP_MIN_LANDMARKS else 0
		feat["velocity_jump_v_thresh_used"] = robust_v_thresh
		feat["velocity_jump_a_thresh_used"] = robust_a_thresh

	# 首幀導數補 0
	if results:
		first = results[0]
		# 嘴部導數
		for k in ["mouth_vx","mouth_vy","mouth_vz","mouth_ax","mouth_ay","mouth_az"]:
			if math.isnan(first.get(k, math.nan)):
				first[k] = 0.0
		# landmark 加速度 (首幀 ax/ay/az -> 0)
		for idx in SELECTED_LANDMARKS:
			for comp in ["ax","ay","az"]:
				key = f"l{idx}_{comp}"
				if math.isnan(first.get(key, math.nan)):
					first[key] = 0.0
		# landmark 速度 (首幀 vx/vy/vz -> 0)
		for idx in SELECTED_LANDMARKS:
			for comp in ["vx","vy","vz"]:
				key = f"l{idx}_{comp}"
				if math.isnan(first.get(key, math.nan)):
					first[key] = 0.0
	return results


def write_feature_csv(path: str, features: List[Dict[str, float]]):
	if not features:
		return
	os.makedirs(os.path.dirname(path), exist_ok=True)
	# 保持欄位順序: frame,time_sec,... 其他排序
	base_keys = [
		"frame", "time_sec",
		"yaw_est", "yaw_raw_ratio", "yaw_valid_flag", "face_orient_class",
		"mouth_x", "mouth_y", "mouth_z", "mouth_width_3d", "mouth_width_2d", "normalized_mouth_width",
		"mouth_conf_raw", "mouth_conf_adj", "mouth_from_estimation",
		"mouth_vx", "mouth_vy", "mouth_vz", "mouth_ax", "mouth_ay", "mouth_az",
		"dist_leftHand_mouth", "dist_rightHand_mouth", "dist_hands", "dist_nose_leftHand", "dist_nose_rightHand", "dist_wrists",
		"dist2d_leftHand_mouth", "dist2d_rightHand_mouth", "dist2d_hands", "dist2d_nose_leftHand", "dist2d_nose_rightHand", "dist2d_wrists",
		"norm_dist_leftHand_mouth", "norm_dist_rightHand_mouth", "norm_dist_hands", "norm_dist_nose_leftHand", "norm_dist_nose_rightHand", "norm_dist_wrists",
		"norm2d_dist_leftHand_mouth", "norm2d_dist_rightHand_mouth", "norm2d_dist_hands", "norm2d_dist_nose_leftHand", "norm2d_dist_nose_rightHand", "norm2d_dist_wrists",
		"shoulder_width_3d", "shoulder_width_2d",
		"occlusion_flag", "velocity_jump_flag", "velocity_jump_count", "velocity_jump_v_thresh_used", "velocity_jump_a_thresh_used"
	]
	# 追加 landmark 動態欄位
	dyn_cols = []
	for idx in SELECTED_LANDMARKS:
		dyn_cols.extend([
			f"l{idx}_vx", f"l{idx}_vy", f"l{idx}_vz", f"l{idx}_ax", f"l{idx}_ay", f"l{idx}_az"
		])
	header = base_keys + dyn_cols
	# 補上 features 中其它可能欄位 (保險)
	for k in features[0].keys():
		if k not in header:
			header.append(k)
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(header)
		for feat in features:
			row = []
			for k in header:
				val = feat.get(k, math.nan)
				if isinstance(val, float):
					if math.isnan(val):
						row.append("")
					else:
						row.append(f"{val:.6f}")
				else:
					row.append(val)
			writer.writerow(row)


def process_all():
	if not os.path.isdir(POSE_INPUT_DIR):
		print(f"⚠️ Pose 輸入資料夾不存在: {POSE_INPUT_DIR}")
		return
	os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
	pose_files = [f for f in os.listdir(POSE_INPUT_DIR) if f.endswith("_pose.csv")]
	if not pose_files:
		print("⚠️ 沒有找到 *_pose.csv")
		return
	for fname in pose_files:
		in_path = os.path.join(POSE_INPUT_DIR, fname)
		out_name = fname.replace("_pose.csv", "_eig.csv")
		out_path = os.path.join(FEATURE_OUTPUT_DIR, out_name)
		try:
			frames = read_pose_csv(in_path)
			feats = compute_features(frames)
			write_feature_csv(out_path, feats)
			print(f"✅ 特徵完成: {fname} -> {out_path}")
		except Exception as e:
			print(f"❌ 處理失敗 {fname}: {e}")

	# 生成 schema 說明
	schema_path = os.path.join(FEATURE_OUTPUT_DIR, "feature_schema.json")
	try:
		import json
		schema: Dict[str, Dict[str, str]] = {}
		desc = {
			"frame": "幀索引",
			"time_sec": "時間 (秒)",
			"yaw_est": "Clamp 後臉部 yaw (-1~1)",
			"yaw_raw_ratio": "未 clamp 前 nose 在眼睛線段比例 (可超出 0~1)",
			"yaw_valid_flag": "yaw 是否有效 (1 有效 / 0 無)",
			"face_orient_class": "臉方向分類 -1 左 0 正 1 右",
			"mouth_x": "嘴中心 X",
			"mouth_y": "嘴中心 Y",
			"mouth_z": "嘴中心 Z",
			"mouth_width_3d": "嘴左右點 3D 距離",
			"mouth_width_2d": "嘴左右點 2D (XY) 距離",
			"normalized_mouth_width": "嘴寬 2D / 肩寬 2D (首幀若缺失為 NaN)",
			"mouth_conf_raw": "嘴點原始平均可見度",
			"mouth_conf_adj": "嘴點調整後可見度",
			"mouth_from_estimation": "是否估算嘴 (1 是)",
			"mouth_vx": "嘴中心速度 x (首幀=0)",
			"mouth_vy": "嘴中心速度 y (首幀=0)",
			"mouth_vz": "嘴中心速度 z (首幀=0)",
			"mouth_ax": "嘴中心加速度 x (首幀=0)",
			"mouth_ay": "嘴中心加速度 y (首幀=0)",
			"mouth_az": "嘴中心加速度 z (首幀=0)",
			"dist_leftHand_mouth": "左手-嘴 3D 距離",
			"dist_rightHand_mouth": "右手-嘴 3D 距離",
			"dist_hands": "雙手 3D 距離",
			"dist_nose_leftHand": "鼻-左手 3D 距離",
			"dist_nose_rightHand": "鼻-右手 3D 距離",
			"dist_wrists": "雙腕 3D 距離",
			"dist2d_leftHand_mouth": "左手-嘴 2D 距離",
			"dist2d_rightHand_mouth": "右手-嘴 2D 距離",
			"dist2d_hands": "雙手 2D 距離",
			"dist2d_nose_leftHand": "鼻-左手 2D 距離",
			"dist2d_nose_rightHand": "鼻-右手 2D 距離",
			"dist2d_wrists": "雙腕 2D 距離",
			"norm_dist_leftHand_mouth": "左手-嘴 3D 距離 / 肩寬 3D",
			"norm_dist_rightHand_mouth": "右手-嘴 3D 距離 / 肩寬 3D",
			"norm_dist_hands": "雙手 3D 距離 / 肩寬 3D",
			"norm_dist_nose_leftHand": "鼻-左手 3D 距離 / 肩寬 3D",
			"norm_dist_nose_rightHand": "鼻-右手 3D 距離 / 肩寬 3D",
			"norm_dist_wrists": "雙腕 3D 距離 / 肩寬 3D",
			"norm2d_dist_leftHand_mouth": "左手-嘴 2D 距離 / 肩寬 2D",
			"norm2d_dist_rightHand_mouth": "右手-嘴 2D 距離 / 肩寬 2D",
			"norm2d_dist_hands": "雙手 2D 距離 / 肩寬 2D",
			"norm2d_dist_nose_leftHand": "鼻-左手 2D 距離 / 肩寬 2D",
			"norm2d_dist_nose_rightHand": "鼻-右手 2D 距離 / 肩寬 2D",
			"norm2d_dist_wrists": "雙腕 2D 距離 / 肩寬 2D",
			"shoulder_width_3d": "左右肩 3D 距離",
			"shoulder_width_2d": "左右肩 2D 距離",
			"occlusion_flag": "嘴被任一手接近遮擋",
			"velocity_jump_flag": "多 landmark 異常跳變綜合旗標 (>=JUMP_MIN_LANDMARKS)",
			"velocity_jump_count": "本幀超過 robust 門檻 (速度或加速度) 的 landmark 數",
			"velocity_jump_v_thresh_used": "使用的速度 magnitude robust 門檻 (max(基礎,median+k*IQR))",
			"velocity_jump_a_thresh_used": "使用的加速度 magnitude robust 門檻 (max(基礎,median+k*IQR))",
		}
		for k, v in desc.items():
			schema[k] = {"description": v}
		# Landmark 動態欄位
		for idx in SELECTED_LANDMARKS:
			for comp in ["vx", "vy", "vz", "ax", "ay", "az"]:
				name = f"l{idx}_{comp}"
				base = f"landmark {idx} {comp}"
				if comp.startswith('a') or comp.startswith('v'):
					schema[name] = {"description": base + " (首幀=0)"}
				else:
					schema[name] = {"description": base}
		with open(schema_path, "w", encoding="utf-8") as sf:
			json.dump(schema, sf, ensure_ascii=False, indent=2)
		print(f"🛈 已輸出 schema: {schema_path}")
	except Exception as e:
		print(f"⚠️ 無法寫入 schema: {e}")


if __name__ == "__main__":
	print(f"[INFO] 來源目錄: {POSE_INPUT_DIR}")
	print(f"[INFO] 輸出目錄: {FEATURE_OUTPUT_DIR}")
	process_all()

