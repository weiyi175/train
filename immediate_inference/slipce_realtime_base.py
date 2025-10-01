from __future__ import annotations

"""
Realtime Pose->Feature extractor used by stream_infer.

設計目標:
- 從單張 BGR 影格以 MediaPipe Pose 取 0-6,9-22 關鍵點
- 線上計算必要的 36 維特徵（順序對齊 stream_infer.FEATURE_COLS）
- 維護跨幀狀態: landmark 速度/加速度、嘴部位置/速度、robust 門檻的滾動統計

注意:
- 為避免非即時模式也要安裝 mediapipe，本模組在 __init__ 中延遲 import mediapipe
"""

import math
from typing import Dict, Tuple, List, Optional
from collections import deque

import numpy as np

# Landmark 索引與集合 (與 Tool/extract_pose.py 保持一致)
SELECTED_LANDMARKS = list(range(0, 7)) + list(range(9, 23))
NOSE = 0
LEFT_EYE_GROUP = [1, 2, 3]
RIGHT_EYE_GROUP = [4, 5, 6]
MOUTH_L = 9
MOUTH_R = 10
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_INDEX = 19
RIGHT_INDEX = 20

# 門檻/常數 (與 Tool/Eigenvalue.py 對齊)
VIS_THRESH_NOSE = 0.5
VIS_THRESH_MOUTH = 0.5
YAW_LEFT = -0.2
YAW_RIGHT = 0.2
HAND_MOUTH_OCCLUDE_DIST = 0.08
DEFAULT_MOUTH_OFFSET_Y = 0.07
VELOCITY_JUMP_V_THRESH = 3.0
ACC_JUMP_A_THRESH = 6.0
ROBUST_V_IQR_K = 3.0
ROBUST_A_IQR_K = 3.0
JUMP_MIN_LANDMARKS = 3

# 與 stream_infer 同步的特徵欄位順序
FEATURE_COLS = [
	"dist_leftHand_mouth", "dist_rightHand_mouth",
	"norm_dist_leftHand_mouth", "norm_dist_rightHand_mouth",
	"dist_nose_leftHand", "dist_nose_rightHand",
	"mouth_conf_adj", "occlusion_flag", "mouth_vx", "mouth_vy", "mouth_vz",
	# l15, l16, l19, l20 的 (vx,vy,vz,ax,ay,az)
	"l15_vx", "l15_vy", "l15_vz", "l15_ax", "l15_ay", "l15_az",
	"l16_vx", "l16_vy", "l16_vz", "l16_ax", "l16_ay", "l16_az",
	"l19_vx", "l19_vy", "l19_vz", "l19_ax", "l19_ay", "l19_az",
	"l20_vx", "l20_vy", "l20_vz", "l20_ax", "l20_ay", "l20_az",
	"velocity_jump_flag",
]


def _dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
	if any(math.isnan(c) for c in a + b):
		return math.nan
	return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _dist2(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
	if any(math.isnan(c) for c in (a[0], a[1], b[0], b[1])):
		return math.nan
	return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def _avg_points(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
	xs = [p[0] for p in points if not any(math.isnan(c) for c in p)]
	ys = [p[1] for p in points if not any(math.isnan(c) for c in p)]
	zs = [p[2] for p in points if not any(math.isnan(c) for c in p)]
	if not xs or not ys or not zs:
		return (math.nan, math.nan, math.nan)
	return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


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


class OnlinePoseFeatureExtractor:
	"""每幀輸入 BGR 影像 + dt，輸出 36 維特徵向量 (np.float32)。"""

	def __init__(self,
				 model_complexity: int = 1,
				 min_detection_confidence: float = 0.5,
				 min_tracking_confidence: float = 0.5,
				 visibility_threshold: float = 0.0,
				 robust_window: int = 150):
		try:
			import mediapipe as mp  # 延遲 import 避免非即時模式破壞環境
		except ImportError as e:
			raise RuntimeError("缺少 mediapipe，請安裝: pip install mediapipe") from e
		self._mp_pose = mp.solutions.pose
		self._pose = self._mp_pose.Pose(
			static_image_mode=False,
			model_complexity=model_complexity,
			enable_segmentation=False,
			smooth_landmarks=True,
			min_detection_confidence=min_detection_confidence,
			min_tracking_confidence=min_tracking_confidence,
		)
		self.vis_thresh = visibility_threshold
		# 狀態
		self.prev_coords: Optional[Dict[int, Tuple[float, float, float]]] = None
		self.last_vel: Dict[int, Tuple[float, float, float]] = {}
		self.last_mouth_pos: Optional[Tuple[float, float, float]] = None
		self.last_mouth_vel: Optional[Tuple[float, float, float]] = None
		self.last_mouth_offset: Optional[Tuple[float, float, float]] = None
		# robust 門檻的滾動緩衝 (儲存 magnitude)
		self.v_mags = deque(maxlen=max(1, robust_window * len(SELECTED_LANDMARKS)))
		self.a_mags = deque(maxlen=max(1, robust_window * len(SELECTED_LANDMARKS)))

	@property
	def feature_dim(self) -> int:
		return len(FEATURE_COLS)

	def close(self):
		try:
			self._pose.close()
		except Exception:
			pass

	def _extract_landmarks(self, frame_bgr) -> Dict[int, Tuple[float, float, float, float]]:
		import cv2
		image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		result = self._pose.process(image_rgb)
		coords_now: Dict[int, Tuple[float, float, float, float]] = {}
		if result.pose_landmarks:
			lm_list = result.pose_landmarks.landmark
			for idx in SELECTED_LANDMARKS:
				lm = lm_list[idx]
				x, y, z, v = lm.x, lm.y, lm.z, lm.visibility
				if self.vis_thresh and v < self.vis_thresh:
					coords_now[idx] = (math.nan, math.nan, math.nan, v)
				else:
					coords_now[idx] = (x, y, z, v)
		for idx in SELECTED_LANDMARKS:
			if idx not in coords_now:
				coords_now[idx] = (math.nan, math.nan, math.nan, math.nan)
		return coords_now

	def _compute_velocity_acc(self, coords_now: Dict[int, Tuple[float, float, float, float]], dt: float) -> Dict[str, Dict[int, Tuple[float, float, float]]]:
		# 回傳: { 'v': {idx:(vx,vy,vz)}, 'a': {idx:(ax,ay,az)} }
		V: Dict[int, Tuple[float, float, float]] = {}
		A: Dict[int, Tuple[float, float, float]] = {}
		eff_dt = max(dt, 1e-6)
		for idx in SELECTED_LANDMARKS:
			x, y, z, _ = coords_now[idx]
			if self.prev_coords and idx in self.prev_coords:
				px, py, pz = self.prev_coords[idx]
				if not any(math.isnan(v) for v in (x, y, z, px, py, pz)):
					vx = (x - px) / eff_dt
					vy = (y - py) / eff_dt
					vz = (z - pz) / eff_dt
				else:
					vx = vy = vz = math.nan
			else:
				vx = vy = vz = 0.0  # 首幀速度 0
			V[idx] = (vx, vy, vz)
			# 加速度
			if idx in self.last_vel and not any(math.isnan(v) for v in (vx, vy, vz) + self.last_vel[idx]):
				ax = vx - self.last_vel[idx][0]
				ay = vy - self.last_vel[idx][1]
				az = vz - self.last_vel[idx][2]
			else:
				ax = ay = az = 0.0 if self.prev_coords is None else math.nan
			A[idx] = (ax, ay, az)
		# 更新狀態
		self.last_vel = V.copy()
		self.prev_coords = {i: (c[0], c[1], c[2]) for i, c in coords_now.items()}
		return {"v": V, "a": A}

	def _online_jump_flag(self, V: Dict[int, Tuple[float, float, float]], A: Dict[int, Tuple[float, float, float]]) -> Tuple[int, float, float]:
		# 更新 magnitude 緩衝
		cur_v_mags = []
		cur_a_mags = []
		for idx in SELECTED_LANDMARKS:
			vx, vy, vz = V[idx]
			ax, ay, az = A[idx]
			if not any(math.isnan(c) for c in (vx, vy, vz)):
				cur_v_mags.append(math.sqrt(vx*vx + vy*vy + vz*vz))
			if not any(math.isnan(c) for c in (ax, ay, az)):
				cur_a_mags.append(math.sqrt(ax*ax + ay*ay + az*az))
		self.v_mags.extend(cur_v_mags)
		self.a_mags.extend(cur_a_mags)
		# robust 門檻
		v_med, _, _, v_iqr = _median_iqr(list(self.v_mags))
		a_med, _, _, a_iqr = _median_iqr(list(self.a_mags))
		robust_v = max(VELOCITY_JUMP_V_THRESH, (v_med + ROBUST_V_IQR_K * v_iqr) if (not math.isnan(v_med) and not math.isnan(v_iqr)) else VELOCITY_JUMP_V_THRESH)
		robust_a = max(ACC_JUMP_A_THRESH, (a_med + ROBUST_A_IQR_K * a_iqr) if (not math.isnan(a_med) and not math.isnan(a_iqr)) else ACC_JUMP_A_THRESH)
		# 計數
		count = 0
		for idx in SELECTED_LANDMARKS:
			vx, vy, vz = V[idx]
			ax, ay, az = A[idx]
			vmag = math.sqrt(vx*vx + vy*vy + vz*vz) if not any(math.isnan(c) for c in (vx, vy, vz)) else math.nan
			amag = math.sqrt(ax*ax + ay*ay + az*az) if not any(math.isnan(c) for c in (ax, ay, az)) else math.nan
			if (not math.isnan(vmag) and vmag > robust_v) or (not math.isnan(amag) and amag > robust_a):
				count += 1
		flag = 1 if count >= JUMP_MIN_LANDMARKS else 0
		return flag, robust_v, robust_a

	def process(self, frame_bgr, dt: float) -> Tuple[np.ndarray, Dict[str, float]]:
		"""回傳 (features_36[np.float32], extras)
		extras: 可包含 debug 用的中介變數
		"""
		coords_now = self._extract_landmarks(frame_bgr)
		dyn = self._compute_velocity_acc(coords_now, dt)
		V, A = dyn["v"], dyn["a"]

		# 眼睛中心與鼻
		def get_point(idx: int) -> Tuple[float, float, float, float]:
			return coords_now.get(idx, (math.nan, math.nan, math.nan, math.nan))

		l_eye = _avg_points([get_point(i)[:3] for i in LEFT_EYE_GROUP])
		r_eye = _avg_points([get_point(i)[:3] for i in RIGHT_EYE_GROUP])
		nose_xyzv = get_point(NOSE)
		nose_pos, nose_vis = nose_xyzv[:3], nose_xyzv[3]

		# 嘴中心與置信度 (含估算)
		mouth_l = get_point(MOUTH_L)
		mouth_r = get_point(MOUTH_R)
		raw_mouth_conf = np.nanmean([mouth_l[3], mouth_r[3]]) if not all(math.isnan(v) for v in [mouth_l[3], mouth_r[3]]) else math.nan
		mouth_center = (math.nan, math.nan, math.nan)
		used_est = 0
		if not any(math.isnan(c) for c in mouth_l[:3] + mouth_r[:3]):
			mouth_center = ((mouth_l[0]+mouth_r[0])/2.0, (mouth_l[1]+mouth_r[1])/2.0, (mouth_l[2]+mouth_r[2])/2.0)
			if not any(math.isnan(c) for c in nose_pos) and (nose_vis >= VIS_THRESH_NOSE):
				self.last_mouth_offset = (mouth_center[0]-nose_pos[0], mouth_center[1]-nose_pos[1], mouth_center[2]-nose_pos[2])
		else:
			if (math.isnan(raw_mouth_conf) or raw_mouth_conf < VIS_THRESH_MOUTH) and not any(math.isnan(c) for c in nose_pos):
				if self.last_mouth_offset is None:
					self.last_mouth_offset = (0.0, DEFAULT_MOUTH_OFFSET_Y, 0.0)
				mouth_center = (nose_pos[0]+self.last_mouth_offset[0], nose_pos[1]+self.last_mouth_offset[1], nose_pos[2]+self.last_mouth_offset[2])
				used_est = 1
		mouth_conf_adj = raw_mouth_conf
		if (math.isnan(raw_mouth_conf) or raw_mouth_conf < VIS_THRESH_MOUTH) and not math.isnan(nose_vis):
			mouth_conf_adj = max(0.0 if math.isnan(raw_mouth_conf) else raw_mouth_conf, nose_vis * 0.6)

		# 手遮擋 (嘴到手距離)
		left_hand = get_point(LEFT_INDEX)
		if any(math.isnan(c) for c in left_hand[:3]):
			left_hand = get_point(LEFT_WRIST)
		right_hand = get_point(RIGHT_INDEX)
		if any(math.isnan(c) for c in right_hand[:3]):
			right_hand = get_point(RIGHT_WRIST)
		d_l = _dist3(mouth_center, left_hand[:3]) if not any(math.isnan(c) for c in mouth_center + left_hand[:3]) else math.nan
		d_r = _dist3(mouth_center, right_hand[:3]) if not any(math.isnan(c) for c in mouth_center + right_hand[:3]) else math.nan
		occlusion = (not math.isnan(d_l) and d_l < HAND_MOUTH_OCCLUDE_DIST) or (not math.isnan(d_r) and d_r < HAND_MOUTH_OCCLUDE_DIST)
		if occlusion and (math.isnan(mouth_conf_adj) or mouth_conf_adj < 0.3):
			mouth_conf_adj = 0.3

		# 嘴速度/加速度 (透過位置差分)
		mouth_v = (math.nan, math.nan, math.nan)
		mouth_a = (math.nan, math.nan, math.nan)
		if self.last_mouth_pos and not any(math.isnan(c) for c in self.last_mouth_pos + mouth_center):
			eff_dt = max(dt, 1e-6)
			mouth_v = ((mouth_center[0]-self.last_mouth_pos[0])/eff_dt,
					   (mouth_center[1]-self.last_mouth_pos[1])/eff_dt,
					   (mouth_center[2]-self.last_mouth_pos[2])/eff_dt)
			if self.last_mouth_vel and not any(math.isnan(c) for c in self.last_mouth_vel + mouth_v):
				mouth_a = (mouth_v[0]-self.last_mouth_vel[0], mouth_v[1]-self.last_mouth_vel[1], mouth_v[2]-self.last_mouth_vel[2])
		self.last_mouth_vel = mouth_v
		self.last_mouth_pos = mouth_center

		# 其他距離
		nose_left_hand = _dist3(nose_pos, left_hand[:3])
		nose_right_hand = _dist3(nose_pos, right_hand[:3])
		# 正規化尺度: 肩寬
		shoulder_l = get_point(11)
		shoulder_r = get_point(12)
		shoulder_width_3d = _dist3(shoulder_l[:3], shoulder_r[:3]) if not any(math.isnan(c) for c in shoulder_l[:3] + shoulder_r[:3]) else math.nan

		def _norm(v, denom):
			if math.isnan(v) or math.isnan(denom) or denom <= 1e-6:
				return math.nan
			return v / denom

		norm_left = _norm(d_l, shoulder_width_3d)
		norm_right = _norm(d_r, shoulder_width_3d)

		# 線上 jump flag
		jump_flag, robust_v, robust_a = self._online_jump_flag(V, A)

		# 建 36F 特徵 (順序與 FEATURE_COLS 完全一致)
		def _nz(x: float) -> float:
			return 0.0 if (x is None or math.isnan(x)) else float(x)

		feat_vec: List[float] = [
			_nz(d_l), _nz(d_r),
			_nz(norm_left), _nz(norm_right),
			_nz(nose_left_hand), _nz(nose_right_hand),
			_nz(mouth_conf_adj), float(1 if occlusion else 0),
			_nz(mouth_v[0]), _nz(mouth_v[1]), _nz(mouth_v[2]),
		]
		for lid in (15, 16, 19, 20):
			vx, vy, vz = V.get(lid, (math.nan, math.nan, math.nan))
			ax, ay, az = A.get(lid, (math.nan, math.nan, math.nan))
			feat_vec.extend([_nz(vx), _nz(vy), _nz(vz), _nz(ax), _nz(ay), _nz(az)])
		feat_vec.append(float(jump_flag))

		return np.asarray(feat_vec, dtype=np.float32), {
			"robust_v_thresh": robust_v,
			"robust_a_thresh": robust_a,
			"occlusion": int(occlusion),
		}


__all__ = [
	"OnlinePoseFeatureExtractor",
	"FEATURE_COLS",
]


