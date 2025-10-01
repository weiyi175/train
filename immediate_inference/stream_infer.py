#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import cv2
from collections import deque

from rt_model_loader import load_short_model, load_long_model, load_fusion_mlp, infer_short_window, infer_long_window, fuse_mlp

"""
真即時骨架→特徵→短/長窗→融合：
- 兩種特徵來源模式：
    1) eigen_csv（預設）：直接讀 36F Eigenvalue CSV
    2) mediapipe：以 MediaPipe 即時抽 landmark 並即時計算 36F 特徵
- 正規化策略：
    - 短窗：window z-score
    - 長窗：
         • eigen_csv：使用整部影片統計的 mean/std
         • mediapipe：採用線上 Welford 統計近似影片層 mean/std
"""

FEATURE_COLS = [
    "dist_leftHand_mouth", "dist_rightHand_mouth",
    "norm_dist_leftHand_mouth", "norm_dist_rightHand_mouth",
    "dist_nose_leftHand", "dist_nose_rightHand",
    "mouth_conf_adj", "occlusion_flag", "mouth_vx", "mouth_vy", "mouth_vz",
    "l15_vx", "l15_vy", "l15_vz", "l15_ax", "l15_ay", "l15_az",
    "l16_vx", "l16_vy", "l16_vz", "l16_ax", "l16_ay", "l16_az",
    "l19_vx", "l19_vy", "l19_vz", "l19_ax", "l19_ay", "l19_az",
    "l20_vx", "l20_vy", "l20_vz", "l20_ax", "l20_ay", "l20_az",
    "velocity_jump_flag"
]


def read_eigen_csv(csv_path: str):
    import csv, math
    rows=[]; frames=[]
    with open(csv_path,'r',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        for r in reader:
            try:
                frame=int(float(r.get('frame',0)))
            except:
                frame=len(rows)
            frames.append(frame)
            feat=[r.get(c) for c in FEATURE_COLS]
            vec=[]
            for v in feat:
                try: vec.append(float(v))
                except: vec.append(math.nan)
            rows.append(vec)
    X=np.array(rows,dtype=np.float32)  # (N,F)
    return frames, X


def _mean_std_no_nan(X: np.ndarray):
    """Compute per-dim mean/std ignoring NaNs without raising warnings.
    If a dim has no valid values, mean=0, std=1.
    Uses population std (ddof=0) to match np.nanstd default.
    """
    mask = ~np.isnan(X)
    count = mask.sum(axis=0)
    X0 = np.where(mask, X, 0.0)
    sum_x = X0.sum(axis=0, dtype=np.float64)
    mean = np.divide(sum_x, count, out=np.zeros_like(sum_x, dtype=np.float64), where=count>0)
    sum_x2 = np.square(X0, dtype=np.float64).sum(axis=0)
    ex2 = np.divide(sum_x2, count, out=np.zeros_like(sum_x2, dtype=np.float64), where=count>0)
    var = np.maximum(ex2 - np.square(mean), 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_window(X):
    # X: (T,F)
    m, s = _mean_std_no_nan(X)
    Z=(np.nan_to_num(X, nan=0.0)-m)/s
    return Z.astype(np.float32)


def zscore_video(X):
    m, s = _mean_std_no_nan(X)
    def norm_win(W):
        return ((np.nan_to_num(W,nan=0.0)-m)/s).astype(np.float32)
    return norm_win


class RunningVideoNorm:
    """線上估計影片層 mean/std，供長窗正規化使用（mediapipe 模式）。"""
    def __init__(self, F: int):
        self.F = F
        # per-dimension counts for valid (non-NaN) updates
        self.n = np.zeros((F,), dtype=np.int64)
        self.mean = np.zeros((F,), dtype=np.float64)
        self.M2 = np.zeros((F,), dtype=np.float64)   # sum of squares of diffs

    def update(self, x: np.ndarray):
        # x: (F,)
        xv = x.astype(np.float64)
        mask = ~np.isnan(xv)
        if not np.any(mask):
            return
        # Welford per-dimension where valid
        n_old = self.n[mask].astype(np.float64)
        n_new = n_old + 1.0
        delta = xv[mask] - self.mean[mask]
        self.mean[mask] += delta / n_new
        delta2 = xv[mask] - self.mean[mask]
        self.M2[mask] += delta * delta2
        self.n[mask] += 1

    def norm_win(self, W: np.ndarray) -> np.ndarray:
        # W: (T,F)
        valid = self.n > 1
        mean = self.mean.copy()
        var = np.zeros_like(self.M2)
        var[valid] = self.M2[valid] / (self.n[valid] - 1)
        std = np.sqrt(np.maximum(var, 1e-8))
        X = np.nan_to_num(W.astype(np.float32), nan=0.0)
        return ((X - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video-id', required=True, help='例如 12（會讀取 test_data/video/12.mp4 與 test_data/Eigenvalue/12_eig.csv）')
    ap.add_argument('--root', default='/home/user/projects/train')
    ap.add_argument('--short-run-dir', required=True, help='短窗 PyTorch run 目錄（含 config.json, best.ckpt）')
    ap.add_argument('--long-weights', required=False, help='長窗 Keras 權重 best.weights 路徑（可省略=>僅短窗）')
    ap.add_argument('--fusion-dir', required=True, help='融合 MLP 目錄（含 fusion.ckpt, selected_threshold.json）')
    ap.add_argument('--short-win', type=int, default=30)
    ap.add_argument('--short-stride', type=int, default=15)
    ap.add_argument('--long-win', type=int, default=75)
    ap.add_argument('--long-stride', type=int, default=20)
    ap.add_argument('--out', default='immediate_inference/out_realtime')
    ap.add_argument('--feature-mode', choices=['eigen_csv','mediapipe'], default='eigen_csv', help='per-frame 特徵來源')
    ap.add_argument('--max-frames', type=int, default=-1, help='最多處理幀數（>0 有效）')
    # 平滑與去抖
    ap.add_argument('--ema-alpha', type=float, default=0.2, help='fused 機率的 EMA 平滑係數')
    ap.add_argument('--hyst-up', type=float, default=0.70, help='hysteresis 上升門檻（由 0->1）')
    ap.add_argument('--hyst-down', type=float, default=0.55, help='hysteresis 下降門檻（由 1->0）')
    # 長窗確認 gating
    ap.add_argument('--require-long-confirm', action='store_true', help='短窗欲由 0->1 時，需要長窗於確認視窗內出現一次 >= long-confirm-threshold 才真正轉為 1')
    ap.add_argument('--long-confirm-window', type=int, default=60, help='短窗觸發後等待長窗確認的最大幀數範圍')
    ap.add_argument('--long-confirm-threshold', type=float, default=0.5, help='長窗確認所需的最低長窗或融合機率 (使用融合若同幀存在)')
    # Mediapipe 參數
    ap.add_argument('--mp-model-complexity', type=int, default=1)
    ap.add_argument('--mp-min-detection-confidence', type=float, default=0.5)
    ap.add_argument('--mp-min-tracking-confidence', type=float, default=0.5)
    ap.add_argument('--mp-visibility-threshold', type=float, default=0.0)
    args=ap.parse_args()

    root=Path(args.root)
    vid=str(args.video_id)
    video_path=root/'test_data'/'video'/f'{vid}.mp4'
    assert video_path.exists(), f'missing {video_path}'

    # 讀整部影片 per-frame 特徵來源
    use_mediapipe = (args.feature_mode == 'mediapipe')
    if not use_mediapipe:
        eig_csv=root/'test_data'/'Eigenvalue'/f'{vid}_eig.csv'
        assert eig_csv.exists(), f'missing {eig_csv}'
        frames, X_all = read_eigen_csv(str(eig_csv))  # (N,F)
        norm_long = zscore_video(X_all)
    else:
        from slipce_realtime_base import OnlinePoseFeatureExtractor, FEATURE_COLS as MP_FEATURE_COLS
        X_all = None
        # 長窗線上正規化器
        running_norm = RunningVideoNorm(F=len(MP_FEATURE_COLS))

    # 載入模型
    short_model, torch_device, F = load_short_model(args.short_run_dir)
    # 檢查長窗權重路徑：支援 H5/KERAS 檔或 TF checkpoint 前綴（需有同名 .index 檔）
    def _is_tf_weights(path_str: str) -> bool:
        if not path_str:
            return False
        p = Path(path_str)
        if p.suffix.lower() in {'.h5', '.keras'}:
            return p.exists()
        # 視為 checkpoint 前綴
        idx = Path(str(p) + '.index')
        return idx.exists()

    long_model = None
    if args.long_weights and _is_tf_weights(args.long_weights):
        long_model = load_long_model(args.long_weights, input_shape=(F, args.long_win))
    fusion_model, fusion_device = load_fusion_mlp(args.fusion_dir)
    meta = json.load(open(Path(args.fusion_dir)/'selected_threshold.json'))
    thr = float(meta['selected_threshold'])
    # 若未指定，與 selected_threshold 保持一致（此處僅覆寫顏色參考，不影響 hysteresis）
    up_thr = float(args.hyst_up)
    down_thr = float(args.hyst_down)
    if down_thr > up_thr:
        down_thr = max(0.0, up_thr - 0.1)

    # 滑動視窗緩衝
    short_buf = deque(maxlen=args.short_win)
    long_buf = deque(maxlen=args.long_win)

    out_dir=Path(args.out)/vid; out_dir.mkdir(parents=True, exist_ok=True)
    csv_path=out_dir/'realtime.csv'
    overlay_path=out_dir/'overlay.mp4'

    cap=cv2.VideoCapture(str(video_path))
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    bar_h=40
    writer=cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h+bar_h))

    # Mediapipe extractor
    if use_mediapipe:
        try:
            extractor = OnlinePoseFeatureExtractor(
                model_complexity=args.mp_model_complexity,
                min_detection_confidence=args.mp_min_detection_confidence,
                min_tracking_confidence=args.mp_min_tracking_confidence,
                visibility_threshold=args.mp_visibility_threshold,
            )
        except Exception as e:
            raise SystemExit(f"Mediapipe 模式啟動失敗，請確認環境: {e}")
        assert extractor.feature_dim == F, f"特徵維度與模型不符: {extractor.feature_dim} != {F}"

    with open(csv_path,'w',encoding='utf-8') as f:
        f.write('frame,short_prob,long_prob,fused_prob,pred\n')
        frame_idx=0
        # Keep last known probabilities to forward-fill for visualization
        last_short=0.0; last_long=0.0; last_fused=0.0; last_pred=0
        ema_fused=0.0
        # 短窗候選觸發等待長窗確認的狀態 (-1 表示無候選)
        pending_confirm_frame = -1
        pending_confirm_prob = 0.0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
            if not use_mediapipe:
                # 取對應幀的特徵，越界則填 0
                feat = X_all[frame_idx] if frame_idx < len(X_all) else np.zeros((len(FEATURE_COLS),),dtype=np.float32)
            else:
                dt = 1.0/float(fps if fps>0 else 25.0)
                feat, _extras = extractor.process(frame, dt)
                running_norm.update(feat.astype(np.float32))
            short_buf.append(feat); long_buf.append(feat)

            p_short=None; p_long=None; p_fused=None
            # 短窗輸出時機
            if len(short_buf)==args.short_win and ((frame_idx+1-args.short_win)%args.short_stride==0):
                Xs=np.stack(short_buf,axis=0)  # (T,F)
                Xs=zscore_window(Xs)
                p_short=infer_short_window(short_model, torch_device, Xs)
            # 長窗輸出時機
            if long_model is not None and len(long_buf)==args.long_win and ((frame_idx+1-args.long_win)%args.long_stride==0):
                Xl=np.stack(long_buf,axis=0)
                if not use_mediapipe:
                    Xl=norm_long(Xl)
                else:
                    Xl=running_norm.norm_win(Xl)
                p_long=infer_long_window(long_model, Xl)
            # 前向填補（僅用於顯示）：本幀未更新就沿用上一筆
            disp_short = p_short if p_short is not None else last_short
            disp_long = p_long if p_long is not None else last_long

            # 視覺化/CSV 的概率採用前向填補後的融合，並做 EMA 平滑
            p_fused = fuse_mlp(fusion_model, fusion_device, disp_short, disp_long)
            ema_fused = float(args.ema_alpha) * p_fused + (1.0 - float(args.ema_alpha)) * float(ema_fused)
            last_fused = p_fused
            last_short = disp_short
            last_long = disp_long

            # 觸發判定策略（事件式）：
            # - 當本幀有新的短窗輸出時才判定；
            # - 若同幀也有新的長窗，使用融合；否則使用短窗本身。
            if p_short is not None and p_long is not None:
                fused_new = fuse_mlp(fusion_model, fusion_device, p_short, p_long)
                if args.require_long_confirm:
                    # 先處理候選到確認
                    if last_pred == 0:
                        # 嘗試進入候選或直接確認
                        if pending_confirm_frame < 0 and fused_new >= up_thr:
                            # 需要長窗確認：建立候選，尚不立刻轉 1
                            pending_confirm_frame = frame_idx
                            pending_confirm_prob = fused_new
                        # 如果已在候選期內，本幀同時有長窗=>可直接用 fused_new 當長窗訊號
                        if pending_confirm_frame >= 0:
                            # 是否本幀達到長窗確認門檻
                            if fused_new >= args.long_confirm_threshold:
                                pred = 1
                                last_pred = pred
                                pending_confirm_frame = -1
                            else:
                                # 檢查是否超過窗口
                                if frame_idx - pending_confirm_frame > args.long_confirm_window:
                                    pending_confirm_frame = -1  # 放棄
                                pred = last_pred
                        else:
                            pred = last_pred
                    else:
                        # 已是 1，套下降 hysteresis
                        if fused_new <= down_thr:
                            pred = 0
                            last_pred = pred
                        else:
                            pred = last_pred
                else:
                    # 原本 hysteresis 流程
                    if last_pred == 0 and fused_new >= up_thr:
                        pred = 1
                    elif last_pred == 1 and fused_new <= down_thr:
                        pred = 0
                    else:
                        pred = last_pred
                    last_pred = pred
            elif p_short is not None:
                if args.require_long_confirm:
                    if last_pred == 0:
                        # 只用短窗：若尚未建立候選且達上升門檻，進候選等待後續長窗輸出
                        if pending_confirm_frame < 0 and p_short >= up_thr:
                            pending_confirm_frame = frame_idx
                            pending_confirm_prob = p_short
                        # 在候選期間內但本幀無長窗輸出，仍維持 0；若超時，放棄
                        if pending_confirm_frame >= 0:
                            if frame_idx - pending_confirm_frame > args.long_confirm_window:
                                pending_confirm_frame = -1
                            pred = last_pred
                        else:
                            pred = last_pred
                    else:
                        # 已是 1，需看短窗下降是否強烈到直接清零
                        if p_short <= down_thr:
                            pred = 0
                            last_pred = pred
                        else:
                            pred = last_pred
                else:
                    if last_pred == 0 and p_short >= up_thr:
                        pred = 1
                    elif last_pred == 1 and p_short <= down_thr:
                        pred = 0
                    else:
                        pred = last_pred
                    last_pred = pred
            else:
                pred = last_pred  # 非觸發幀沿用上一次判定，避免抖動/誤判

            # 視覺化 bar
            bar = np.full((bar_h, w, 3), (60,60,60), dtype=np.uint8)
            prob = ema_fused if p_fused is not None else 0.0
            pw = int(max(0,min(1.0, prob))*w)
            color = (40,180,40) if last_pred == 1 else (40,40,200)
            bar[:, :pw] = color
            txt = f"f:{frame_idx} p(ema):{(prob or 0):.3f} up:{up_thr:.2f} dn:{down_thr:.2f}"
            cv2.putText(bar, txt, (10, int(bar_h*0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            canvas = np.vstack([frame, bar])
            writer.write(canvas)

            f.write(f"{frame_idx},{disp_short:.6f},{disp_long:.6f},{prob:.6f},{pred}\n")
            frame_idx+=1

    writer.release(); cap.release()
    if use_mediapipe:
        try:
            extractor.close()
        except Exception:
            pass
    print('Saved:', csv_path, overlay_path)

if __name__=='__main__':
    main()
