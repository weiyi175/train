#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import cv2

"""可視化融合結果：將長視窗 (75) 的 fused 機率對應到影片時間範圍，輸出帶標註影片與每幀 CSV。
資料來源：
  - windows_npz.npz (long) 提供 long_video_id / long_start_frame / long_end_frame
  - fused_probs.npy 與 selected_threshold.json (MLP 融合輸出)
輸出：
  out_root/<video_id>/overlay.mp4  (含每幀文字與顏色)
  out_root/<video_id>/frame_probs.csv (frame,prob,label,pred)
假設：影片 FPS 與 frame index 對齊 windows_npz 中的 frame 編號。
"""

COLOR_BG = (25,25,25)
COLOR_TEXT = (255,255,255)
COLOR_BAR_BG = (60,60,60)
COLOR_BAR_POS = (40,180,40)
COLOR_BAR_NEG = (40,40,200)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows-npz', required=True, help='對應測試集 long windows 的 npz (test_data/slipce_thresh040/windows_npz.npz)')
    ap.add_argument('--fused-probs', required=True, help='MLP 融合輸出的 fused_probs.npy (長視窗數量)')
    ap.add_argument('--selected-threshold-json', required=True, help='包含 selected_threshold 的 JSON')
    ap.add_argument('--videos-root', required=True, help='原始影片資料夾根目錄')
    ap.add_argument('--out-root', required=True, help='輸出根目錄 (將建立子資料夾)')
    ap.add_argument('--max-videos', type=int, default=50, help='最多處理多少支影片 (避免一次太多)')
    ap.add_argument('--resize-width', type=int, default=960)
    ap.add_argument('--bar-height', type=int, default=40)
    ap.add_argument('--font-scale', type=float, default=0.6)
    ap.add_argument('--thickness', type=int, default=1)
    ap.add_argument('--skip-existing', action='store_true')
    return ap.parse_args()


def load_windows(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    vids = d['long_video_id'].tolist()
    s = d['long_start_frame'].astype(int)
    e = d['long_end_frame'].astype(int)
    labels = d['long_label'].astype(int) if 'long_label' in d else None
    return vids, s, e, labels


def build_video_map(vids, s, e, probs, labels):
    # 依影片 -> 列表[(start,end,prob,label)]
    mp = {}
    for vid, a, b, p, lab in zip(vids, s, e, probs, labels):
        mp.setdefault(str(vid), []).append((int(a), int(b), float(p), int(lab)))
    return mp


def expand_frame_probs(ranges, total_frames: int, strategy: str = 'max'):
    # ranges: list of (s,e,p,label) with frame 半開區間 [s,e)
    frame_prob = np.zeros(total_frames, dtype='float32')
    frame_cnt = np.zeros(total_frames, dtype='int32')
    frame_lab = np.zeros(total_frames, dtype='int32') - 1
    for (s,e,p,lab) in ranges:
        for f in range(s, e):
            if 0 <= f < total_frames:
                if strategy == 'max':
                    if frame_cnt[f] == 0 or p > frame_prob[f]:
                        frame_prob[f] = p
                        frame_lab[f] = lab
                else:  # average accumulate
                    frame_prob[f] += p
                    frame_cnt[f] += 1
                    frame_lab[f] = lab
    if strategy != 'max':
        nz = frame_cnt > 0
        frame_prob[nz] /= np.maximum(1, frame_cnt[nz])
    return frame_prob, frame_lab


def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, total


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    vids, s, e, labels = load_windows(args.windows_npz)
    probs = np.load(args.fused_probs).astype('float32')
    meta = json.load(open(args.selected_threshold_json))
    thr = float(meta['selected_threshold'])

    assert len(probs) == len(vids) == len(s) == len(e) == len(labels), '長視窗數量不相符'

    vid_map = build_video_map(vids, s, e, probs, labels)

    out_root = Path(args.out_root); ensure_dir(out_root)
    processed = 0
    for vid, ranges in vid_map.items():
        if processed >= args.max_videos:
            break
        video_path = Path(args.videos_root)/f"{vid}.mp4"
        if not video_path.exists():
            # 嘗試其他延伸
            alts = list(Path(args.videos_root).glob(f"{vid}.*"))
            if alts:
                video_path = alts[0]
            else:
                print('[WARN] 找不到影片', vid)
                continue
        cap, total_frames = open_video(str(video_path))
        if cap is None:
            print('[WARN] 無法開啟影片', video_path)
            continue

        # 展開 frame probability
        frame_prob, frame_lab = expand_frame_probs([(a,b,p,lab) for (a,b,p,lab) in ranges], total_frames)
        frame_pred = (frame_prob >= thr).astype(int)

        out_dir = out_root/vid; ensure_dir(out_dir)
        overlay_path = out_dir/'overlay.mp4'
        csv_path = out_dir/'frame_probs.csv'
        if args.skip_existing and overlay_path.exists():
            print('[SKIP]', vid)
            cap.release()
            processed += 1
            continue

        # 先寫 CSV
        import csv
        with open(csv_path,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f)
            w.writerow(['frame','prob','label','pred'])
            for idx,(p,lab,pd) in enumerate(zip(frame_prob, frame_lab, frame_pred)):
                w.writerow([idx, f"{p:.6f}", lab, pd])

        # 讀第一幀取尺寸
        ret, frame = cap.read()
        if not ret:
            cap.release(); print('[WARN] 空影片', video_path); continue
        h0, w0 = frame.shape[:2]
        new_w = args.resize_width
        scale = new_w / w0
        new_h = int(h0 * scale)
        bar_h = args.bar_height
        canvas_h = new_h + bar_h

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(overlay_path), fourcc, cap.get(cv2.CAP_PROP_FPS) or 25.0, (new_w, canvas_h))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 回到第一幀
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (new_w, new_h))
            prob = frame_prob[idx] if idx < len(frame_prob) else 0.0
            pred = frame_pred[idx] if idx < len(frame_pred) else 0
            lab = frame_lab[idx] if idx < len(frame_lab) else -1

            bar = np.full((bar_h, new_w, 3), COLOR_BAR_BG, dtype=np.uint8)
            # prob bar
            pw = int(prob * new_w)
            color = COLOR_BAR_POS if prob >= thr else COLOR_BAR_NEG
            bar[:, :pw] = color
            txt = f"frame:{idx} prob:{prob:.3f} thr:{thr:.2f} pred:{pred} lab:{lab}"
            cv2.putText(bar, txt, (10, int(bar_h*0.7)), font, args.font_scale, COLOR_TEXT, args.thickness, cv2.LINE_AA)

            canvas = np.vstack([frame, bar])
            writer.write(canvas)
            idx += 1
        writer.release(); cap.release()
        print('[OK]', vid, '->', overlay_path)
        processed += 1

    print('Done. Processed videos:', processed)

if __name__ == '__main__':
    main()
