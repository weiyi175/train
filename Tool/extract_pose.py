import os
import sys
import csv
import json
import argparse
import math
from typing import List, Tuple, Dict, Optional
import cv2

try:
    import mediapipe as mp
except ImportError:
    print("請先安裝 mediapipe: pip install mediapipe")
    sys.exit(1)

# 需要的骨架點：0~6 + 9~22
SELECTED_LANDMARKS = list(range(0, 7)) + list(range(9, 23))

def parse_via_csv_folder(csv_folder: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    回傳: { video_filename: [(start_sec, end_sec), ...] }
    若無法解析任何時間段，回傳空 dict (代表全部影格都要處理)。
    支援欄位或 JSON 中包含:
      - start / end
      - t1 / t2
      - temporal_segment_start / temporal_segment_end
    """
    intervals: Dict[str, List[Tuple[float, float]]] = {}
    if not os.path.isdir(csv_folder):
        return intervals
    for name in os.listdir(csv_folder):
        if not name.lower().endswith('.csv'):
            continue
        path = os.path.join(csv_folder, name)
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = (row.get('filename') or row.get('file_name') or row.get('file') or '').strip()
                    if not filename:
                        continue
                    json_candidates = []
                    for key in ['region_shape_attributes','region_attributes','file_attributes']:
                        v = row.get(key)
                        if v and v.strip().startswith('{') and v.strip().endswith('}'):
                            json_candidates.append(v)
                    timing = {}
                    for jc in json_candidates:
                        try:
                            obj = json.loads(jc)
                            if isinstance(obj, dict):
                                timing.update(obj)
                        except Exception:
                            pass
                    for k,v in row.items():
                        lk = k.lower()
                        if any(x in lk for x in ['start','end','t1','t2']):
                            timing[lk] = v
                    def _to_float(x):
                        if x is None: return None
                        try:
                            return float(str(x).strip())
                        except:
                            return None
                    start_candidates = [timing.get(k) for k in ['start','t1','temporal_segment_start','segment_start'] if k in timing]
                    end_candidates = [timing.get(k) for k in ['end','t2','temporal_segment_end','segment_end'] if k in timing]
                    start_val = None
                    for c in start_candidates:
                        start_val = _to_float(c)
                        if start_val is not None: break
                    end_val = None
                    for c in end_candidates:
                        end_val = _to_float(c)
                        if end_val is not None: break
                    if start_val is not None and end_val is not None and end_val > start_val:
                        intervals.setdefault(filename, []).append((start_val, end_val))
        except Exception as e:
            print(f"⚠️ 讀取標記檔失敗 {path}: {e}")
    for fn, segs in intervals.items():
        segs.sort()
        merged = []
        cur_s, cur_e = segs[0]
        for s,e in segs[1:]:
            if s <= cur_e + 1e-6:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        intervals[fn] = merged
    return intervals

def is_time_in_intervals(t: float, segs: List[Tuple[float, float]]) -> bool:
    if not segs:
        return True
    for s,e in segs:
        if s <= t <= e:
            return True
    return False

def extract_pose_from_video(video_path: str,
                            intervals: List[Tuple[float, float]],
                            output_csv: str,
                            model_complexity: int = 1,
                            min_detection_confidence: float = 0.5,
                            min_tracking_confidence: float = 0.5,
                            zero_first_velocity: bool = True,
                            progress_every: int = 500,
                            visibility_threshold: float = 0.0,
                            retry_times: int = 0) -> bool:
    """抽取單支影片骨架.

    visibility_threshold: landmark visibility 低於此值則視為缺失 (x,y,z 置 NaN, v 保留)
    retry_times: 失敗重新嘗試次數 (重新初始化 VideoCapture 與 Pose)
    回傳: 是否成功
    """
    attempt = 0
    while attempt <= retry_times:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("無法開啟影片")
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            if fps <= 0: fps = 30.0
            dt = 1.0 / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                enable_segmentation=False,
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ['frame','time_sec','video_width','video_height','fps']
                for idx in SELECTED_LANDMARKS:
                    header += [f'l{idx}_x', f'l{idx}_y', f'l{idx}_z', f'l{idx}_v', f'l{idx}_vx', f'l{idx}_vy', f'l{idx}_vz']
                writer.writerow(header)
                prev_coords = None
                last_written_frame_idx: Optional[int] = None
                frame_idx = 0
                written = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    time_sec = frame_idx * dt
                    if not is_time_in_intervals(time_sec, intervals):
                        frame_idx += 1
                        continue
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    try:
                        result = pose.process(image_rgb)
                    except Exception as e:
                        raise RuntimeError(f'pose.process 失敗: {e}')
                    row = [frame_idx, f'{time_sec:.6f}', width, height, f'{fps:.6f}']
                    coords_now = {}
                    if result.pose_landmarks:
                        lm_list = result.pose_landmarks.landmark
                        for idx in SELECTED_LANDMARKS:
                            lm = lm_list[idx]
                            x,y,z,v = lm.x, lm.y, lm.z, lm.visibility
                            if v < visibility_threshold:
                                coords_now[idx] = (math.nan, math.nan, math.nan, v)
                            else:
                                coords_now[idx] = (x,y,z,v)
                    for idx in SELECTED_LANDMARKS:
                        if idx not in coords_now:
                            coords_now[idx] = (math.nan, math.nan, math.nan, math.nan)
                    frame_gap = 1
                    if last_written_frame_idx is not None:
                        gap = frame_idx - last_written_frame_idx
                        if gap >= 1: frame_gap = gap
                    effective_dt = dt * frame_gap
                    if effective_dt <= 0: effective_dt = dt
                    for idx in SELECTED_LANDMARKS:
                        x,y,z,v = coords_now[idx]
                        if prev_coords and idx in prev_coords:
                            px,py,pz = prev_coords[idx]
                            if not any(math.isnan(val) for val in [x,y,z,px,py,pz]):
                                vx = (x-px)/effective_dt
                                vy = (y-py)/effective_dt
                                vz = (z-pz)/effective_dt
                            else:
                                vx=vy=vz=math.nan
                        else:
                            vx=vy=vz = (0.0 if zero_first_velocity else math.nan)
                        row += [f'{x:.6f}' if not math.isnan(x) else '',
                                f'{y:.6f}' if not math.isnan(y) else '',
                                f'{z:.6f}' if not math.isnan(z) else '',
                                f'{v:.6f}' if not math.isnan(v) else '',
                                f'{vx:.6f}' if not math.isnan(vx) else '',
                                f'{vy:.6f}' if not math.isnan(vy) else '',
                                f'{vz:.6f}' if not math.isnan(vz) else '']
                    prev_coords = {i:(c[0],c[1],c[2]) for i,c in coords_now.items()}
                    writer.writerow(row)
                    written += 1
                    last_written_frame_idx = frame_idx
                    frame_idx += 1
                    if progress_every and written % progress_every == 0:
                        print(f'   ... processed {written} frames for {os.path.basename(video_path)} (current frame_idx={frame_idx})')
            cap.release()
            print(f'✅ 完成: {os.path.basename(video_path)} -> {output_csv} (frames={written})')
            return True
        except Exception as e:
            attempt += 1
            print(f'❌ 失敗 ({attempt}/{retry_times}) {os.path.basename(video_path)}: {e}')
            try: cap.release()
            except Exception: pass
            if attempt > retry_times:
                print(f'⚠️ 放棄: {video_path}')
                return False
            else:
                print('↻ 重新嘗試...')
    return False

def batch_process_videos(csv_folder: str, video_folder: str, output_folder: str,
                         model_complexity: int,
                         min_detection_confidence: float,
                         min_tracking_confidence: float,
                         zero_first_velocity: bool,
                         progress_every: int,
                         visibility_threshold: float,
                         retry_times: int,
                         skip_existing: bool = False,
                         start_index: int = 0,
                         max_count: int = -1,
                         include_set: Optional[set] = None):
    intervals_map = parse_via_csv_folder(csv_folder)
    # 若 intervals_map 為空，代表所有影片全時段
    video_files = [f for f in os.listdir(video_folder)
                   if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))]
    video_files.sort()

    total_videos = len(video_files)
    if start_index < 0:
        start_index = 0
    if start_index >= total_videos:
        print(f"⚠️ start_index={start_index} 超出範圍 (total={total_videos})")
        return
    if max_count is not None and max_count >= 0:
        end_index = min(start_index + max_count, total_videos)
    else:
        end_index = total_videos
    selected_videos = video_files[start_index:end_index]
    if include_set is not None:
        before = len(selected_videos)
        selected_videos = [vf for vf in selected_videos if (os.path.splitext(vf)[0] in include_set) or (vf in include_set)]
        print(f"[INFO] include-list 過濾: 從 {before} 篩到 {len(selected_videos)} 支")

    print(f"[INFO] 總影片數={total_videos}, 將處理區間 [{start_index}:{end_index}) => {len(selected_videos)} 支")

    if not video_files:
        print("⚠️ 找不到影片檔")
        return

    def _has_frames(csv_path: str) -> bool:
        if not os.path.isfile(csv_path):
            return False
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # header + 至少一行資料
                lines = 0
                for _ in f:
                    lines += 1
                    if lines >= 2:
                        return True
            return False
        except Exception:
            return False

    processed = 0
    skipped = 0
    for vf in selected_videos:
        video_path = os.path.join(video_folder, vf)
        segs = intervals_map.get(vf, [])
        out_name = os.path.splitext(vf)[0] + "_pose.csv"
        output_csv = os.path.join(output_folder, out_name)
        if skip_existing and _has_frames(output_csv):
            skipped += 1
            if skipped % 20 == 0:
                print(f"[Skip] 已略過 {skipped} 個已存在輸出 (最新: {out_name})")
            continue
        extract_pose_from_video(
            video_path,
            segs,
            output_csv,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            zero_first_velocity=zero_first_velocity,
            progress_every=progress_every,
            visibility_threshold=visibility_threshold,
            retry_times=retry_times,
        )
        processed += 1

    print(f"[INFO] 完成批次: 實際處理 {processed} 支影片, 略過 {skipped} 支 (skip_existing={skip_existing})")

# ======================
# 📌 執行區
# ======================
if __name__ == "__main__":
    # 預設路徑
    csv_folder = "/home/user/projects/train/train_data/VIA_smoke_nosmoke"
    video_folder = "/home/user/projects/train/train_data/video"
    output_folder = "/home/user/projects/train/train_data/extract_pose"

    def _norm_path(p: str) -> str:
        if p is None:
            return p
        p = str(p).strip()
        if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
            p = p[1:-1].strip()
        p = os.path.expanduser(p)
        try:
            return os.path.abspath(p)
        except Exception:
            return p

    parser = argparse.ArgumentParser(description="Extract selected MediaPipe Pose landmarks (0-6 & 9-22) with optional VIA time segments")
    parser.add_argument('--csv-folder', '-c', help='VIA CSV folder (absolute path)')
    parser.add_argument('--video-folder', '-v', help='Video folder (absolute path)')
    parser.add_argument('--output-folder', '-o', help='Output folder (absolute path)')
    parser.add_argument('--model-complexity', type=int, default=1, choices=[0,1,2], help='MediaPipe Pose model complexity (0/1/2)')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5)
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5)
    parser.add_argument('--no-zero-first-velocity', action='store_true', help='Do not force first frame velocities to 0')
    parser.add_argument('--visibility-threshold', type=float, default=0.0, help='Landmark visibility threshold; below => coordinates NaN')
    parser.add_argument('--retry-times', type=int, default=0, help='Retry times on failure')
    parser.add_argument('--progress-every', type=int, default=500, help='Log progress every N written frames (0 to disable)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip already generated pose csv (with >=1 data row)')
    parser.add_argument('--start-index', type=int, default=0, help='Start index in sorted video list')
    parser.add_argument('--max-count', type=int, default=-1, help='Max number of videos to process from start-index (-1 for all)')
    parser.add_argument('--include-list', type=str, help='File with video base names (or filenames) to include, one per line')
    args, _ = parser.parse_known_args()

    include_set = None
    if args.include_list:
        inc_path = _norm_path(args.include_list)
        if os.path.isfile(inc_path):
            try:
                with open(inc_path, 'r', encoding='utf-8') as f:
                    include_set = {line.strip() for line in f if line.strip() and not line.strip().startswith('#')}
                print(f"[INFO] 讀取 include-list: {inc_path} (共 {len(include_set)} 條)")
                if include_set:
                    print(f"       範例: {list(include_set)[:5]}")
            except Exception as e:
                print(f"⚠️ include-list 讀取失敗: {e}")
        else:
            print(f"⚠️ 找不到 include-list 檔案: {inc_path}")

    csv_folder = _norm_path(args.csv_folder) if args.csv_folder else _norm_path(csv_folder)
    video_folder = _norm_path(args.video_folder) if args.video_folder else _norm_path(video_folder)
    output_folder = _norm_path(args.output_folder) if args.output_folder else _norm_path(output_folder)

    try:
        print(f"[INFO] 目前執行檔: {__file__}")
    except NameError:
        pass
    print(f"[INFO] 解析後 csv_folder   = {csv_folder}")
    print(f"[INFO] 解析後 video_folder = {video_folder}")
    print(f"[INFO] 解析後 output_folder= {output_folder}")

    if not os.path.exists(video_folder):
        print(f"⚠️ video_folder not found: {video_folder}")
        sys.exit(1)
    if not os.path.exists(csv_folder):
        print(f"⚠️ csv_folder not found: {csv_folder}")
        sys.exit(1)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    batch_process_videos(
        csv_folder,
        video_folder,
        output_folder,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        zero_first_velocity=not args.no_zero_first_velocity,
        progress_every=args.progress_every if args.progress_every > 0 else 0,
        visibility_threshold=args.visibility_threshold,
        retry_times=args.retry_times,
        skip_existing=args.skip_existing,
        start_index=args.start_index,
        max_count=args.max_count,
        include_set=include_set,
    )
    print("✅ 所有影片處理完成！")