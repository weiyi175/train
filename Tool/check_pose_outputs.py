#!/usr/bin/env python
"""簡單檢查: 原始影片 是否都有對應 *_pose.csv 輸出

預設路徑:
  影片:   /home/user/projects/train/train_data/video
  輸出:   /home/user/projects/train/train_data/extract_pose

對應規則:
  <basename>.<ext>  對應  <basename>_pose.csv

輸出內容:
  - 總影片數 / 有輸出數 / 有資料(>1行) / 只有標頭 / 缺漏
  - 列出前若干缺漏與空檔
  - 產生 missing_pose_list.txt (僅列缺漏 basename)

使用:
  python check_pose_outputs.py
  python check_pose_outputs.py --video-dir <dir> --pose-dir <dir> --save-list my_missing.txt
"""
from __future__ import annotations
import os
import argparse
from typing import List, Tuple

VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.avi')

def scan_videos(video_dir: str) -> List[str]:
    return sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(VIDEO_EXTS) and os.path.isfile(os.path.join(video_dir, f))
    ])

def scan_pose_outputs(pose_dir: str) -> List[str]:
    return sorted([
        f for f in os.listdir(pose_dir)
        if f.endswith('_pose.csv') and os.path.isfile(os.path.join(pose_dir, f))
    ])

def has_data(path: str) -> bool:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # header + 至少一行資料
            lines = 0
            for _ in f:
                lines += 1
                if lines >= 2:
                    return True
        return False
    except Exception:
        return False

def analyze(video_dir: str, pose_dir: str) -> Tuple[dict, list, list]:
    videos = scan_videos(video_dir)
    video_bases = [os.path.splitext(v)[0] for v in videos]
    pose_files = scan_pose_outputs(pose_dir)
    pose_bases = [p[:-9] for p in pose_files]  # remove _pose.csv

    existing = set(pose_bases)
    missing = sorted(set(video_bases) - existing)

    with_data = []
    empty = []
    for base, fname in zip(pose_bases, pose_files):
        full = os.path.join(pose_dir, fname)
        (with_data if has_data(full) else empty).append(base)

    summary = {
        'total_videos': len(video_bases),
        'total_pose_outputs': len(pose_bases),
        'pose_with_data': len(with_data),
        'pose_empty_or_header_only': len(empty),
        'missing_outputs': len(missing),
    }
    return summary, missing, empty

def main():
    parser = argparse.ArgumentParser(description='檢查影片與 pose 輸出缺漏')
    parser.add_argument('--video-dir', default='/home/user/projects/train/train_data/video', help='原始影片資料夾')
    parser.add_argument('--pose-dir', default='/home/user/projects/train/train_data/extract_pose', help='pose 輸出資料夾')
    parser.add_argument('--save-list', default='/home/user/projects/train/Tool/missing_pose_list.txt', help='缺漏名單輸出檔名 (相對目前工作目錄)')
    parser.add_argument('--show', type=int, default=30, help='顯示前 N 筆缺漏與空檔')
    args = parser.parse_args()

    video_dir = os.path.abspath(args.video_dir)
    pose_dir = os.path.abspath(args.pose_dir)
    if not os.path.isdir(video_dir):
        print(f'⚠️ 找不到影片資料夾: {video_dir}')
        return
    if not os.path.isdir(pose_dir):
        print(f'⚠️ 找不到輸出資料夾: {pose_dir}')
        return

    summary, missing, empty = analyze(video_dir, pose_dir)
    print('=== 檢查結果 ===')
    for k,v in summary.items():
        print(f'{k}: {v}')

    n = args.show
    if missing:
        print(f'\n缺漏(前 {n}):')
        for m in missing[:n]:
            print('MISSING', m)
    else:
        print('\n缺漏: 無')

    if empty:
        print(f'\n空檔或僅標頭(前 {n}):')
        for e in empty[:n]:
            print('EMPTY', e)
    else:
        print('\n空檔: 無')

    # 寫檔
    try:
        with open(args.save_list, 'w', encoding='utf-8') as f:
            for m in missing:
                f.write(m + '\n')
        print(f'\n已寫入缺漏名單: {args.save_list} (共 {len(missing)} 行)')
    except Exception as e:
        print(f'⚠️ 寫入缺漏名單失敗: {e}')

    print('\n提示: 可用以下指令只補缺漏 (若 extract_pose.py 支援 include-list):')
    print('  python extract_pose.py --include-list missing_pose_list.txt --skip-existing --visibility-threshold 0.5')

if __name__ == '__main__':
    main()
