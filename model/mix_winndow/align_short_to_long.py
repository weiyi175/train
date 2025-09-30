#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--short-npz-dense', required=True, help='含 short_* 中介資訊的 dense 測試 npz (windows_dense_npz.npz)')
    ap.add_argument('--long-npz', required=True, help='含 long_* 中介資訊與 long_label 的測試 npz (windows_npz.npz)')
    ap.add_argument('--short-probs', required=True, help='短視窗逐短窗機率 (與 short-npz-dense 對齊)')
    ap.add_argument('--out', required=True, help='輸出對齊到 long 的 short 機率檔 (npy)')
    ap.add_argument('--min_overlap', type=int, default=1, help='判定重疊的最小 frame 數')
    return ap.parse_args()


def to_list(a):
    if isinstance(a, np.ndarray) and a.dtype == object:
        try:
            return a.tolist()
        except Exception:
            return [str(x) for x in a]
    return a.tolist() if hasattr(a, 'tolist') else list(a)


def compute_overlap(a0:int,a1:int,b0:int,b1:int) -> int:
    # intervals are inclusive? In npz they look like start_frame/end_frame ranges (end exclusive?), assume inclusive-exclusive [s,e)
    # We treat them as [s,e). If end is inclusive, off-by-one is minor for threshold>=1.
    s=max(a0,b0); e=min(a1,b1)
    return max(0, e - s)


def main():
    args = parse_args()
    d_short = np.load(args.short_npz_dense, allow_pickle=True)
    d_long = np.load(args.long_npz, allow_pickle=True)

    short_probs = np.load(args.short_probs).reshape(-1).astype('float32')

    sv = to_list(d_short['short_video_id'])
    ss = np.asarray(d_short['short_start_frame']).reshape(-1)
    se = np.asarray(d_short['short_end_frame']).reshape(-1)
    lv = to_list(d_long['long_video_id'])
    ls = np.asarray(d_long['long_start_frame']).reshape(-1)
    le = np.asarray(d_long['long_end_frame']).reshape(-1)

    assert len(short_probs) == len(sv) == len(ss) == len(se), 'short probs 與 short 索引不一致'

    # 建立 per-video 索引
    vid_to_short: Dict[str, List[Tuple[int,int,float]]] = {}
    for vid, s, e, p in zip(sv, ss, se, short_probs):
        vid_to_short.setdefault(str(vid), []).append((int(s), int(e), float(p)))

    out = np.zeros(len(lv), dtype='float32')
    for i, (vid, s0, s1) in enumerate(zip(lv, ls, le)):
        arr = vid_to_short.get(str(vid), [])
        acc = []
        for (a0, a1, p) in arr:
            if compute_overlap(a0,a1, int(s0), int(s1)) >= args.min_overlap:
                acc.append(p)
        if acc:
            out[i] = float(np.mean(acc))
        else:
            # fallback: 最近的短窗 (取距離最近一個短窗的機率)
            if arr:
                # 距離以中心點距離
                c = 0.5*(int(s0)+int(s1))
                idx = np.argmin([abs(0.5*(a0+a1)-c) for (a0,a1,_) in arr])
                out[i] = float(arr[idx][2])
            else:
                out[i] = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    print('Saved aligned short_on_long probs:', out_path, 'shape=', out.shape)

if __name__ == '__main__':
    main()
