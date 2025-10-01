#!/usr/bin/env python3
"""Streaming (simulated) realtime inference for short/long window fusion.

This version does NOT recompute features from raw frames. Instead it replays
existing long-window metadata (windows_npz.npz) plus previously saved
short_on_long / long / fused probabilities to emulate timing.

Later you can replace the ProbabilityProvider with live models.
"""
from __future__ import annotations
import argparse, time, json
from pathlib import Path
import numpy as np

class ProbabilityProvider:
    def __init__(self, short_on_long: np.ndarray, long_probs: np.ndarray, fused: np.ndarray, threshold: float):
        self.short_on_long = short_on_long
        self.long_probs = long_probs
        self.fused = fused
        self.threshold = threshold
        assert len(short_on_long)==len(long_probs)==len(fused)
    def get(self, idx: int):
        ps = float(self.short_on_long[idx])
        pl = float(self.long_probs[idx])
        pf = float(self.fused[idx])
        return ps, pl, pf

class WindowStreamSimulator:
    def __init__(self, windows_npz: str, provider: ProbabilityProvider,
                 short_win: int=30, short_stride: int=15, long_win: int=75, long_stride: int=40):
        d = np.load(windows_npz, allow_pickle=True)
        self.vids = d['long_video_id'].tolist()
        self.s = d['long_start_frame'].astype(int)
        self.e = d['long_end_frame'].astype(int)
        self.labels = d['long_label'].astype(int) if 'long_label' in d else np.zeros(len(self.vids),dtype=int)
        self.provider = provider
        self.long_win = long_win
        self.long_stride = long_stride
        self.short_win = short_win
        self.short_stride = short_stride
        assert len(self.vids)==len(self.s)==len(self.e)==len(self.labels)
    def iter_events(self):
        # yield in chronological order by end frame (long window ready)
        order = np.argsort(self.e)
        for k in order:
            yield k, int(self.e[k])


def run_sim(args):
    short_on_long = np.load(args.short_on_long)
    long_probs = np.load(args.long_probs)
    fused_probs = np.load(args.fused_probs)
    meta = json.load(open(args.selected_threshold_json))
    thr = float(meta['selected_threshold'])
    provider = ProbabilityProvider(short_on_long, long_probs, fused_probs, thr)
    sim = WindowStreamSimulator(args.windows_npz, provider)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir/'stream_log.csv'
    with open(log_path,'w',encoding='utf-8') as f:
        f.write('event_index,video_id,long_end_frame,short_prob,long_prob,fused_prob,fused_pred,label\n')
        for idx, endf in sim.iter_events():
            ps, pl, pf = provider.get(idx)
            pred = int(pf >= provider.threshold)
            f.write(f"{idx},{sim.vids[idx]},{endf},{ps:.6f},{pl:.6f},{pf:.6f},{pred},{sim.labels[idx]}\n")
    print('Saved stream log to', log_path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows-npz', required=True)
    ap.add_argument('--short-on-long', required=True)
    ap.add_argument('--long-probs', required=True)
    ap.add_argument('--fused-probs', required=True)
    ap.add_argument('--selected-threshold-json', required=True)
    ap.add_argument('--out', default='immediate_stream_out')
    return ap.parse_args()

if __name__=='__main__':
    args = parse_args()
    run_sim(args)
