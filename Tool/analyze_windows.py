#!/usr/bin/env python3
"""Analyze sliced short/long window CSV outputs.

Usage:
  python Tool/analyze_windows.py \
    --short_csv /path/short_windows.csv \
    --long_csv /path/long_windows.csv \
    --sample 200 --export_json analysis_summary.json

Outputs concise statistics to stdout and optional JSON file.
"""
from __future__ import annotations
import csv, math, argparse, json, statistics as st, os
from collections import Counter, defaultdict
from typing import List, Dict, Any

META_PREFIX = ["video_id","scale","window_index","start_frame","end_frame","start_time","end_time","label","smoke_ratio","n_frames"]

def load_rows(path: str) -> List[Dict[str,str]]:
    with open(path,'r',encoding='utf-8') as f:
        r=csv.DictReader(f)
        return list(r)

def percentile(data: List[float], p: float) -> float:
    if not data: return math.nan
    d=sorted(data)
    k=(len(d)-1)*p
    f=math.floor(k)
    c=math.ceil(k)
    if f==c: return d[f]
    return d[f]+(d[c]-d[f])*(k-f)

def analyze(rows: List[Dict[str,str]], tag: str, sample_n: int) -> Dict[str,Any]:
    out: Dict[str,Any] = {"tag": tag}
    total=len(rows)
    if total==0:
        out['total']=0
        return out
    label_counter=Counter(r['label'] for r in rows)
    ratios=[float(r['smoke_ratio']) for r in rows]
    out.update({
        'total': total,
        'label_dist': dict(label_counter),
        'smoke_ratio_mean': sum(ratios)/total,
        'smoke_ratio_median': st.median(ratios),
        'smoke_ratio_p10': percentile(ratios,0.10),
        'smoke_ratio_p90': percentile(ratios,0.90),
        'smoke_ratio_min': min(ratios),
        'smoke_ratio_max': max(ratios),
    })
    # optional: weight for short windows
    if 'weight' in rows[0]:
        try:
            ws=[float(r.get('weight','1.0') or 1.0) for r in rows]
            out['weight_mean']=sum(ws)/len(ws)
            out['weight_min']=min(ws)
            out['weight_max']=max(ws)
        except Exception:
            pass
    # detect phase columns
    phase_cols=[c for c in rows[0].keys() if c.startswith('phase_t')]
    if phase_cols:
        phase_counter=Counter()
        sample_rows=rows[:min(sample_n,len(rows))]
        for r in sample_rows:
            for pc in phase_cols:
                v=r[pc]
                if v!='':
                    phase_counter[v]+=1
        out['phase_sample_dist']=dict(phase_counter)
        # global phase presence count (first occurrence per window) using majority
        per_window_phase=Counter()
        for r in rows:
            vals=[r[pc] for pc in phase_cols if r[pc] != '']
            if vals:
                # majority vote
                mc=Counter(vals).most_common(1)[0][0]
                per_window_phase[mc]+=1
        out['phase_majority_window_dist']=dict(per_window_phase)
    # summary metrics for long
    if tag=='LONG' and 'approach_speed_peak' in rows[0]:
        def collect(name):
            return [float(r[name]) for r in rows if r.get(name,'')!='']
        aps=collect('approach_speed_peak')
        # support both legacy and new names
        if 'hold_duration' in rows[0]:
            hfs=collect('hold_duration')
            out['hold_frames_stats']={}
        else:
            hfs=collect('hold_frames') if 'hold_frames' in rows[0] else []
        hss=collect('hold_seconds') if 'hold_seconds' in rows[0] else []
        lps=collect('leave_speed_peak')
        def stat(arr):
            if not arr: return {}
            return {
                'min': min(arr), 'max': max(arr), 'mean': sum(arr)/len(arr),
                'p10': percentile(arr,0.10), 'p90': percentile(arr,0.90)
            }
        out['approach_speed_peak_stats']=stat(aps)
        if hfs:
            out['hold_frames_stats']=stat(hfs)
        if hss:
            out['hold_seconds_stats']=stat(hss)
        out['leave_speed_peak_stats']=stat(lps)
        # fps fields if present
        if 'fps' in rows[0]:
            try:
                fps_vals=[float(r['fps']) for r in rows if r.get('fps','')!='']
                out['fps_mean']=sum(fps_vals)/len(fps_vals) if fps_vals else None
                if 'fps_estimated' in rows[0]:
                    est_ct=sum(1 for r in rows if r.get('fps_estimated','0') in ('1','True','true'))
                    out['fps_estimated_ratio']=est_ct/len(rows)
            except Exception:
                pass
    # missing check for first raw feature column
    first_raw=[c for c in rows[0].keys() if c.endswith('_raw_t0')]
    if first_raw:
        fr=first_raw[0]
        missing=sum(1 for r in rows if r.get(fr,'')=='')
        out['missing_first_raw_col']=missing
    # estimate window length
    some_raw=[c for c in rows[0] if c.endswith('_raw_t0')]
    # cannot easily infer length reliably w/out parsing pattern; provide count of time steps per feature by scanning suffixes for first feat
    if some_raw:
        feat=some_raw[0].rsplit('_raw_t',1)[0]
        t_indices=[int(c.split('_raw_t')[-1]) for c in rows[0] if c.startswith(feat+'_raw_t')]
        if t_indices:
            out['window_size_est']=max(t_indices)+1
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--short_csv', required=True)
    ap.add_argument('--long_csv', required=True)
    ap.add_argument('--sample', type=int, default=200)
    ap.add_argument('--export_json')
    args=ap.parse_args()

    short_rows=load_rows(args.short_csv)
    long_rows=load_rows(args.long_csv)
    short_stats=analyze(short_rows,'SHORT', args.sample)
    long_stats=analyze(long_rows,'LONG', args.sample)

    print("=== SHORT WINDOWS ===")
    print(json.dumps(short_stats, ensure_ascii=False, indent=2))
    print("=== LONG WINDOWS ===")
    print(json.dumps(long_stats, ensure_ascii=False, indent=2))

    if args.export_json:
        out={'short': short_stats, 'long': long_stats}
        with open(args.export_json,'w',encoding='utf-8') as f:
            json.dump(out,f,ensure_ascii=False,indent=2)
        print(f"Saved JSON summary -> {args.export_json}")

if __name__ == '__main__':
    main()
