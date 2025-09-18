#!/usr/bin/env python3
import json
import csv
from pathlib import Path


def find_results(root: Path):
    for p in root.rglob("results.json"):
        if root.as_posix() in p.as_posix():
            yield p


def composite(auc, f1, recall):
    try:
        return 0.5 * float(recall) + 0.3 * float(f1) + 0.2 * float(auc)
    except Exception:
        return None


def main():
    base = Path(__file__).parent
    out_root = base / (os.environ.get("RESULT_BASE") or "result_focal_sweep")
    if not out_root.exists():
        print(f"No focal sweep dir: {out_root}")
        return
    rows = []
    for p in sorted(find_results(out_root)):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        params = data.get("params", {})
        test = data.get("test", {})
        auc = test.get("auc"); f1 = test.get("f1"); rec = test.get("recall")
        comp = composite(auc, f1, rec)
        rows.append({
            "run_dir": p.parent.as_posix(),
            "auc": auc,
            "f1": f1,
            "recall": rec,
            "TP": test.get("TP"),
            "FP": test.get("FP"),
            "FN": test.get("FN"),
            "TN": test.get("TN"),
            "composite": comp,
            "alpha": params.get("focal_alpha"),
            "gamma_start": params.get("focal_gamma_start"),
            "gamma_end": params.get("focal_gamma_end"),
            "cw_neg": params.get("class_weight_neg"),
            "cw_pos": params.get("class_weight_pos"),
            "mask_threshold": params.get("mask_threshold"),
        })

    if not rows:
        print("No results found.")
        return

    out_csv = out_root / "aggregate.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    top = sorted(rows, key=lambda r: (r["composite"] or -1), reverse=True)[:10]
    print(f"Wrote {len(rows)} rows -> {out_csv}")
    print("Top 5 by composite:")
    for r in top[:5]:
        print(f"cw={r['cw_neg']}:{r['cw_pos']} alpha={r['alpha']} gamma_end={r['gamma_end']} -> comp={r['composite']:.4f} auc={r['auc']:.4f} f1={r['f1']:.4f} rec={r['recall']:.4f}")


if __name__ == "__main__":
    import os
    main()
