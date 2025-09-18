#!/usr/bin/env python3
import json
from pathlib import Path
import csv


def load_csv(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # Cast numeric fields
            for k in [
                "threshold",
                "window_gate",
                "effective_batch",
                "val_best_epoch",
                "val_best_score",
                "test_auc",
                "test_f1",
                "test_recall",
                "test_TP",
                "test_FP",
                "test_FN",
                "test_TN",
                "test_composite",
            ]:
                if row.get(k) is None or row[k] == "":
                    continue
                try:
                    if k in {"val_best_epoch", "test_TP", "test_FP", "test_FN", "test_TN"}:
                        row[k] = int(float(row[k]))
                    else:
                        row[k] = float(row[k])
                except Exception:
                    pass
            rows.append(row)
    return rows


def compute_precision(tp, fp):
    tp = int(tp or 0)
    fp = int(fp or 0)
    return (tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def main():
    base = Path(__file__).parent
    agg_csv = base / "result_mask_sweep" / "aggregate.csv"
    if not agg_csv.exists():
        print(f"Aggregate not found: {agg_csv}. Run aggregate_mask_sweep.py first.")
        return

    rows = load_csv(agg_csv)
    if not rows:
        print("No rows in aggregate.csv")
        return

    # Derive precision per row
    for r in rows:
        r["test_precision"] = compute_precision(r.get("test_TP"), r.get("test_FP"))

    # Ranking by composite
    ranked = sorted(rows, key=lambda x: (x.get("test_composite") or -1), reverse=True)

    out_md = base / "result_mask_sweep" / "report.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Mask Threshold Sweep Report\n\n")
        # Summary
        f.write("## Summary (Top by composite)\n\n")
        f.write("| Rank | Threshold | Gate | AUC | F1 | Recall | Precision | Composite | Run Dir |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|:--|\n")
        for i, r in enumerate(ranked[:10], 1):
            f.write(
                f"| {i} | {r['threshold']} | {r['window_gate']} | {r['test_auc']:.4f} | {r['test_f1']:.4f} | {r['test_recall']:.4f} | {r['test_precision']:.4f} | {r['test_composite']:.4f} | {r['run_dir']} |\n"
            )
        f.write("\n")

        # Details per run
        f.write("## Details per run\n\n")
        for r in ranked:
            f.write(f"### th={r['threshold']} gate={r['window_gate']}\n\n")
            f.write(f"- Effective batch: {int(r.get('effective_batch') or 0)}\n")
            f.write(f"- Best val epoch: {r.get('val_best_epoch')} (score={r.get('val_best_score')})\n")
            f.write(
                f"- Test: AUC={r['test_auc']:.4f}, F1={r['test_f1']:.4f}, Recall={r['test_recall']:.4f}, Precision={r['test_precision']:.4f}, Composite={r['test_composite']:.4f}\n"
            )

            # Load confusion from results.json to be safe
            try:
                res_json = Path(r["run_dir"]) / "results.json"
                with res_json.open("r", encoding="utf-8") as jf:
                    jd = json.load(jf)
                t = jd.get("test", {})
                tp, fp, fn, tn = t.get("TP", 0), t.get("FP", 0), t.get("FN", 0), t.get("TN", 0)
            except Exception:
                tp, fp, fn, tn = r.get("test_TP", 0), r.get("test_FP", 0), r.get("test_FN", 0), r.get("test_TN", 0)

            f.write("- Confusion (TP/FP/FN/TN): ")
            f.write(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}\n\n")

    print(f"Report written to {out_md}")


if __name__ == "__main__":
    main()
