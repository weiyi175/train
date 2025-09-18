# Multi-Seed Precision-Aware Stability

This directory contains tooling to aggregate and analyze multi-seed runs launched via `run_multi_seed.sh`.

## Workflow
1. Launch seeds (example for precision-aware monitoring):
   ```bash
   CHECK_METRIC=precision_aware PARALLEL=3 ./run_multi_seed.sh
   ```
   Output directory: `result_multi_seed_precision_aware/seedN/<run_id>/` each with `results.json` & `report.md`.

2. Aggregate & summarize:
   ```bash
   python aggregate_multi_seed.py --base result_multi_seed_precision_aware --out multi_seed_summary.csv
   ```

## Outputs
- Console: per-seed test metrics, summary stats (mean/std/95% CI), validation↔test correlations.
- `multi_seed_summary.csv`: tabular summary of metrics.
- `multi_seed_summary_corr.json`: Pearson correlations of validation best scores vs test metrics.

## Interpretation Tips
- Prefer checkpoint metric whose validation correlation with its test counterpart and deployment KPI (e.g., precision_aware vs composite) is stable and high (>=0.7 desirable).
- Examine width of 95% CI relative to mean: narrower indicates stability; large relative CI (>10% of mean) suggests more variance and need for more runs or stratification.
- If precision-aware mean exceeds previous composite-centric selection with similar or better recall, consider adopting precision-aware checkpointing.

## Extending
- To also evaluate AUC- or composite-monitored runs, re-run seeds with `CHECK_METRIC=auc` or `composite` (produces different base directory name) and point `--base` accordingly.
