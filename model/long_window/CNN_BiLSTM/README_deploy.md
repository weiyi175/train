Deploy README — TOP-10 logit-averaging ensemble

Purpose
- 說明如何使用 `deploy_ensemble_top10.py` 對 TOP-10 checkpoint 做 logit-averaging推論、如何做 temperature calibration（validation-based）、以及如何在生產套用已儲存的 temperature 與 threshold。

Files
- `deploy_ensemble_top10.py`: 載入 TOP-10 seeds 的 checkpoint，對輸入 windows NPZ 做逐模型預測，平均 logits，輸出機率與預測，並做 threshold 建議與 temperature-scaling 校準。
- `result_ensemble_weights/`: 產生的 artifacts（probs、calibrated probs、preds、meta、threshold grid）。

Quick usage
1) 產生 ensemble probabilities (預設把 `--windows` 當作輸入，並從 `result_ensemble_weights/ensemble_top10_results.json` 讀 selected seeds):

```bash
python3 deploy_ensemble_top10.py --weights_dir /path/to/result_ensemble_weights --windows /path/to/windows_npz.npz
```

2) 使用獨立 validation NPZ 作為 calibration source（推薦）：

```bash
python3 deploy_ensemble_top10.py --weights_dir /path/to/result_ensemble_weights \
  --windows /path/to/test_windows_npz.npz \
  --calib_windows /path/to/validation_windows_npz.npz
```

This will:
- write `ensemble_top10_probs.npy` (test probabilities)
- write `ensemble_top10_preds_0.5.npy` (default 0.5 preds)
- write `ensemble_top10_probs_calibrated.npy` and `ensemble_top10_preds_calibrated_{temp:.3f}.npy`
- write `deploy_threshold_grid.csv` and `deploy_ensemble_top10_meta.json` (contains fitted temperature and metric summaries)

Applying saved temperature and threshold in production
- Load saved `ensemble_top10_probs.npy` (or compute ensemble logits for new samples and then apply steps below).
- To apply fitted temperature T and get calibrated probabilities p_cal from raw probs p_raw:
  - Convert p_raw -> logit: logit = log(p_raw / (1-p_raw))
  - Scale: logit_scaled = logit / T
  - Back to prob: p_cal = 1 / (1 + exp(-logit_scaled))
- Use chosen threshold (from `deploy_ensemble_top10_meta.json` suggestions or your own) to get binary predictions.

Recommended workflow
1. Reserve a dedicated calibration/validation NPZ (not test). Provide it via `--calib_windows` when running `deploy_ensemble_top10.py` to fit temperature. 2. Evaluate thresholds on held-out test set (`--windows`) and pick threshold satisfying your precision/recall constraints. 3. Save the chosen threshold and fitted temperature in metadata for production.

Notes
- TensorFlow may print warnings about missing optimizer variables when loading weights; these are non-fatal for inference-only scenarios.
- If saved checkpoints are missing, the script will skip those seeds and proceed with available ones.

Example paths in this repo
- train_data (for training/validation): `/home/user/projects/train/train_data/slipce/windows_npz.npz`
- test_data: `/home/user/projects/train/test_data/slipce/windows_npz.npz`

If you want, I can now run calibration using the validation NPZ you prefer (recommend: a held-out val set)."}]}
