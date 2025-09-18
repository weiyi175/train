TCN_Residual

This folder contains an example TCN + Residual + Dropout training script and utility functions.

Usage (from this folder):

python3 train_tcn_residual.py --windows /path/to/windows_npz.npz --epochs 10 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir result

Notes:
- The script expects the NPZ to contain X and y arrays (or two arrays in order).
- Outputs are written to `result/01`, `result/02`, ... with a `report.md` and `results.json` and saved model `best_model.h5`.

Design highlights:
- Residual connection uses 1x1 Conv when channel sizes differ.
- Dropout is applied after Activation in the main path; shortcut is untouched.
- Uses causal padding for Conv1D to preserve temporal causality.
