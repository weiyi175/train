# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.5 --focal_gamma_end 0.5 --curriculum_epochs 1 --run_seed 23
```

## Test metrics
- AUC: 0.5836
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 10: auc=0.6337, f1=0.0727, recall=0.0377, score=0.2270  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 9: auc=0.6319, f1=0.0727, recall=0.0377, score=0.2265  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 11: auc=0.6264, f1=0.0714, recall=0.0377, score=0.2244  TP: 2 FP: 1 FN: 51 TN: 74
- epoch 13: auc=0.6196, f1=0.0714, recall=0.0377, score=0.2224  TP: 2 FP: 1 FN: 51 TN: 74
