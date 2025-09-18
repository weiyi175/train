# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw06_fg01 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.1 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7059
- F1: 0.3889
- Recall: 0.2667
## Confusion matrix (TP/FP/FN/TN)
- TP: 28
- FP: 11
- FN: 77
- TN: 140

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 40: auc=0.7197, f1=0.5275, recall=0.4528, score=0.5286  TP: 24 FP: 14 FN: 29 TN: 61
- epoch 39: auc=0.7200, f1=0.4773, recall=0.3962, score=0.4853  TP: 21 FP: 14 FN: 32 TN: 61
- epoch 35: auc=0.7311, f1=0.4819, recall=0.3774, score=0.4795  TP: 20 FP: 10 FN: 33 TN: 65
- epoch 34: auc=0.7356, f1=0.4578, recall=0.3585, score=0.4637  TP: 19 FP: 11 FN: 34 TN: 64
