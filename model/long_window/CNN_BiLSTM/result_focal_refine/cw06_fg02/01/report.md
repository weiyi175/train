# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw06_fg02 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.1 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7072
- F1: 0.5085
- Recall: 0.4286
## Confusion matrix (TP/FP/FN/TN)
- TP: 45
- FP: 27
- FN: 60
- TN: 124

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 51: auc=0.7746, f1=0.6400, recall=0.6038, score=0.6488  TP: 32 FP: 15 FN: 21 TN: 60
- epoch 48: auc=0.7532, f1=0.6465, recall=0.6038, score=0.6465  TP: 32 FP: 14 FN: 21 TN: 61
- epoch 58: auc=0.7527, f1=0.6055, recall=0.6226, score=0.6435  TP: 33 FP: 23 FN: 20 TN: 52
- epoch 42: auc=0.7248, f1=0.6154, recall=0.6038, score=0.6315  TP: 32 FP: 19 FN: 21 TN: 56
