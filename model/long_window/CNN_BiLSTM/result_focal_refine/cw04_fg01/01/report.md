# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw04_fg01 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.95 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7306
- F1: 0.5455
- Recall: 0.4571
## Confusion matrix (TP/FP/FN/TN)
- TP: 48
- FP: 23
- FN: 57
- TN: 128

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 52: auc=0.7187, f1=0.5766, recall=0.6038, score=0.6186  TP: 32 FP: 26 FN: 21 TN: 49
- epoch 46: auc=0.7255, f1=0.5743, recall=0.5472, score=0.5910  TP: 29 FP: 19 FN: 24 TN: 56
- epoch 49: auc=0.7389, f1=0.5895, recall=0.5283, score=0.5888  TP: 28 FP: 14 FN: 25 TN: 61
- epoch 34: auc=0.6855, f1=0.5745, recall=0.5094, score=0.5642  TP: 27 FP: 14 FN: 26 TN: 61
