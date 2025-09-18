# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw08_fg05 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7498
- F1: 0.5263
- Recall: 0.4286
## Confusion matrix (TP/FP/FN/TN)
- TP: 45
- FP: 21
- FN: 60
- TN: 130

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 41: auc=0.7024, f1=0.6102, recall=0.6792, score=0.6632  TP: 36 FP: 29 FN: 17 TN: 46
- epoch 48: auc=0.7104, f1=0.5902, recall=0.6792, score=0.6588  TP: 36 FP: 33 FN: 17 TN: 42
- epoch 49: auc=0.7213, f1=0.5965, recall=0.6415, score=0.6440  TP: 34 FP: 27 FN: 19 TN: 48
- epoch 43: auc=0.7442, f1=0.6111, recall=0.6226, score=0.6435  TP: 33 FP: 22 FN: 20 TN: 53
