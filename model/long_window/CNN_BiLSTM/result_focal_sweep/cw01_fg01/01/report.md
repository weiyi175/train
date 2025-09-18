# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw01_fg01 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7313
- F1: 0.5962
- Recall: 0.5905
## Confusion matrix (TP/FP/FN/TN)
- TP: 62
- FP: 41
- FN: 43
- TN: 110

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 58: auc=0.7313, f1=0.5862, recall=0.6415, score=0.6429  TP: 34 FP: 29 FN: 19 TN: 46
- epoch 64: auc=0.7293, f1=0.5893, recall=0.6226, score=0.6340  TP: 33 FP: 26 FN: 20 TN: 49
- epoch 66: auc=0.7099, f1=0.5766, recall=0.6038, score=0.6168  TP: 32 FP: 26 FN: 21 TN: 49
- epoch 70: auc=0.7467, f1=0.6122, recall=0.5660, score=0.6160  TP: 30 FP: 15 FN: 23 TN: 60
