# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw04_fg02 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.95 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6970
- F1: 0.5543
- Recall: 0.4857
## Confusion matrix (TP/FP/FN/TN)
- TP: 51
- FP: 28
- FN: 54
- TN: 123

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 42: auc=0.7575, f1=0.6355, recall=0.6415, score=0.6629  TP: 34 FP: 20 FN: 19 TN: 55
- epoch 49: auc=0.7640, f1=0.6126, recall=0.6415, score=0.6573  TP: 34 FP: 24 FN: 19 TN: 51
- epoch 48: auc=0.7507, f1=0.6182, recall=0.6415, score=0.6563  TP: 34 FP: 23 FN: 19 TN: 52
- epoch 44: auc=0.7489, f1=0.5941, recall=0.5660, score=0.6110  TP: 30 FP: 18 FN: 23 TN: 57
