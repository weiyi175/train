# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw01_fg10 --focal_alpha 0.35 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.75 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7289
- F1: 0.5872
- Recall: 0.6095
## Confusion matrix (TP/FP/FN/TN)
- TP: 64
- FP: 49
- FN: 41
- TN: 102

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 67: auc=0.7836, f1=0.6880, recall=0.8113, score=0.7688  TP: 43 FP: 29 FN: 10 TN: 46
- epoch 50: auc=0.7844, f1=0.6842, recall=0.7358, score=0.7301  TP: 39 FP: 22 FN: 14 TN: 53
- epoch 37: auc=0.7731, f1=0.6783, recall=0.7358, score=0.7260  TP: 39 FP: 23 FN: 14 TN: 52
- epoch 65: auc=0.8010, f1=0.6729, recall=0.6792, score=0.7017  TP: 36 FP: 18 FN: 17 TN: 57
