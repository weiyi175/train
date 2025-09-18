# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw02_fg08 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7169
- F1: 0.4815
- Recall: 0.3714
## Confusion matrix (TP/FP/FN/TN)
- TP: 39
- FP: 18
- FN: 66
- TN: 133

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 31: auc=0.6800, f1=0.6116, recall=0.6981, score=0.6685  TP: 37 FP: 31 FN: 16 TN: 44
- epoch 47: auc=0.7281, f1=0.6286, recall=0.6226, score=0.6455  TP: 33 FP: 19 FN: 20 TN: 56
- epoch 41: auc=0.6918, f1=0.6071, recall=0.6415, score=0.6413  TP: 34 FP: 25 FN: 19 TN: 50
- epoch 29: auc=0.6855, f1=0.6038, recall=0.6038, score=0.6201  TP: 32 FP: 21 FN: 21 TN: 54
