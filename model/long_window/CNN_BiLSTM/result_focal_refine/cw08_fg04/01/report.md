# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw08_fg04 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7102
- F1: 0.5500
- Recall: 0.5238
## Confusion matrix (TP/FP/FN/TN)
- TP: 55
- FP: 40
- FN: 50
- TN: 111

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 51: auc=0.7469, f1=0.6372, recall=0.6792, score=0.6802  TP: 36 FP: 24 FN: 17 TN: 51
- epoch 56: auc=0.7192, f1=0.6018, recall=0.6415, score=0.6451  TP: 34 FP: 26 FN: 19 TN: 49
- epoch 45: auc=0.7160, f1=0.6000, recall=0.6226, score=0.6345  TP: 33 FP: 24 FN: 20 TN: 51
- epoch 40: auc=0.7200, f1=0.5962, recall=0.5849, score=0.6153  TP: 31 FP: 20 FN: 22 TN: 55
