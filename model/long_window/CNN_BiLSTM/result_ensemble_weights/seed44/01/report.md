# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed44 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 44 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7026
- F1: 0.6794
- Recall: 0.7717
- Precision: 0.6068
- Composite Score: 0.7302 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6478 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 142
- FP: 92
- FN: 42
- TN: 78

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.5172, f1=0.5833, recall=0.9245, precision=0.4261, score=0.7407, precisionAware=0.4915  TP:49 FP:66 FN:4 TN:9
- epoch 59: auc=0.6986, f1=0.6400, recall=0.7547, precision=0.5556, score=0.7091, precisionAware=0.6095  TP:40 FP:32 FN:13 TN:43
- epoch 60: auc=0.6991, f1=0.6250, recall=0.7547, precision=0.5333, score=0.7047, precisionAware=0.5940  TP:40 FP:35 FN:13 TN:40
- epoch 55: auc=0.6702, f1=0.6202, recall=0.7547, precision=0.5263, score=0.6974, precisionAware=0.5832  TP:40 FP:36 FN:13 TN:39

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 6: auc=0.6020, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6315, composite=0.1409  TP:1 FP:0 FN:52 TN:75
- epoch 59: auc=0.6986, f1=0.6400, recall=0.7547, precision=0.5556, precisionAware=0.6095, composite=0.7091  TP:40 FP:32 FN:13 TN:43
- epoch 58: auc=0.7064, f1=0.6154, recall=0.6792, precision=0.5625, precisionAware=0.6071, composite=0.6655  TP:36 FP:28 FN:17 TN:47
- epoch 57: auc=0.6964, f1=0.6154, recall=0.6792, precision=0.5625, precisionAware=0.6051, composite=0.6635  TP:36 FP:28 FN:17 TN:47
