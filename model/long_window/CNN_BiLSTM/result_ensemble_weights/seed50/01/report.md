# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed50 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 50 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7070
- F1: 0.6649
- Recall: 0.6630
- Precision: 0.6667
- Composite Score: 0.6724 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6742 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 122
- FP: 61
- FN: 62
- TN: 109

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.4823, f1=0.5856, recall=1.0000, precision=0.4141, score=0.7721, precisionAware=0.4792  TP:53 FP:75 FN:0 TN:0
- epoch 60: auc=0.7522, f1=0.6723, recall=0.7547, precision=0.6061, score=0.7295, precisionAware=0.6552  TP:40 FP:26 FN:13 TN:49
- epoch 2: auc=0.5374, f1=0.5595, recall=0.8868, precision=0.4087, score=0.7187, precisionAware=0.4797  TP:47 FP:68 FN:6 TN:7
- epoch 59: auc=0.7479, f1=0.6667, recall=0.7358, precision=0.6094, score=0.7175, precisionAware=0.6543  TP:39 FP:25 FN:14 TN:50

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7522, f1=0.6723, recall=0.7547, precision=0.6061, precisionAware=0.6552, composite=0.7295  TP:40 FP:26 FN:13 TN:49
- epoch 59: auc=0.7479, f1=0.6667, recall=0.7358, precision=0.6094, precisionAware=0.6543, composite=0.7175  TP:39 FP:25 FN:14 TN:50
- epoch 58: auc=0.7421, f1=0.6549, recall=0.6981, precision=0.6167, precisionAware=0.6532, composite=0.6939  TP:37 FP:23 FN:16 TN:52
- epoch 47: auc=0.7150, f1=0.6239, recall=0.6415, precision=0.6071, precisionAware=0.6337, composite=0.6509  TP:34 FP:22 FN:19 TN:53
