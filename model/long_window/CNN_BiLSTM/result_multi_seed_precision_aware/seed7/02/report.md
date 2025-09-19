# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed7 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 7 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7172
- F1: 0.6350
- Recall: 0.5815
- Precision: 0.6993
- Composite Score: 0.6247 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6836 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 107
- FP: 46
- FN: 77
- TN: 124

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 43: auc=0.7364, f1=0.6415, recall=0.6415, precision=0.6415, score=0.6605, precisionAware=0.6605  TP:34 FP:19 FN:19 TN:56
- epoch 51: auc=0.7130, f1=0.5785, recall=0.6604, precision=0.5147, score=0.6463, precisionAware=0.5735  TP:35 FP:33 FN:18 TN:42
- epoch 27: auc=0.7130, f1=0.5946, recall=0.6226, precision=0.5690, score=0.6323, precisionAware=0.6055  TP:33 FP:25 FN:20 TN:50
- epoch 38: auc=0.7499, f1=0.6078, recall=0.5849, precision=0.6327, score=0.6248, precisionAware=0.6487  TP:31 FP:18 FN:22 TN:57

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 43: auc=0.7364, f1=0.6415, recall=0.6415, precision=0.6415, precisionAware=0.6605, composite=0.6605  TP:34 FP:19 FN:19 TN:56
- epoch 24: auc=0.7389, f1=0.5455, recall=0.4528, precision=0.6857, precisionAware=0.6543, composite=0.5378  TP:24 FP:11 FN:29 TN:64
- epoch 38: auc=0.7499, f1=0.6078, recall=0.5849, precision=0.6327, precisionAware=0.6487, composite=0.6248  TP:31 FP:18 FN:22 TN:57
- epoch 4: auc=0.6038, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6319, composite=0.1413  TP:1 FP:0 FN:52 TN:75
