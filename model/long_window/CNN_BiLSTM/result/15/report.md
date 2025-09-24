# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 50 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6903
- F1: 0.5994
- Recall: 0.5326
- Precision: 0.6853
- Composite Score: 0.5842 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6605 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 98
- FP: 45
- FN: 86
- TN: 125

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 41: auc=0.6654, f1=0.5763, recall=0.6415, precision=0.5231, score=0.6267, precisionAware=0.5675  TP:34 FP:31 FN:19 TN:44
- epoch 21: auc=0.6606, f1=0.5769, recall=0.5660, precision=0.5882, score=0.5882, precisionAware=0.5993  TP:30 FP:21 FN:23 TN:54
- epoch 35: auc=0.6518, f1=0.5357, recall=0.5660, precision=0.5085, score=0.5741, precisionAware=0.5453  TP:30 FP:29 FN:23 TN:46
- epoch 22: auc=0.6541, f1=0.5385, recall=0.5283, precision=0.5490, score=0.5565, precisionAware=0.5669  TP:28 FP:23 FN:25 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 21: auc=0.6606, f1=0.5769, recall=0.5660, precision=0.5882, precisionAware=0.5993, composite=0.5882  TP:30 FP:21 FN:23 TN:54
- epoch 29: auc=0.6707, f1=0.5306, recall=0.4906, precision=0.5778, precisionAware=0.5822, composite=0.5386  TP:26 FP:19 FN:27 TN:56
- epoch 41: auc=0.6654, f1=0.5763, recall=0.6415, precision=0.5231, precisionAware=0.5675, composite=0.6267  TP:34 FP:31 FN:19 TN:44
- epoch 22: auc=0.6541, f1=0.5385, recall=0.5283, precision=0.5490, precisionAware=0.5669, composite=0.5565  TP:28 FP:23 FN:25 TN:52
