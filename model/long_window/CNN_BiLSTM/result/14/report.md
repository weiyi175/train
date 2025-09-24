# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 3 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 42 --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7368
- F1: 0.0928
- Recall: 0.0489
- Precision: 0.9000
- Composite Score: 0.1997 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6252 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 9
- FP: 1
- FN: 175
- TN: 169

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 3: auc=0.6886, f1=0.1690, recall=0.0952, precision=0.7500, score=0.2361, precisionAware=0.5634  TP:6 FP:2 FN:57 TN:63
- epoch 2: auc=0.6606, f1=0.0000, recall=0.0000, precision=0.0000, score=0.1321, precisionAware=0.1321  TP:0 FP:0 FN:63 TN:65
- epoch 1: auc=0.6176, f1=0.0000, recall=0.0000, precision=0.0000, score=0.1235, precisionAware=0.1235  TP:0 FP:0 FN:63 TN:65

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 3: auc=0.6886, f1=0.1690, recall=0.0952, precision=0.7500, precisionAware=0.5634, composite=0.2361  TP:6 FP:2 FN:57 TN:63
- epoch 2: auc=0.6606, f1=0.0000, recall=0.0000, precision=0.0000, precisionAware=0.1321, composite=0.1321  TP:0 FP:0 FN:63 TN:65
- epoch 1: auc=0.6176, f1=0.0000, recall=0.0000, precision=0.0000, precisionAware=0.1235, composite=0.1235  TP:0 FP:0 FN:63 TN:65
