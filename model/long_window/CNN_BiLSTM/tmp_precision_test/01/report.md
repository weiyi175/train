# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 1 --batch 8 --accumulate_steps 8 --result_dir model/long_window/CNN_BiLSTM/tmp_precision_test --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed None --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6338
- F1: 0.0000
- Recall: 0.0000
- Precision: 0.0000
- Composite Score: 0.1268 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.1268 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.5985, f1=0.0000, recall=0.0000, precision=0.0000, score=0.1197, precisionAware=0.1197  TP:0 FP:0 FN:53 TN:75

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.5985, f1=0.0000, recall=0.0000, precision=0.0000, precisionAware=0.1197, composite=0.1197  TP:0 FP:0 FN:53 TN:75
