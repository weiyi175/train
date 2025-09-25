# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows /home/user/projects/train/Val_data/slipce_thresh040/windows_npz.npz --test_windows /home/user/projects/train/test_data/slipce_thresh040/windows_npz.npz --kfold 0 --kfold_seed 42 --final_internal_val_ratio 0.0 --epochs 1 --batch 32 --accumulate_steps 1 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/tmp_maskfix_run --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed 43 --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7588
- F1: 0.3768
- Recall: 0.2407
- Precision: 0.8667
- Composite Score: 0.3852 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6981 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 52
- FP: 8
- FN: 164
- TN: 130

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.6656, f1=0.3182, recall=0.1909, precision=0.9545, score=0.3240, precisionAware=0.7058  TP:21 FP:1 FN:89 TN:41

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.6656, f1=0.3182, recall=0.1909, precision=0.9545, precisionAware=0.7058, composite=0.3240  TP:21 FP:1 FN:89 TN:41
