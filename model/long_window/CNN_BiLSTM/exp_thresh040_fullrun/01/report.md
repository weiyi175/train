# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows /home/user/projects/train/Val_data/slipce_thresh040/windows_npz.npz --test_windows /home/user/projects/train/test_data/slipce_thresh040/windows_npz.npz --kfold 0 --kfold_seed 42 --final_internal_val_ratio 0.0 --epochs 50 --batch 32 --accumulate_steps 1 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/exp_thresh040_fullrun --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed 43 --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.5923
- F1: 0.6025
- Recall: 0.5509
- Precision: 0.6648
- Composite Score: 0.5747 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6316 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 119
- FP: 60
- FN: 97
- TN: 78

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 13: auc=0.6920, f1=0.8426, recall=0.9000, precision=0.7920, score=0.8412, precisionAware=0.7872  TP:99 FP:26 FN:11 TN:16
- epoch 37: auc=0.7654, f1=0.8387, recall=0.8273, precision=0.8505, score=0.8183, precisionAware=0.8299  TP:91 FP:16 FN:19 TN:26
- epoch 45: auc=0.7236, f1=0.8198, recall=0.8273, precision=0.8125, score=0.8043, precisionAware=0.7969  TP:91 FP:21 FN:19 TN:21
- epoch 34: auc=0.7584, f1=0.8165, recall=0.8091, precision=0.8241, score=0.8012, precisionAware=0.8087  TP:89 FP:19 FN:21 TN:23

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 37: auc=0.7654, f1=0.8387, recall=0.8273, precision=0.8505, precisionAware=0.8299, composite=0.8183  TP:91 FP:16 FN:19 TN:26
- epoch 40: auc=0.7628, f1=0.8113, recall=0.7818, precision=0.8431, precisionAware=0.8175, composite=0.7869  TP:86 FP:16 FN:24 TN:26
- epoch 22: auc=0.7563, f1=0.7435, recall=0.6455, precision=0.8765, precisionAware=0.8126, composite=0.6970  TP:71 FP:10 FN:39 TN:32
- epoch 42: auc=0.7792, f1=0.7487, recall=0.6636, precision=0.8588, precisionAware=0.8099, composite=0.7123  TP:73 FP:12 FN:37 TN:30
