# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite/seed8 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 8 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7375
- F1: 0.7194
- Recall: 0.7663
- Precision: 0.6779
- Composite Score: 0.7465 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.7022 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 141
- FP: 67
- FN: 43
- TN: 103

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7736, f1=0.7389, recall=0.9206, precision=0.6170, score=0.8367, precisionAware=0.6849  TP:58 FP:36 FN:5 TN:29
- epoch 68: auc=0.7988, f1=0.7517, recall=0.8889, precision=0.6512, score=0.8297, precisionAware=0.7108  TP:56 FP:30 FN:7 TN:35
- epoch 40: auc=0.7729, f1=0.7403, recall=0.9048, precision=0.6264, score=0.8290, precisionAware=0.6898  TP:57 FP:34 FN:6 TN:31
- epoch 57: auc=0.7949, f1=0.7417, recall=0.8889, precision=0.6364, score=0.8259, precisionAware=0.6997  TP:56 FP:32 FN:7 TN:33

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 37: auc=0.7897, f1=0.7227, recall=0.6825, precision=0.7679, precisionAware=0.7587, composite=0.7160  TP:43 FP:13 FN:20 TN:52
- epoch 19: auc=0.7863, f1=0.6847, recall=0.6032, precision=0.7917, precisionAware=0.7585, composite=0.6643  TP:38 FP:10 FN:25 TN:55
- epoch 61: auc=0.7973, f1=0.7460, recall=0.7460, precision=0.7460, precisionAware=0.7563, composite=0.7563  TP:47 FP:16 FN:16 TN:49
- epoch 66: auc=0.8198, f1=0.7704, recall=0.8254, precision=0.7222, precisionAware=0.7562, composite=0.8078  TP:52 FP:20 FN:11 TN:45
