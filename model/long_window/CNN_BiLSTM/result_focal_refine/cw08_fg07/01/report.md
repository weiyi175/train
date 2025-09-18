# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw08_fg07 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7044
- F1: 0.5446
- Recall: 0.5238
## Confusion matrix (TP/FP/FN/TN)
- TP: 55
- FP: 42
- FN: 50
- TN: 109

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 27: auc=0.7313, f1=0.6441, recall=0.7170, score=0.6980  TP: 38 FP: 27 FN: 15 TN: 48
- epoch 33: auc=0.7089, f1=0.6441, recall=0.7170, score=0.6935  TP: 38 FP: 27 FN: 15 TN: 48
- epoch 31: auc=0.7235, f1=0.6281, recall=0.7170, score=0.6916  TP: 38 FP: 30 FN: 15 TN: 45
- epoch 35: auc=0.7192, f1=0.6066, recall=0.6981, score=0.6749  TP: 37 FP: 32 FN: 16 TN: 43
