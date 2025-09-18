# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw01_fg05 --focal_alpha 0.1 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6980
- F1: 0.4625
- Recall: 0.3524
## Confusion matrix (TP/FP/FN/TN)
- TP: 37
- FP: 18
- FN: 68
- TN: 133

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 57: auc=0.7135, f1=0.5652, recall=0.4906, score=0.5575  TP: 26 FP: 13 FN: 27 TN: 62
- epoch 65: auc=0.7326, f1=0.5412, recall=0.4340, score=0.5258  TP: 23 FP: 9 FN: 30 TN: 66
- epoch 59: auc=0.7187, f1=0.4941, recall=0.3962, score=0.4901  TP: 21 FP: 11 FN: 32 TN: 64
- epoch 61: auc=0.7127, f1=0.4941, recall=0.3962, score=0.4889  TP: 21 FP: 11 FN: 32 TN: 64
