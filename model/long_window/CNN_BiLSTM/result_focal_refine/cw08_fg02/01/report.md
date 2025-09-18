# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw08_fg02 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7479
- F1: 0.5969
- Recall: 0.5429
## Confusion matrix (TP/FP/FN/TN)
- TP: 57
- FP: 29
- FN: 48
- TN: 122

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 54: auc=0.7273, f1=0.6346, recall=0.6226, score=0.6472  TP: 33 FP: 18 FN: 20 TN: 57
- epoch 61: auc=0.7582, f1=0.6400, recall=0.6038, score=0.6455  TP: 32 FP: 15 FN: 21 TN: 60
- epoch 64: auc=0.8161, f1=0.6327, recall=0.5849, score=0.6455  TP: 31 FP: 14 FN: 22 TN: 61
- epoch 62: auc=0.7708, f1=0.6214, recall=0.6038, score=0.6425  TP: 32 FP: 18 FN: 21 TN: 57
