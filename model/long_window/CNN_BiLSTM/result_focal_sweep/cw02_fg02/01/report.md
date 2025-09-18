# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw02_fg02 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7130
- F1: 0.4884
- Recall: 0.4000
## Confusion matrix (TP/FP/FN/TN)
- TP: 42
- FP: 25
- FN: 63
- TN: 126

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 36: auc=0.7321, f1=0.5794, recall=0.5849, score=0.6127  TP: 31 FP: 23 FN: 22 TN: 52
- epoch 41: auc=0.7286, f1=0.5636, recall=0.5849, score=0.6073  TP: 31 FP: 26 FN: 22 TN: 49
- epoch 34: auc=0.7203, f1=0.5686, recall=0.5472, score=0.5882  TP: 29 FP: 20 FN: 24 TN: 55
- epoch 44: auc=0.6986, f1=0.5505, recall=0.5660, score=0.5879  TP: 30 FP: 26 FN: 23 TN: 49
