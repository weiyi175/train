# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw08_fg10 --focal_alpha 0.35 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7085
- F1: 0.5755
- Recall: 0.5810
## Confusion matrix (TP/FP/FN/TN)
- TP: 61
- FP: 46
- FN: 44
- TN: 105

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 33: auc=0.7250, f1=0.6555, recall=0.7358, score=0.7096  TP: 39 FP: 27 FN: 14 TN: 48
- epoch 44: auc=0.6984, f1=0.6341, recall=0.7358, score=0.6978  TP: 39 FP: 31 FN: 14 TN: 44
- epoch 42: auc=0.7157, f1=0.6372, recall=0.6792, score=0.6739  TP: 36 FP: 24 FN: 17 TN: 51
- epoch 38: auc=0.7102, f1=0.6154, recall=0.6792, score=0.6663  TP: 36 FP: 28 FN: 17 TN: 47
