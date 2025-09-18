# Mask Threshold Sweep Report

## Summary (Top by composite)

| Rank | Threshold | Gate | AUC | F1 | Recall | Precision | Composite | Run Dir |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 1 | 0.6 | 0.65 | 0.5048 | 0.0189 | 0.0095 | 1.0000 | 0.1114 | /home/user/projects/train/model/long_window/CNN_BiLSTM/result_mask_sweep/run_01/01 |
| 2 | 0.6 | 0.65 | 0.5048 | 0.0189 | 0.0095 | 1.0000 | 0.1114 | /home/user/projects/train/model/long_window/CNN_BiLSTM/result_mask_sweep/run_01/02 |
| 3 | 0.75 | 0.65 | 0.5048 | 0.0189 | 0.0095 | 1.0000 | 0.1114 | /home/user/projects/train/model/long_window/CNN_BiLSTM/result_mask_sweep/run_02/02 |
| 4 | 0.8 | 0.65 | 0.5048 | 0.0189 | 0.0095 | 1.0000 | 0.1114 | /home/user/projects/train/model/long_window/CNN_BiLSTM/result_mask_sweep/run_03/01 |
| 5 | 0.85 | 0.65 | 0.5048 | 0.0189 | 0.0095 | 1.0000 | 0.1114 | /home/user/projects/train/model/long_window/CNN_BiLSTM/result_mask_sweep/run_04/01 |
| 6 | 0.75 | 0.65 | 0.5048 | 0.0000 | 0.0000 | 0.0000 | 0.1010 | /home/user/projects/train/model/long_window/CNN_BiLSTM/result_mask_sweep/run_02/01 |

## Details per run

### th=0.6 gate=0.65

- Effective batch: 8
- Best val epoch: 1 (score=0.1)
- Test: AUC=0.5048, F1=0.0189, Recall=0.0095, Precision=1.0000, Composite=0.1114
- Confusion (TP/FP/FN/TN): TP=1, FP=0, FN=104, TN=151

### th=0.6 gate=0.65

- Effective batch: 128
- Best val epoch: 1 (score=0.1)
- Test: AUC=0.5048, F1=0.0189, Recall=0.0095, Precision=1.0000, Composite=0.1114
- Confusion (TP/FP/FN/TN): TP=1, FP=0, FN=104, TN=151

### th=0.75 gate=0.65

- Effective batch: 128
- Best val epoch: 1 (score=0.1)
- Test: AUC=0.5048, F1=0.0189, Recall=0.0095, Precision=1.0000, Composite=0.1114
- Confusion (TP/FP/FN/TN): TP=1, FP=0, FN=104, TN=151

### th=0.8 gate=0.65

- Effective batch: 128
- Best val epoch: 1 (score=0.1)
- Test: AUC=0.5048, F1=0.0189, Recall=0.0095, Precision=1.0000, Composite=0.1114
- Confusion (TP/FP/FN/TN): TP=1, FP=0, FN=104, TN=151

### th=0.85 gate=0.65

- Effective batch: 128
- Best val epoch: 1 (score=0.1)
- Test: AUC=0.5048, F1=0.0189, Recall=0.0095, Precision=1.0000, Composite=0.1114
- Confusion (TP/FP/FN/TN): TP=1, FP=0, FN=104, TN=151

### th=0.75 gate=0.65

- Effective batch: 8
- Best val epoch: 1 (score=0.1)
- Test: AUC=0.5048, F1=0.0000, Recall=0.0000, Precision=0.0000, Composite=0.1010
- Confusion (TP/FP/FN/TN): TP=0, FP=0, FN=105, TN=151

