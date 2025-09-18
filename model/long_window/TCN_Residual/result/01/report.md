# TCN_Residual Report

## Test metrics
- AUC: 0.6134
- F1: 0.4569
- Recall: 0.4286
## Confusion matrix (TP/FP/FN/TN)
- TP: 45
- FP: 47
- FN: 60
- TN: 104

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 18: auc=0.5701, f1=0.4865, recall=0.5094, score=0.5207  TP: 27 FP: 31 FN: 26 TN: 44
- epoch 14: auc=0.5635, f1=0.4906, recall=0.4906, score=0.5125  TP: 26 FP: 27 FN: 27 TN: 48
- epoch 13: auc=0.5625, f1=0.4808, recall=0.4717, score=0.5017  TP: 25 FP: 26 FN: 28 TN: 49
- epoch 17: auc=0.5751, f1=0.4673, recall=0.4717, score=0.5014  TP: 25 FP: 29 FN: 28 TN: 46

## "attn_units": 64, "gate_type": "sigmoid"