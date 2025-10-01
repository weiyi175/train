# TCN Attention (Long) Extended Report

## Top 4 epochs by Composite
- epoch 4: auc=0.5771, f1=0.8293, recall=0.9273, precision=0.7500, composite=0.8278, precisionAware=0.7392 TP:102 FP:34 FN:8 TN:8
- epoch 18: auc=0.5303, f1=0.7950, recall=0.8636, precision=0.7364, composite=0.7764, precisionAware=0.7128 TP:95 FP:34 FN:15 TN:8
- epoch 13: auc=0.5695, f1=0.7826, recall=0.8182, precision=0.7500, composite=0.7578, precisionAware=0.7237 TP:90 FP:30 FN:20 TN:12
- epoch 6: auc=0.5554, f1=0.7911, recall=0.8091, precision=0.7739, composite=0.7530, precisionAware=0.7354 TP:89 FP:26 FN:21 TN:16

## Top 4 epochs by Precision-aware
- epoch 8: auc=0.6102, f1=0.7727, recall=0.7727, precision=0.7727, precisionAware=0.7402, composite=0.7402 TP:85 FP:25 FN:25 TN:17
- epoch 4: auc=0.5771, f1=0.8293, recall=0.9273, precision=0.7500, precisionAware=0.7392, composite=0.8278 TP:102 FP:34 FN:8 TN:8
- epoch 6: auc=0.5554, f1=0.7911, recall=0.8091, precision=0.7739, precisionAware=0.7354, composite=0.7530 TP:89 FP:26 FN:21 TN:16
- epoch 5: auc=0.6134, f1=0.6907, recall=0.6091, precision=0.7976, precisionAware=0.7287, composite=0.6344 TP:67 FP:17 FN:43 TN:25

## Independent Test metrics
- AUC: 0.6464
- F1: 0.5241
- Recall: 0.4028
- Precision: 0.7500
- Composite Score: 0.4879
- Precision-aware Score: 0.6615
- Confusion (TP/FP/FN/TN): 87/29/129/109