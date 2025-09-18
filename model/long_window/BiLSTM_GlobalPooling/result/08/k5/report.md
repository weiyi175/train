# K-Fold Aggregated Report
- folds: 5
- oof_auc: 0.6145126989605327
- ensemble_avg_prob_auc: 0.8573511348871204
- ensemble_majority_vote_auc: 0.6089055140490499

## Top epochs across folds (by combined score 0.5*AUC+0.3*F1+0.2*recall)
1. fold 2 epoch 15: comb_score=0.5976, val_auc=0.616913, val_f1=0.5600
   - confusion (TP,FP,FN,TN): 63,58,41,93
2. fold 2 epoch 13: comb_score=0.5861, val_auc=0.620606, val_f1=0.5413
   - confusion (TP,FP,FN,TN): 59,55,45,96
3. fold 2 epoch 12: comb_score=0.5834, val_auc=0.613665, val_f1=0.5438
   - confusion (TP,FP,FN,TN): 59,54,45,97
4. fold 2 epoch 14: comb_score=0.5758, val_auc=0.616149, val_f1=0.5333
   - confusion (TP,FP,FN,TN): 56,50,48,101