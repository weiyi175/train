# Baseline_test

快速初測兩個 baseline：
- MLP over Flattened Features（短/長分開；輸入為 raw+norm 串接後展平）
- StatPool (Mean+Std) + MLP（時間池化表示）

資料來源：`/home/user/projects/train/train_data/slipce/windows_dense_npz.npz`

輸出位置：`/home/user/projects/train/model/Baseline_test/result/01`, `02`, `03`…（每次執行自動遞增）

## 使用方式

 dry-run（載入資料、建立模型、吐出一批資料尺寸後結束）
```bash
python /home/user/projects/train/model/Baseline_test/train_baseline.py --model mlp_flat --split short --dry_run
```

 MLP over Flatten（短視窗）
```bash
python /home/user/projects/train/model/Baseline_test/train_baseline.py \
  --model mlp_flat --split short --epochs 5 --batch_size 128 --lr 1e-3
```

 StatPool + MLP（長視窗，使用 normalized 特徵）
```bash
python /home/user/projects/train/model/Baseline_test/train_baseline.py \
  --model statpool_mlp --split long --use_norm --epochs 5 --batch_size 128 --lr 1e-3
```

重要參數：
- `--split {short,long}`：選擇短/長視窗。
- `--model {mlp_flat,statpool_mlp}`：模型類型。
- `--use_norm`：StatPool 預設用 normalized 特徵；也可移除改用 raw。
- `--temporal_jitter_frames`：時間軸 jitter（避免重複樣本過度記憶）。
- `--balance_by_class`、`--amplify_hard_negative`：訓練取樣時的平衡與強化難負樣本（需 dense NPZ 的 short_weight）。

輸出內容（於自動建立的 run 目錄內）：
- `config.json`：本次參數
- `train_log.jsonl`：每 epoch 的 loss/acc
- `best.ckpt`：最佳驗證準確率的權重
- `last.ckpt`：最後一個 epoch 的權重
- `model_spec.txt`：輸入維度與模型結構摘要