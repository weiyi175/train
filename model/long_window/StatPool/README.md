StatPool + Summary training scaffold

目標：對 long-window 特徵做 Mask-aware Statistic Pooling (StatPool)，再以一個小型 MLP 做分類。

檔案：
- dataset_statpool.py: Dataset 與 StatPool / summary feature 計算
- utils_scaler.py: 小型 StandardScaler（fit/transform, json save/load）
- model_statpool.py: MLP 分類器
- train_statpool.py: 訓練腳本（產生 result/01,02...）

資料預期位置（可用 --feature_dir 指定）:
- 預設 feature_dir: train_data/Eigenvalue
- 單檔格式支援 .npz（keys: 'features' = (T,F), optional 'mask'=(T,) or (T,1), 'label'=int）

快速使用範例：
python train_statpool.py --feature_dir /home/user/projects/train/train_data/Eigenvalue --out /home/user/projects/train/model/long_window/StatPool/result --epochs 10 --batch 32

注意：
- 本 scaffold 側重資料管線與 StatPool 計算。具體 domain summary（例如嘴部接近手之頻率）需使用者根據原始骨架自行擴展到 `custom_summary` 函式。
