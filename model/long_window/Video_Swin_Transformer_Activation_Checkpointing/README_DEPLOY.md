# Video Swin 3D (Long Window) 部署說明

## 目標
提供從特徵序列 (T,36) 轉為 pseudo video，使用訓練完成之模型在多種格式 (PyTorch state dict / TorchScript / ONNX) 進行推論的統一流程。

## 產出物
- 訓練輸出資料夾: `result_mod/<run_id>`
  - `config.json`：訓練設定
  - `best.ckpt`：最佳 (F1) 權重 (PyTorch state_dict)
  - `last.ckpt`：最後一個 epoch 權重
  - `model_ts.pt`：TorchScript (trace fallback) 模型 (可選)
  - `model.onnx`：ONNX 匯出 (可選，支援動態 batch / time)
  - `train_log.jsonl`：每 epoch 指標 (含 F1 / AUC / 混淆矩陣)

## 前處理
原始輸入：每筆樣本一個 tensor/ndarray，形狀 (T,36)。
轉換步驟 (`utils/preprocess.py::to_pseudo_video`):
1. 36 -> 6x6 reshape
2. 增加 channel 維度 (1,6,6)
3. 複製為 3 通道 -> (3,6,6)
4. 堆疊時間步 -> (T,3,6,6)
5. 多樣本堆成 batch -> (B,T,3,6,6)

## 推論方式
### 1. PyTorch state dict (原生)
```
python scripts/infer.py --run_dir result_mod/01_ckpt_test --input sample.npy
```

### 2. TorchScript
```
python scripts/infer.py --run_dir result_mod/01_ckpt_test --input sample.npy --torchscript
```

### 3. ONNX (CPU)
```
pip install onnxruntime
python scripts/infer.py --run_dir result_mod/01_ckpt_test --input sample.npy --onnx
```

`--input` 可為：
- 單一 `.npy` (T,36)
- 批次 `.npy` (B,T,36)
- `.npz`：內含多個上述陣列 (自動展開)

## 匯出模型 (若未事先匯出)
```
python scripts/export_model.py --run_dir result_mod/01_ckpt_test --torchscript --onnx --opset 17 --dynamic
```

## 動態時間長度支援
ONNX 匯出已設定 `dynamic_axes`：
- batch 維度 0: dynamic
- time 維度 1: dynamic
若輸入較長序列，前處理需確保 (T,36) -> (T,3,6,6) 一致即可。

## F1 早停策略
`best.ckpt` 以驗證 F1 為準。若改回 accuracy，需同步調整 `scripts/train.py` 對應欄位與儲存 key。

## 混淆矩陣 & 指標
在 `train_log.jsonl` 每一行紀錄：
```
{"epoch":1, "train_loss":..., "val_acc":..., "val_f1":..., "val_auc":..., "val_tn":..., "val_fp":..., "val_fn":..., "val_tp":...}
```

## 依賴建議 (最小)
```
torch >= 2.0
numpy
pyyaml
onnx (匯出/驗證用)
onnxruntime (ONNX 推論; GPU 可用 onnxruntime-gpu)
```

## 常見問題
| 問題 | 原因 | 解法 |
|------|------|------|
| TorchScript script 失敗 | checkpoint + 動態控制流 | 已自動 fallback trace；若需完全 script 移除 checkpoint 重匯出 |
| ONNX TracerWarning | 分支判斷常量化 | 屬正常；不影響推論結果 |
| ONNXRuntime shape 錯誤 | 輸入少維度 | 確認輸入為 (B,T,3,6,6) 且 dtype=float32 |

## 後續可強化
- 加 mask 於全域平均排除 padding
- 多分類 (macro/micro F1) 支援
- torch.compile 加速 (訓練/推論)
- CI 自動測試 window_partition / patch merging 正確性

## 範例
```
python scripts/infer.py --run_dir result_mod/01_ckpt_test --input example.npy --torchscript
Sample 0: pred=1 prob=0.7345 topk=[(1, 0.7345), (0, 0.2655)]
```

---
如需 GPU ONNX：
```
pip install onnxruntime-gpu
```
啟動時自動使用 CUDA Provider（若可用），可自行修改 infer.py。
