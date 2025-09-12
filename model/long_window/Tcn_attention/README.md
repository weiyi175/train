# Long Window TCN + Attention

針對長窗 (T=75) 的 TCN + Attention 分類模型訓練腳本。來源改編自 `model/short_window/Tcn_attention`，主要差異:
- Dataset `split='long'`
- 預設使用較寬的感受野 (較大 dilation depth) 與較小 kernel 以控制 padding 開銷
- 保留 `--use_norm` 旗標選擇 normalized features
- 自動記錄輸出到本目錄 `result/`

快速啟動 (預設參數已附在腳本 RUN_ARGS):
```
python train_tcn_attn_long.py
```
可調參數請查看腳本內 argparse 定義。
