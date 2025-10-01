#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader

# Short window (PyTorch)
ROOT = Path(__file__).resolve().parents[1]
SHORT_DIR = ROOT / 'model' / 'short_window' / 'GCN_TCN'
LONG_DIR = ROOT / 'model' / 'long_window' / 'CNN_BiLSTM'
MIX_DIR = ROOT / 'model' / 'mix_winndow'

# Dynamically import project modules without altering originals
GCN_models = SourceFileLoader('gcn_models', str(SHORT_DIR / 'models.py')).load_module()
CNN_model = SourceFileLoader('cnn_model', str(LONG_DIR / 'model.py')).load_module()
MIX_models = SourceFileLoader('mix_models', str(MIX_DIR / 'models_mixed.py')).load_module()


def load_short_model(run_dir: str, device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir/'config.json').read_text())
    params = cfg['params']; in_dim = cfg['data']['F']
    model = GCN_models.GCN_TCN_Classifier(
        in_dim=in_dim, n_classes=2,
        gcn_hidden=params['gcn_hidden'],
        tcn_channels=tuple(int(x) for x in params['tcn_channels'].split(',')),
        tcn_kernel=params['tcn_kernel'], tcn_dropout=params['tcn_dropout'],
        tcn_dil_growth=params['tcn_dil_growth'],
        fc_hidden=params['fc_hidden'], fc_dropout=params['fc_dropout']
    ).to(device)
    state = torch.load(run_dir/'best.ckpt', map_location=device)
    model.load_state_dict(state['model']); model.eval()
    return model, device, in_dim


def load_long_model(weights_path: str, input_shape=(36,75)):
    import tensorflow as tf
    from pathlib import Path as _Path

    def _is_h5_or_keras(p: str) -> bool:
        suf = _Path(p).suffix.lower()
        return suf in {'.h5', '.keras'}

    def _is_tf_ckpt_prefix(p: str) -> bool:
        return _Path(p + '.index').exists()

    # Try common attention unit configs to match checkpoints
    attn_try = [64, 32]
    last_err = None
    for attn_units in attn_try:
        try:
            model = CNN_model.build_cnn_bilstm(input_shape=input_shape, use_mask=False, attn_units=attn_units)
            # Build weights by a dummy forward
            dummy = np.zeros((1,)+input_shape, dtype=np.float32)
            _ = model.predict(dummy, verbose=0)

            if _is_h5_or_keras(weights_path):
                # Simple case: .h5/.keras
                model.load_weights(weights_path)
            elif _is_tf_ckpt_prefix(weights_path):
                # Use Checkpoint with expect_partial to avoid optimizer related warnings
                ckpt = tf.train.Checkpoint(model=model)
                status = ckpt.restore(weights_path)
                try:
                    status.expect_partial()
                except Exception:
                    pass
            else:
                # Fallback to load_weights (will likely raise)
                model.load_weights(weights_path)
            return model
        except Exception as e:
            last_err = e
            continue
    # If all attempts failed, raise last error
    raise last_err


def load_fusion_mlp(ckpt_dir: str, device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = MIX_models.MLPFusion()
    state = torch.load(str(Path(ckpt_dir)/'fusion.ckpt'), map_location=device)
    model.load_state_dict(state['model']); model.to(device).eval()
    return model, device


def infer_short_window(model, device, win_x: np.ndarray) -> float:
    # win_x: (T=30, F)
    x = torch.from_numpy(win_x).unsqueeze(0).to(device)
    mask = torch.ones((1, win_x.shape[0]), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits,_ = model(x, mask=mask)
        p = torch.softmax(logits, dim=-1)[0,1].item()
    return float(p)


def infer_long_window(model, win_x: np.ndarray) -> float:
    # Keras expects (F,T)
    x = np.expand_dims(win_x.T.astype('float32'), axis=0)
    p = float(model.predict(x, verbose=0)[0,0])
    return p


def fuse_mlp(model, device, p_short: float, p_long: float) -> float:
    import torch
    x = torch.tensor([[p_short, p_long]], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x)
        p = torch.softmax(logits, dim=-1)[:,1].item()
    return float(p)
