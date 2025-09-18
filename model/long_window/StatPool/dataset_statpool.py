from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

# Simple loader for .npz feature files with keys: 'features' (T,F), optional 'mask' (T,), 'label'
class StatPoolFeatureDataset(Dataset):
    def __init__(self, feature_dir: str, split: str = 'train', transform=None, scaler=None, min_T: int = 1):
        self.feature_dir = Path(feature_dir)
        self.files = sorted(self.feature_dir.glob('*.npz'))
        self.transform = transform
        self.scaler = scaler
        self.min_T = min_T
        self.smooth_window = 0
        self.zscore_mode = 'global'
        self.use_raw_skeleton = False
    def __init__(self, feature_dir: str, split: str = 'train', transform=None, scaler=None, min_T: int = 1,
                 smooth_window: int = 0, zscore_mode: str = 'global', use_raw_skeleton: bool = False):
        self.feature_dir = Path(feature_dir)
        # support: directory of .npz files or a single .npz archive containing many samples
        if self.feature_dir.is_file() and self.feature_dir.suffix == '.npz':
            # try to interpret single npz: if it contains per-sample arrays like long_norm/long_label
            try:
                with np.load(self.feature_dir, allow_pickle=True) as d:
                    # if it contains an array called 'files' treat as file list
                    if 'files' in d:
                        files = [Path(x) for x in d['files'].tolist()]
                    else:
                        # detect prefixes like 'long' or 'short' that have *_norm or *_raw
                        prefixes = set()
                        for k in d.keys():
                            if k.endswith('_norm') or k.endswith('_raw'):
                                prefixes.add(k.rsplit('_', 1)[0])
                        # prefer 'long' if present, else pick any
                        pref = None
                        if 'long' in prefixes:
                            pref = 'long'
                        elif prefixes:
                            pref = sorted(prefixes)[0]
                        if pref is not None:
                            arr_name = f"{pref}_norm" if f"{pref}_norm" in d else (f"{pref}_raw" if f"{pref}_raw" in d else None)
                            if arr_name is not None:
                                N = d[arr_name].shape[0]
                                files = [Path(f"{self.feature_dir}::{pref}::{i}") for i in range(N)]
                            else:
                                files = [self.feature_dir]
                        else:
                            # fallback: create virtual list referencing the same npz for each key named sample_*
                            sample_keys = [k for k in d.keys() if k.startswith('sample')]
                            if sample_keys:
                                files = [Path(f"{self.feature_dir}::{k}") for k in sample_keys]
                            else:
                                files = [self.feature_dir]
            except Exception:
                files = [self.feature_dir]
            self.files = files
        else:
            self.files = sorted(self.feature_dir.glob('*.npz'))
        self.transform = transform
        self.scaler = scaler
        self.min_T = min_T
        self.smooth_window = int(smooth_window)
        # zscore_mode: 'none'|'per_file'|'global'
        self.zscore_mode = zscore_mode
        self.use_raw_skeleton = bool(use_raw_skeleton)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        raw = None
        label = 0
        # support pseudo-path format path::key for single-npz multiple-sample
        pstr = str(p)
        if '::' in pstr:
            parts = pstr.split('::')
            if len(parts) == 2:
                npz_path, key = parts
                with np.load(npz_path, allow_pickle=True) as d:
                    entry = d[key]
                    if isinstance(entry, np.ndarray) and entry.ndim >= 2:
                        feats = entry
                    elif isinstance(entry, np.ndarray) and entry.ndim == 1:
                        feats = entry.reshape(-1, 1)
                    else:
                        feats = d.get('features')
                    mask = None
            elif len(parts) == 3:
                # format: /path/to/file.npz::prefix::index
                npz_path, prefix, idx = parts
                with np.load(npz_path, allow_pickle=True) as d:
                    # prefer prefix_norm, then prefix_raw, then prefix_features
                    if f"{prefix}_norm" in d:
                        feats = d[f"{prefix}_norm"][int(idx)]
                    elif f"{prefix}_raw" in d:
                        feats = d[f"{prefix}_raw"][int(idx)]
                    elif 'features' in d:
                        feats = d['features']
                    else:
                        raise KeyError(f'No features found for prefix {prefix} in {npz_path}')
                    mask = None
                    if f"{prefix}_label" in d:
                        label = int(d[f"{prefix}_label"][int(idx)])
                    else:
                        label = int(d[f"{prefix}_label"]) if f"{prefix}_label" in d and np.isscalar(d[f"{prefix}_label"]) else 0
                    # raw skeleton if available per-sample
                    if f"{prefix}_skeleton" in d:
                        raw = d[f"{prefix}_skeleton"][int(idx)]
                    elif f"{prefix}_joints" in d:
                        raw = d[f"{prefix}_joints"][int(idx)]
    # normal file path already handled above
        else:
            # normal file path
            with np.load(p, allow_pickle=True) as d:
                feats = d['features']  # (T, F)
                mask = d.get('mask')
                label = int(d['label']) if 'label' in d else 0
                if self.use_raw_skeleton:
                    if 'skeleton' in d:
                        raw = d['skeleton']
                    elif 'joints' in d:
                        raw = d['joints']
        feats = np.asarray(feats, dtype=np.float32)
        T, F = feats.shape
        # optional smoothing (moving average along time)
        if self.smooth_window and self.smooth_window > 1:
            k = self.smooth_window
            # pad reflect
            pad = k//2
            feats = np.pad(feats, ((pad,pad),(0,0)), mode='edge')
            kernel = np.ones((k,))/k
            feats = np.stack([np.convolve(feats[:,j], kernel, mode='valid') for j in range(F)], axis=1)
        if self.scaler is not None:
            if self.zscore_mode == 'global':
                feats = self.scaler.transform(feats)
            elif self.zscore_mode == 'per_file':
                # per-file zscore
                mu = feats.mean(axis=0, keepdims=True)
                sd = feats.std(axis=0, keepdims=True)
                sd[sd < 1e-6] = 1.0
                feats = (feats - mu) / sd
            else:
                # none: do nothing
                pass
        # simple masking: if mask provided, ensure shape
        if mask is None:
            mask = np.ones((T,), dtype=np.bool_)
        else:
            mask = np.asarray(mask).astype(bool).reshape(T,)
        # statpool: calculate masked mean and std
        valid = feats[mask]
        if len(valid) < 1:
            mv = np.zeros((F,), dtype=np.float32)
            sd = np.zeros((F,), dtype=np.float32)
        else:
            mv = valid.mean(axis=0)
            sd = valid.std(axis=0)
        # summary features: mean, std, median, max, min
        med = np.median(valid, axis=0) if len(valid)>0 else np.zeros_like(mv)
        mx = valid.max(axis=0) if len(valid)>0 else np.zeros_like(mv)
        mn = valid.min(axis=0) if len(valid)>0 else np.zeros_like(mv)
        pooled = np.concatenate([mv, sd, med, mx, mn], axis=0).astype(np.float32)
        # domain summary features from raw skeleton if available
        if raw is not None:
            try:
                raw_a = np.asarray(raw)
                # expect (T, J, 2) or (T, J, 3)
                if raw_a.ndim == 3 and raw_a.shape[0] == feats.shape[0]:
                    # simple heuristic: mouth index 0, hands last two joints
                    J = raw_a.shape[1]
                    mouth = raw_a[:, 0, :2]
                    left_hand = raw_a[:, -2, :2]
                    right_hand = raw_a[:, -1, :2]
                    d1 = np.linalg.norm(left_hand - mouth, axis=1)
                    d2 = np.linalg.norm(right_hand - mouth, axis=1)
                    dmin = np.minimum(d1, d2)
                    # fraction frames where hand near mouth (relative threshold)
                    thr = np.median(dmin) * 0.6 if np.median(dmin) > 0 else 10.0
                    near_frac = float((dmin < thr).mean())
                    # mouth confidence ratio if provided
                    mouth_conf = None
                    if raw_a.shape[2] >= 3:
                        # if third dim contains confidence for mouth joint
                        mouth_conf = float(raw_a[:,0,2].mean())
                    extra = np.array([near_frac, mouth_conf if mouth_conf is not None else 0.0], dtype=np.float32)
                    pooled = np.concatenate([pooled, extra], axis=0)
            except Exception:
                pass
        sample = {'pooled': torch.from_numpy(pooled), 'label': torch.tensor(label, dtype=torch.long), 'path': str(p)}
        if self.transform:
            sample = self.transform(sample)
        return sample

def collate_batch(batch):
    ps = torch.stack([b['pooled'] for b in batch], 0)
    ys = torch.stack([b['label'] for b in batch], 0)
    paths = [b['path'] for b in batch]
    return {'pooled': ps, 'label': ys, 'path': paths}
