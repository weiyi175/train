import os
import json
import numpy as np

def next_result_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    nums = []
    for d in existing:
        try:
            nums.append(int(d))
        except Exception:
            pass
    next_num = 1 if not nums else max(nums) + 1
    folder = os.path.join(base_dir, f"{next_num:02d}")
    os.makedirs(folder, exist_ok=True)
    return folder


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_npz_windows(path):
    """Load window NPZ and return (X, y) consistently as (N, F, T).

    - Prefers long_* over short_* when both exist (training/inference default).
    - Handles object npz (N,F,T) and dense npz (N,T,F) by using feature_list length to auto-transpose.
    """
    data = np.load(path, allow_pickle=True)

    def _to_numeric_array(a):
        if isinstance(a, np.ndarray) and a.dtype == object:
            try:
                return np.asarray(a.tolist(), dtype=np.float32)
            except Exception:
                try:
                    return np.stack([np.asarray(x, dtype=np.float32) for x in a])
                except Exception:
                    return np.asarray(a)
        else:
            return np.asarray(a)

    # try to get feature_list length to infer axis order
    feat_list = []
    if 'feature_list' in data:
        try:
            feat_list = list(data['feature_list'])
        except Exception:
            feat_list = []
    feat_count = len(feat_list) if isinstance(feat_list, list) else 0

    # choose arrays
    if 'long_norm' in data and 'long_label' in data:
        X = _to_numeric_array(data['long_norm'])
        y = _to_numeric_array(data['long_label'])
    elif 'short_norm' in data and 'short_label' in data:
        X = _to_numeric_array(data['short_norm'])
        y = _to_numeric_array(data['short_label'])
    elif 'X' in data and 'y' in data:
        X = _to_numeric_array(data['X'])
        y = _to_numeric_array(data['y'])
    else:
        keys = list(data.files)
        if len(keys) >= 2:
            X = _to_numeric_array(data[keys[0]])
            y = _to_numeric_array(data[keys[1]])
        else:
            raise ValueError(f"NPZ file {path} does not contain at least two arrays (X and y)")

    # Ensure (N,F,T): if dense (N,T,F) and feature_list length matches last dim, transpose
    if X.ndim == 3 and feat_count > 0:
        N, A, B = X.shape
        # object npz: (N,F,T) -> A should be feat_count
        # dense npz: (N,T,F)  -> B should be feat_count
        if A != feat_count and B == feat_count:
            # transpose to (N,F,T)
            X = np.transpose(X, (0, 2, 1))
        # otherwise leave as is

    return X, y
