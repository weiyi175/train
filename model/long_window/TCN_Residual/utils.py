import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix


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


def get_confusion_elements(y_true, y_pred):
    # returns TP, FP, FN, TN for binary
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return int(tp), int(fp), int(fn), int(tn)


def load_npz_windows(path):
    # allow_pickle=True to support object arrays saved into NPZ (some dataset exporters use object dtype)
    data = np.load(path, allow_pickle=True)
    # helper to convert object arrays to numeric ndarrays when possible
    def _to_numeric_array(a):
        if isinstance(a, np.ndarray) and a.dtype == object:
            # try common conversions
            try:
                return np.asarray(a.tolist(), dtype=np.float32)
            except Exception:
                # fallback: attempt stacking elements
                try:
                    return np.stack([np.asarray(x, dtype=np.float32) for x in a])
                except Exception:
                    return np.asarray(a)
        else:
            return np.asarray(a)

    # default behaviour: prefer long windows if present, else short, else generic X/y
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
        # fallback to first two arrays
        keys = list(data.files)
        if len(keys) >= 2:
            X = _to_numeric_array(data[keys[0]])
            y = _to_numeric_array(data[keys[1]])
        else:
            raise ValueError(f"NPZ file {path} does not contain at least two arrays (X and y)")
    return X, y
