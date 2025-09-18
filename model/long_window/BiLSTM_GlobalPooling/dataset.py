from pathlib import Path
import numpy as np
import torch

class LongWindowNPZDataset(torch.utils.data.Dataset):
    """Reads the windows_npz.npz with keys like long_norm, long_label, etc.
    Produces dict items: {'features': Tensor[T,F], 'label': int, 'path': str, 'mask': Tensor[T]}"""
    def __init__(self, npz_path: str, prefix: str = 'long', transform=None):
        self.npz_path = Path(npz_path)
        self.prefix = prefix
        self.transform = transform
        with np.load(self.npz_path, allow_pickle=True) as d:
            # prefer prefix_norm, then prefix_raw
            if f"{prefix}_norm" in d:
                self.features = d[f"{prefix}_norm"]
            elif f"{prefix}_raw" in d:
                self.features = d[f"{prefix}_raw"]
            else:
                raise RuntimeError(f'No {prefix}_norm or {prefix}_raw in {npz_path}')
            if f"{prefix}_label" in d:
                self.labels = d[f"{prefix}_label"]
            else:
                self.labels = np.zeros((self.features.shape[0],), dtype=np.int64)

    def __len__(self):
        return int(self.features.shape[0])

    def __getitem__(self, idx):
        feats = np.asarray(self.features[int(idx)], dtype=np.float32)
        T = feats.shape[0]
        mask = np.ones((T,), dtype=np.bool_)
        label = int(self.labels[int(idx)])
        sample = {'features': torch.from_numpy(feats), 'label': torch.tensor(label, dtype=torch.long), 'path': f"{self.npz_path}::{self.prefix}::{idx}", 'mask': torch.from_numpy(mask)}
        if self.transform:
            sample = self.transform(sample)
        return sample

def collate_batch(batch):
    # pad sequences to max T in batch
    feats = [b['features'] for b in batch]
    lengths = [f.shape[0] for f in feats]
    maxT = max(lengths)
    F = feats[0].shape[1]
    padded = torch.zeros((len(batch), maxT, F), dtype=feats[0].dtype)
    mask = torch.zeros((len(batch), maxT), dtype=torch.bool)
    for i, f in enumerate(feats):
        t = f.shape[0]
        padded[i, :t] = f
        mask[i, :t] = True
    labels = torch.stack([b['label'] for b in batch], 0)
    paths = [b['path'] for b in batch]
    return {'features': padded, 'mask': mask, 'lengths': torch.tensor(lengths, dtype=torch.long), 'label': labels, 'path': paths}
