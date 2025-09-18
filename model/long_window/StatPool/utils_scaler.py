import json
import numpy as np
from pathlib import Path

class StandardScalerJSON:
    def __init__(self, mean=None, scale=None):
        self.mean = np.array(mean) if mean is not None else None
        self.scale = np.array(scale) if scale is not None else None
    def fit(self, X: np.ndarray):
        # X: (N, D)
        self.mean = np.mean(X, axis=0).tolist()
        self.scale = np.std(X, axis=0).tolist()
        # avoid zero
        self.scale = [s if s>1e-6 else 1.0 for s in self.scale]
    def transform(self, X: np.ndarray):
        return (X - np.array(self.mean)) / np.array(self.scale)
    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)
    def to_json(self, p: Path):
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w') as f:
            json.dump({'mean': list(map(float, self.mean)), 'scale': list(map(float, self.scale))}, f)
    @classmethod
    def from_json(cls, p: Path):
        with open(p,'r') as f:
            j = json.load(f)
        return cls(mean=j['mean'], scale=j['scale'])
