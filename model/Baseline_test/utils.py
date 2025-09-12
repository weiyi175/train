from __future__ import annotations
import os, json, math, time
from typing import Dict, Any


def ensure_incremental_run_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    # find next index like 01, 02, 03 ...
    existing = [d for d in os.listdir(base_dir) if d.isdigit() and len(d)<=3]
    nums = [int(d) for d in existing]
    nxt = max(nums) + 1 if nums else 1
    name = f"{nxt:02d}"
    run_dir = os.path.join(base_dir, name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def save_json(obj: Dict[str, Any], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class AvgMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, val: float, k: int = 1):
        self.sum += float(val) * int(k)
        self.n += int(k)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.n)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def write_lines(path: str, lines: str | list[str]):
    s = lines if isinstance(lines, str) else "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(s)
