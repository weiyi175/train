from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any


def get_next_run_dir(base: str) -> Path:
    base_p = Path(base)
    base_p.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        d = base_p / f"{idx:02d}"
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            return d
        idx += 1


def save_json(obj: Dict[str, Any], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def generate_run_report(run_dir: str):
    # thin wrapper: reuse parent's generate logic if available
    try:
        from ..Tcn_attention.utils import generate_run_report as gr
        gr(run_dir)
    except Exception:
        # fallback: write minimal report
        run = Path(run_dir)
        report = {'note': 'no detailed report generator available; fallback report created'}
        save_json(report, str(run / 'report.json'))
