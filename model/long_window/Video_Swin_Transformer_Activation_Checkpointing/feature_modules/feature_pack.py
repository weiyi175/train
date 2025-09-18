#!/usr/bin/env python3
from __future__ import annotations
"""Feature Pack Builder
Generates multi-channel tensor from raw joint feature sequence.
Input shape: (T, F) where F = joints * dims (e.g., 15*4 = 60 if (x,y,score,visibility)).
Outputs:
  dict with keys:
    base: (T, F) original (optionally normalized already upstream)
    velocity: (T, F) first difference (pad first with zeros)
    accel: (T, F) second difference (pad first two with zeros)
    energy: (T, 1) per-frame motion magnitude (L2 of velocity grouped per joint)
    pairwise: (T, P) optional subset pairwise distances (torso referenced)
Combined (pack_for_model) will stack selected components along channel dim and reshape to (T, C, H, W) given a target grid.
"""
from dataclasses import dataclass
import torch
import math
from typing import List, Tuple, Dict

@dataclass
class FeaturePackConfig:
    use_velocity: bool = True
    use_accel: bool = True
    use_energy: bool = True
    use_pairwise: bool = True
    joints: int = 15
    dims_per_joint: int = 4  # (x,y,score,vis)
    pairwise_subset: int = 20  # number of pairwise distances to keep (top-k by variance later)

class FeaturePackBuilder:
    def __init__(self, cfg: FeaturePackConfig):
        self.cfg = cfg

    def build(self, seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        # seq: (T, F)
        T, F = seq.shape
        out: Dict[str, torch.Tensor] = {'base': seq}
        # Derive joints if product mismatch and divisible by dims_per_joint
        d = self.cfg.dims_per_joint
        joints = self.cfg.joints
        if joints * d != F and F % d == 0:
            joints = F // d
        safe_viewable = (joints * d == F)
        if self.cfg.use_velocity:
            vel = torch.zeros_like(seq)
            vel[1:] = seq[1:] - seq[:-1]
            out['velocity'] = vel
        if self.cfg.use_accel:
            acc = torch.zeros_like(seq)
            acc[2:] = seq[2:] - 2*seq[1:-1] + seq[:-2]
            out['accel'] = acc
        if self.cfg.use_energy and self.cfg.use_velocity and safe_viewable:
            vel_j = vel.view(T, joints, d)
            mag = torch.linalg.norm(vel_j[..., :2], dim=-1)  # (T, joints)
            energy = mag.mean(dim=-1, keepdim=True)
            out['energy'] = energy
        if self.cfg.use_pairwise and safe_viewable:
            base_j = seq.view(T, joints, d)
            ref = base_j[:, :1, :2]
            coords = base_j[..., :2]
            diff = coords - ref
            dist = torch.linalg.norm(diff, dim=-1)
            var = dist.var(dim=0)
            k = min(self.cfg.pairwise_subset, dist.shape[1])
            idx = torch.argsort(var, descending=True)[:k]
            pair_sel = dist[:, idx]
            out['pairwise'] = pair_sel
        return out

    def pack_for_model(self, feat_dict: Dict[str, torch.Tensor], grid: Tuple[int,int]) -> torch.Tensor:
        # Concatenate along feature dimension then reshape to grid channels
        comps = []
        for k in ['base','velocity','accel','energy','pairwise']:
            if k in feat_dict: comps.append(feat_dict[k])
        cat = torch.cat([c if c.ndim==2 else c for c in comps], dim=1)  # (T, totalF)
        T, FT = cat.shape
        H,W = grid
        C = math.ceil(FT / (H*W))
        pad = C*H*W - FT
        if pad>0:
            cat = torch.nn.functional.pad(cat, (0,pad))
        cat = cat.view(T, C, H, W)
        return cat

__all__ = ['FeaturePackConfig','FeaturePackBuilder']
