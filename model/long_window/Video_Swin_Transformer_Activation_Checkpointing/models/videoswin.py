#!/usr/bin/env python3
from __future__ import annotations
"""3D Video Swin Transformer (clean restored single implementation).

Features:
- Windowed + shifted 3D self-attention with relative position bias.
- Hierarchical 2x2x2 patch merging.
- Binary mask propagation for padding-aware global pooling.
- Optional activation checkpointing (later stages only).

Input: (B, T, C, H, W)
Output: logits, pooled_feature
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from typing import Tuple, List, Optional

def _trunc_normal_(tensor, std=0.02):
    nn.init.trunc_normal_(tensor, std=std)

def window_partition(x: torch.Tensor, window_size: Tuple[int,int,int]):
    B, T, H, W, C = x.shape
    Wt, Wh, Ww = window_size
    x = x.view(B, T//Wt, Wt, H//Wh, Wh, W//Ww, Ww, C)
    return x.permute(0,1,3,5,2,4,6,7).contiguous().view(-1, Wt, Wh, Ww, C)

def window_reverse(windows: torch.Tensor, window_size: Tuple[int,int,int], B: int, T: int, H: int, W: int):
    Wt, Wh, Ww = window_size
    x = windows.view(B, T//Wt, H//Wh, W//Ww, Wt, Wh, Ww, -1)
    return x.permute(0,1,4,2,5,3,6,7).contiguous().view(B, T, H, W, -1)

class RelativePositionBias3D(nn.Module):
    def __init__(self, window_size: Tuple[int,int,int], num_heads: int):
        super().__init__()
        Wt, Wh, Ww = window_size
        self.num_heads = num_heads
        coords = torch.stack(torch.meshgrid(
            torch.arange(Wt), torch.arange(Wh), torch.arange(Ww), indexing='ij'))  # (3,Wt,Wh,Ww)
        coords_flat = coords.flatten(1).t()  # (N,3)
        rel = coords_flat[:, None] - coords_flat[None]
        rel[:, :, 0] += Wt - 1
        rel[:, :, 1] += Wh - 1
        rel[:, :, 2] += Ww - 1
        rel[:, :, 0] *= (2*Wh - 1)*(2*Ww - 1)
        rel[:, :, 1] *= (2*Ww - 1)
        rel_index = rel.sum(-1)
        size = (2*Wt-1)*(2*Wh-1)*(2*Ww-1)
        self.register_buffer('rel_index', rel_index, persistent=False)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(size, num_heads))
        _trunc_normal_(self.relative_position_bias_table)
        with torch.no_grad():
            assert int(self.rel_index.max()) < size
    def forward(self):
        idx = self.rel_index.view(-1)
        bias = self.relative_position_bias_table[idx].view(self.rel_index.size(0), self.rel_index.size(1), self.num_heads)
        return bias.permute(2,0,1)

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos = RelativePositionBias3D(window_size, num_heads)
    def forward(self, x):  # (BnW,N,C)
        BnW,N,C = x.shape
        qkv = self.qkv(x).reshape(BnW,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0]*self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) + self.rel_pos().unsqueeze(0)
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1,2).reshape(BnW,N,C)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__(); h=int(dim*mlp_ratio)
        self.fc1=nn.Linear(dim,h); self.act=nn.GELU(); self.fc2=nn.Linear(h,dim); self.drop=nn.Dropout(0.0)
    def forward(self,x):
        x=self.fc1(x); x=self.act(x); x=self.drop(x); x=self.fc2(x); x=self.drop(x); return x

class DropPath(nn.Module):
    def __init__(self, p=0.0): super().__init__(); self.p=p
    def forward(self,x):
        if self.p==0.0 or not self.training: return x
        keep=1-self.p; shape=(x.shape[0],)+(1,)*(x.ndim-1)
        rand=keep+torch.rand(shape, device=x.device, dtype=x.dtype); rand.floor_();
        return x/keep*rand

class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift, mlp_ratio, drop_path):
        super().__init__()
        self.window_size=window_size
        self.shift_size=tuple(w//2 for w in window_size) if shift else (0,0,0)
        self.norm1=nn.LayerNorm(dim)
        self.attn=WindowAttention3D(dim, window_size, num_heads)
        self.drop_path=DropPath(drop_path) if drop_path>0 else nn.Identity()
        self.norm2=nn.LayerNorm(dim)
        self.mlp=MLP(dim, mlp_ratio)
    def forward(self,x):
        B,T,H,W,C=x.shape
        shortcut=x
        x=self.norm1(x.reshape(B*T*H*W,C)).reshape(B,T,H,W,C)
        if any(self.shift_size):
            x=torch.roll(x, shifts=(-self.shift_size[0],-self.shift_size[1],-self.shift_size[2]), dims=(1,2,3))
        Wt,Wh,Ww=self.window_size
        pad_t=(Wt - (T%Wt))%Wt; pad_h=(Wh-(H%Wh))%Wh; pad_w=(Ww-(W%Ww))%Ww
        if pad_t+pad_h+pad_w>0:
            x=F.pad(x,(0,0,0,pad_w,0,pad_h,0,pad_t))
            T_pad,H_pad,W_pad=T+pad_t,H+pad_h,W+pad_w
        else:
            T_pad,H_pad,W_pad=T,H,W
        windows=window_partition(x,self.window_size).view(-1,Wt*Wh*Ww,C)
        attn=self.attn(windows).view(-1,Wt,Wh,Ww,C)
        x=window_reverse(attn,self.window_size,B,T_pad,H_pad,W_pad)
        # Crop padding if added
        x=x[:, :T, :H, :W, :]
        if any(self.shift_size):
            x=torch.roll(x, shifts=self.shift_size, dims=(1,2,3))
        x=shortcut + self.drop_path(x)
        # Update shape vars explicitly (in case of padding path)
        _,T,H,W,_=x.shape
        x2=self.norm2(x.reshape(B*T*H*W,C)).reshape(B,T,H,W,C)
        x=x + self.drop_path(self.mlp(x2.reshape(B*T*H*W,C)).reshape(B,T,H,W,C))
        return x

class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.reduction=nn.Linear(dim*8, dim*2, bias=False); self.norm=nn.LayerNorm(dim*8)
    def forward(self,x,mask:Optional[torch.Tensor]=None):  # mask (B,T,H,W,1) binary
        B,T,H,W,C=x.shape
        pad_t=(2-(T%2))%2; pad_h=(2-(H%2))%2; pad_w=(2-(W%2))%2
        if pad_t+pad_h+pad_w>0:
            x=F.pad(x,(0,0,0,pad_w,0,pad_h,0,pad_t));
            if mask is not None: mask=F.pad(mask,(0,0,0,pad_w,0,pad_h,0,pad_t))
            T+=pad_t; H+=pad_h; W+=pad_w
        x=x.view(B,T//2,2,H//2,2,W//2,2,C).permute(0,1,3,5,2,4,6,7).contiguous().view(B,T//2,H//2,W//2,8*C)
        if mask is not None:
            m=mask.permute(0,4,1,2,3)  # (B,1,T,H,W)
            m=F.max_pool3d(m,2,2)     # binary propagate
            mask=m.permute(0,2,3,4,1)
        x_lin=self.norm(x.reshape(B*T*H*W//8,8*C)); x_lin=self.reduction(x_lin).reshape(B,T//2,H//2,W//2,-1)
        return x_lin, mask

@dataclass
class VideoSwin3DConfig:
    in_chans:int=3; embed_dim:int=96; depths:Tuple[int,...]=(2,2,6,2); num_heads:Tuple[int,...]=(3,6,12,24)
    window_size:Tuple[int,int,int]=(2,7,7); mlp_ratio:float=4.0; drop_rate:float=0.0; attn_drop_rate:float=0.0
    drop_path_rate:float=0.1; num_classes:int=2; use_checkpoint:bool=False

class VideoSwin3DFeature(nn.Module):
    def __init__(self,cfg:VideoSwin3DConfig):
        super().__init__()
        self.use_checkpoint=bool(cfg.use_checkpoint)
        self.patch_embed=nn.Conv3d(cfg.in_chans,cfg.embed_dim,kernel_size=(2,4,4),stride=(2,4,4))
        self.pos_bias=nn.Parameter(torch.zeros(1,1,1,1,cfg.embed_dim)); _trunc_normal_(self.pos_bias)
        dpr=torch.linspace(0,cfg.drop_path_rate,sum(cfg.depths)).tolist()
        self.stages=nn.ModuleList(); dims=[cfg.embed_dim,cfg.embed_dim*2,cfg.embed_dim*4,cfg.embed_dim*8]
        cur=0
        for si,depth in enumerate(cfg.depths):
            blocks=[]
            for b in range(depth):
                shift=(b%2==1)
                blocks.append(SwinBlock3D(dims[si], cfg.num_heads[si], cfg.window_size, shift, cfg.mlp_ratio, dpr[cur+b]))
            cur+=depth
            down=PatchMerging3D(dims[si]) if si < len(cfg.depths)-1 else nn.Identity()
            self.stages.append(nn.ModuleDict({'blocks':nn.ModuleList(blocks),'down':down}))
        self.norm=nn.LayerNorm(dims[-1]); self.head=nn.Linear(dims[-1], cfg.num_classes); self.ckpt_stage_start=2
    def forward(self,x):  # (B,T,C,H,W)
        B,T,C,H,W=x.shape
        x=x.permute(0,2,1,3,4)
        x=self.patch_embed(x)
        T2,H2,W2=x.shape[2],x.shape[3],x.shape[4]
        x=x.permute(0,2,3,4,1)+self.pos_bias
        mask=torch.ones((B,T2,H2,W2,1), device=x.device, dtype=x.dtype)
        for si,stage in enumerate(self.stages):
            for blk in stage['blocks']:
                if self.use_checkpoint and si>=self.ckpt_stage_start and self.training:
                    # Explicitly set use_reentrant=False (PyTorch >=2.3 warning otherwise)
                    x=checkpoint(blk, x, use_reentrant=False)
                else:
                    x=blk(x)
            if isinstance(stage['down'], PatchMerging3D):
                x,mask=stage['down'](x,mask)
        w=(mask>0).float() if mask is not None else None
        if w is not None:
            valid=w.sum(dim=(1,2,3,4)).clamp_min(1.0); feat=(x*w).sum(dim=(1,2,3))/valid.unsqueeze(-1)
        else:
            feat=x.mean(dim=(1,2,3))
        logits=self.head(self.norm(feat))
        return logits, feat

def build_videoswin3d_feature(in_chans:int, embed_dim:int, depths:List[int], num_heads:List[int], window_size:List[int],
                              mlp_ratio:float, drop_rate:float, attn_drop_rate:float, drop_path_rate:float,
                              num_classes:int, use_checkpoint:bool)->VideoSwin3DFeature:
    cfg=VideoSwin3DConfig(in_chans,embed_dim,tuple(depths),tuple(num_heads),tuple(window_size),mlp_ratio,drop_rate,attn_drop_rate,drop_path_rate,num_classes,use_checkpoint)
    return VideoSwin3DFeature(cfg)

# Preset ("full") depth settings akin to Swin video variants.
# tiny : standard light config
# small: deeper (18 blocks in stage 3)
# base : wider embed_dim and more heads
_PRESET_CONFIGS = {
    'tiny':  dict(embed_dim=96,  depths=(2,2,6,2),  num_heads=(3,6,12,24), drop_path_rate=0.1),
    'small': dict(embed_dim=96,  depths=(2,2,18,2), num_heads=(3,6,12,24), drop_path_rate=0.2),
    'base':  dict(embed_dim=128, depths=(2,2,18,2), num_heads=(4,8,16,32), drop_path_rate=0.3),
}

def build_videoswin3d_preset(name: str,
                             in_chans: int = 3,
                             window_size: Tuple[int,int,int] = (2,7,7),
                             mlp_ratio: float = 4.0,
                             drop_rate: float = 0.0,
                             attn_drop_rate: float = 0.0,
                             num_classes: int = 2,
                             use_checkpoint: bool = False,
                             **overrides) -> VideoSwin3DFeature:
    """Build a preset model by name (tiny/small/base) and allow field overrides.

    Examples:
        build_videoswin3d_preset('small', num_classes=5)
        build_videoswin3d_preset('base', drop_rate=0.1, use_checkpoint=True)
    """
    name_l = name.lower()
    if name_l not in _PRESET_CONFIGS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(_PRESET_CONFIGS.keys())}")
    cfg_dict = dict(_PRESET_CONFIGS[name_l])
    cfg_dict.update(overrides)  # user overrides embed_dim/depths/heads/drop_path if given
    cfg = VideoSwin3DConfig(
        in_chans=in_chans,
        embed_dim=cfg_dict['embed_dim'],
        depths=tuple(cfg_dict['depths']),
        num_heads=tuple(cfg_dict['num_heads']),
        window_size=tuple(window_size),
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=cfg_dict.get('drop_path_rate', 0.1),
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
    )
    return VideoSwin3DFeature(cfg)

__all__=['VideoSwin3DFeature','VideoSwin3DConfig','build_videoswin3d_feature','build_videoswin3d_preset']
