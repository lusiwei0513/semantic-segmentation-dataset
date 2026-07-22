#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RTX 3050 VRAM smoke: forward + backward for real MiT-B0 architecture.

Does NOT require HuggingFace download: uses SegformerConfig matching nvidia/mit-b0
with random init (VRAM footprint ≈ pretrained). Optionally loads local checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from transformers import SegformerConfig, SegformerModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.losses.multitask_loss import MultiTaskLoss
from src.models.keypoint_head import KeypointHeatmapHead
from src.models.segmentation_head import MultiScaleSegHead


def mit_b0_config() -> SegformerConfig:
    return SegformerConfig(
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
    )


class SmokeSegFormerB0(nn.Module):
    def __init__(self, keypoint_head: bool = True, channels: int = 4, stride: int = 4):
        super().__init__()
        cfg = mit_b0_config()
        self.backbone = SegformerModel(cfg)
        self.seg_head = MultiScaleSegHead(
            in_channels=list(cfg.hidden_sizes),
            decoder_channels=256,
            num_classes=channels,
        )
        self.use_keypoint_head = keypoint_head
        self.keypoint_out_stride = stride
        if keypoint_head:
            self.kp_head = KeypointHeatmapHead(
                in_channels=cfg.hidden_sizes[0], mid_channels=128
            )

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        n = len(self.backbone.config.hidden_sizes)
        features = list(outputs.hidden_states)[-n:]
        _, _, h, w = pixel_values.shape
        out = {"segmentation_logits": self.seg_head(features, output_size=(h, w))}
        if self.use_keypoint_head:
            hk, wk = h // self.keypoint_out_stride, w // self.keypoint_out_stride
            out["nose_tip_heatmap_logits"] = self.kp_head(features[0], output_size=(hk, wk))
        return out


def vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2)


def run_case(name: str, h: int, w: int, batch: int, amp: bool, keypoint_head: bool) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    model = SmokeSegFormerB0(keypoint_head=keypoint_head).to(device)
    criterion = MultiTaskLoss(
        bce_weight=1.0,
        dice_weight=1.0,
        loss_weights={
            "body": 1.0,
            "windshield": 1.0,
            "bogie": 1.0,
            "door": 1.0,
            "nose_tip": 1.0,
        },
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ok = True
    err = ""
    peak = 0.0
    try:
        for step in range(2):
            images = torch.randn(batch, 3, h, w, device=device)
            seg = torch.randint(0, 2, (batch, 4, h, w), device=device).float()
            valid = torch.ones(batch, 4, device=device)
            hh, ww = max(1, h // 4), max(1, w // 4)
            heat = torch.rand(batch, 1, hh, ww, device=device)
            valid_tip = (
                torch.ones(batch, 1, device=device)
                if keypoint_head
                else torch.zeros(batch, 1, device=device)
            )
            batch_dict = {
                "segmentation": seg,
                "valid_seg_tasks": valid,
                "nose_tip_heatmap": heat,
                "valid_nose_tip": valid_tip,
            }
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(images)
                if not keypoint_head:
                    # criterion expects tip logits key; inject zeros
                    out["nose_tip_heatmap_logits"] = torch.zeros(
                        batch, 1, hh, ww, device=device, requires_grad=True
                    )
                losses = criterion(out, batch_dict)
                loss = losses["loss_total"]
            if amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            peak = max(peak, vram_mb())
            print(f"  [{name}] step={step} loss={float(loss):.4f} peak_vram={peak:.0f}MB")
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        peak = max(peak, vram_mb())

    del model, opt, criterion, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "name": name,
        "ok": ok,
        "peak_vram_mb": peak,
        "error": err,
        "H": h,
        "W": w,
        "batch": batch,
        "amp": amp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required for this smoke"
    print("device:", torch.cuda.get_device_name(0))
    print("total_vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    free, total = torch.cuda.mem_get_info()
    print(f"free_vram_mb={free/1024**2:.0f} total_vram_mb={total/1024**2:.0f}")

    cases = [
        ("front_512_bs2_amp", 512, 512, 2, True, True),
        ("front_512_bs1_amp", 512, 512, 1, True, True),
        ("side_384x768_bs2_amp", 384, 768, 2, True, False),
        ("side_384x768_bs1_amp", 384, 768, 1, True, False),
        ("side_384x1536_bs1_amp", 384, 1536, 1, True, False),
        ("side_448x2016_bs1_amp", 448, 2016, 1, True, False),
        ("side_384x1024_bs1_amp", 384, 1024, 1, True, False),
    ]
    if args.quick:
        cases = [
            ("front_512_bs1_amp", 512, 512, 1, True, True),
            ("side_384x768_bs1_amp", 384, 768, 1, True, False),
        ]

    results = []
    for name, h, w, bs, amp, kp in cases:
        print(f"\n=== {name} ===")
        results.append(run_case(name, h, w, bs, amp, kp))

    print("\n========== SUMMARY ==========")
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        msg = f"{status:4} {r['name']:28} peak={r['peak_vram_mb']:.0f}MB"
        if not r["ok"]:
            msg += f"  err={r['error']}"
        print(msg)

    ok_front = any(r["ok"] and r["name"].startswith("front") for r in results)
    ok_side = any(r["ok"] and r["name"].startswith("side") for r in results)
    print(f"\nfront_feasible={ok_front} side_feasible={ok_side}")
    return 0 if (ok_front and ok_side) else 1


if __name__ == "__main__":
    raise SystemExit(main())
