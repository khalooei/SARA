"""Robustness to non-adversarial distribution shift: Gaussian noise, Gaussian blur,
block occlusion and reduced brightness, applied on the fly to the test set at several
severities. Reports the accuracy of a given checkpoint under each corruption.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import torch
import torch.nn.functional as F

from sara.data import get_loaders
from sara.models import build_model


def gaussian_noise(x, severity):
    sigma = [0.02, 0.05, 0.08, 0.12, 0.18][severity - 1]
    return (x + torch.randn_like(x) * sigma).clamp(0, 1)


def gaussian_blur(x, severity):
    k = [3, 3, 5, 5, 7][severity - 1]
    sigma = [0.5, 1.0, 1.5, 2.0, 2.5][severity - 1]
    pad = k // 2
    coords = torch.arange(k, dtype=torch.float32, device=x.device) - pad
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, -1)
    kernel_x = g.view(1, 1, 1, k).repeat(x.shape[1], 1, 1, 1)
    kernel_y = g.view(1, 1, k, 1).repeat(x.shape[1], 1, 1, 1)
    xb = F.conv2d(x, kernel_x, padding=(0, pad), groups=x.shape[1])
    xb = F.conv2d(xb, kernel_y, padding=(pad, 0), groups=x.shape[1])
    return xb.clamp(0, 1)


def occlusion(x, severity):
    frac = [0.05, 0.1, 0.15, 0.25, 0.35][severity - 1]
    h, w = x.shape[-2], x.shape[-1]
    side = max(1, int((frac ** 0.5) * min(h, w)))
    out = x.clone()
    for i in range(x.shape[0]):
        top = torch.randint(0, max(1, h - side + 1), (1,)).item()
        left = torch.randint(0, max(1, w - side + 1), (1,)).item()
        out[i, :, top:top + side, left:left + side] = 0.0
    return out


def low_light(x, severity):
    factor = [0.8, 0.65, 0.5, 0.35, 0.2][severity - 1]
    return (x * factor).clamp(0, 1)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "occlusion": occlusion,
    "low_light": low_light,
}


@torch.no_grad()
def evaluate_corruptions(model, loader, device, severities=(1, 3, 5)):
    model.eval()
    results = {}
    for name, fn in CORRUPTIONS.items():
        for sev in severities:
            correct, n = 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                x_c = fn(x, sev)
                logits = model(x_c)
                correct += (logits.argmax(dim=1) == y).sum().item()
                n += x.shape[0]
            results[f"{name}_sev{sev}"] = 100.0 * correct / n
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--model", default="lenet")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-subset", type=int, default=2000)
    p.add_argument("--label", default="model")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "runs", "corruption", "corruption.jsonl"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_loaders(args.dataset, batch_size=256, test_subset=args.test_subset)
    model = build_model(args.model, args.dataset).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    out = evaluate_corruptions(model, test_loader, device)
    out["label"] = args.label
    out["checkpoint"] = args.checkpoint
    print(json.dumps(out, indent=2))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(out) + "\n")
