"""PGD sponge/overload attack on an NMS-free YOLOv10 detector.

Crafts an L-inf bounded perturbation that maximizes the number of *diverse*
detections surviving the one2one head's learned suppression, so the
downstream tracker is flooded every frame.

Also generates the random-noise control (same epsilon) — that comparison is
load-bearing: noise must FAIL to flood while the gradient attack succeeds.

CPU-first. Use --device cuda (or auto) on the cluster for bulk crafting.

    python src/attack.py --images data/kitti_0011_clean --out output/kitti_0011_adv --device cpu --iters 40
    python src/attack.py --images data/kitti_0011_clean --out output/kitti_0011_noise --mode noise
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from common import YoloV10Dense, get_device, image_to_tensor


def weighted_var(v, w, eps=1e-6):
    """Weighted variance of values v ([B,A]) under soft weights w ([B,A]).
    Used to reward spatially DIVERSE above-threshold detections (real tracker work)."""
    wsum = w.sum(dim=1, keepdim=True) + eps
    mean = (w * v).sum(dim=1, keepdim=True) / wsum
    var = (w * (v - mean) ** 2).sum(dim=1) / wsum.squeeze(1)
    return var.mean()


def pixel_mask(x0, content):
    """[1,1,H,W] mask: 1 inside the scene-content rect, 0 in the letterbox padding."""
    x0c, y0c, x1c, y1c = content
    m = torch.zeros_like(x0[:, :1])
    m[:, :, y0c:y1c, x0c:x1c] = 1.0
    return m


def anchor_mask(model, content):
    """[A] mask of anchors whose center lies in the scene-content rect. Restricts
    the flood/spread loss to content anchors (valid after a dense() call)."""
    ax, ay = model.anchor_centers()
    x0c, y0c, x1c, y1c = content
    return ((ax >= x0c) & (ax < x1c) & (ay >= y0c) & (ay < y1c)).float()


def pgd_attack(model, x0, content, tau, alpha, eps, step, iters, lam, imgsz):
    """Returns adversarial image (same shape as x0, in [0,1]). The perturbation
    and the loss are restricted to the scene-content region (no padding cheating)."""
    pmask = pixel_mask(x0, content)
    model.dense(x0)                                  # populate head anchors for the mask
    amask = anchor_mask(model, content)
    delta = torch.zeros_like(x0, requires_grad=True)
    for _ in range(iters):
        boxes, conf = model.dense((x0 + delta).clamp(0, 1))
        w = torch.sigmoid(alpha * (conf - tau)) * amask   # only content anchors
        flood = w.sum(dim=1).mean()                       # L_flood: maximize count
        cx = (boxes[:, 0, :] + boxes[:, 2, :]) / 2 / imgsz   # box centers (boxes are xyxy)
        cy = (boxes[:, 1, :] + boxes[:, 3, :]) / 2 / imgsz
        spread = weighted_var(cx, w) + weighted_var(cy, w)   # L_spread: diversity
        loss = -(flood + lam * spread)                   # PGD minimizes -> count goes up
        g = torch.autograd.grad(loss, delta)[0]
        delta = (delta - step * g.sign()).clamp(-eps, eps)
        delta = delta * pmask                            # freeze padding pixels at grey
        delta = ((x0 + delta).clamp(0, 1) - x0).detach().requires_grad_(True)
    return (x0 + delta).clamp(0, 1).detach()


def random_noise(x0, content, eps):
    """Control: uniform L-inf noise of the same budget, restricted to content."""
    d = (torch.rand_like(x0) * 2 - 1) * eps * pixel_mask(x0, content)
    return (x0 + d).clamp(0, 1).detach()


def tensor_to_bgr(t):
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of input images")
    ap.add_argument("--out", required=True, help="output folder for adversarial frames")
    ap.add_argument("--weights", default="yolov10n.pt")
    ap.add_argument("--device", default="cpu")        # <-- the device flag
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--mode", choices=["attack", "noise"], default="attack")
    ap.add_argument("--tau", type=float, default=0.25, help="MUST match deployed conf threshold")
    ap.add_argument("--alpha", type=float, default=50.0, help="soft-count sharpness")
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--step", type=float, default=2 / 255)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--lam", type=float, default=0.1, help="L_spread weight")
    args = ap.parse_args()

    device = get_device(args.device)
    torch.set_num_threads(os.cpu_count() or 1)        # CPU YOLO is thread-bound
    print(f"device={device} threads={torch.get_num_threads()} mode={args.mode}")

    model = YoloV10Dense(args.weights, device)
    os.makedirs(args.out, exist_ok=True)

    paths = sorted(
        p for ext in ("jpg", "jpeg", "png", "bmp")
        for p in glob.glob(os.path.join(args.images, f"*.{ext}"))
    )
    if not paths:
        raise SystemExit(f"no images found in {args.images}")

    for i, p in enumerate(paths):
        x0, content = image_to_tensor(p, args.imgsz, device)
        with torch.no_grad():
            _, c0 = model.dense(x0)
            amask = anchor_mask(model, content)          # count content anchors only
        before = int(((c0 > args.tau).float() * amask).sum())

        if args.mode == "attack":
            adv = pgd_attack(model, x0, content, args.tau, args.alpha, args.eps,
                             args.step, args.iters, args.lam, args.imgsz)
        else:
            adv = random_noise(x0, content, args.eps)

        with torch.no_grad():
            _, c1 = model.dense(adv)
        after = int(((c1 > args.tau).float() * amask).sum())

        out_path = os.path.join(args.out, os.path.basename(p))
        cv2.imwrite(out_path, tensor_to_bgr(adv))
        print(f"[{i+1}/{len(paths)}] {os.path.basename(p)}: "
              f"above-thresh {before} -> {after}")


if __name__ == "__main__":
    main()
