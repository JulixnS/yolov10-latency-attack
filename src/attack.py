"""The attack itself: find a tiny, invisible change to an image that makes the
detector hallucinate a flood of objects (an "overload" / "sponge" attack), so
the tracker behind it gets buried in work.

How it works, in plain terms:
  * We add a small perturbation to the image and ask the (frozen) detector for
    its confidence at every location. We then use gradients to figure out which
    way to nudge each pixel so that MORE of those confidences cross the
    detection threshold. Repeat ~40 times. This is "PGD" (Projected Gradient
    Descent) — gradient descent on the image, with the change projected back into
    a tiny budget after every step so it stays invisible.
  * "Tiny budget" = L-infinity bound: no single pixel value may change by more
    than `eps` (default 8/255). That cap is what keeps the perturbation
    imperceptible.
  * We also run a control with the SAME budget but RANDOM noise instead of
    gradients. It must fail to flood — that's the proof the flood comes from the
    specific adversarial direction, not from the mere size of the change.

CPU-first. Use --device cuda (or auto) on a GPU for bulk crafting.

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
    """How spread-out the values `v` are, counting only the entries with high
    weight `w`. We use it to reward detections that are spatially SPREAD OUT
    rather than stacked on top of each other — spread-out boxes are distinct
    objects the tracker has to match individually, so they cost it more work."""
    wsum = w.sum(dim=1, keepdim=True) + eps
    mean = (w * v).sum(dim=1, keepdim=True) / wsum
    var = (w * (v - mean) ** 2).sum(dim=1) / wsum.squeeze(1)
    return var.mean()


def pixel_mask(x0, content):
    """A [1,1,H,W] mask that is 1 over the real scene and 0 over the grey letterbox
    padding. Multiplying the perturbation by this keeps the padding untouched."""
    x0c, y0c, x1c, y1c = content
    m = torch.zeros_like(x0[:, :1])
    m[:, :, y0c:y1c, x0c:x1c] = 1.0
    return m


def anchor_mask(model, content):
    """A [A] mask marking which anchors sit in the real scene (1) vs the grey
    padding (0). We only let the loss count scene anchors — a real camera attacker
    can't control the padding the pipeline adds, so flooding it would be cheating."""
    ax, ay = model.anchor_centers()
    x0c, y0c, x1c, y1c = content
    return ((ax >= x0c) & (ax < x1c) & (ay >= y0c) & (ay < y1c)).float()


def pgd_attack(model, x0, content, tau, alpha, eps, step, iters, lam, imgsz):
    """Run the PGD loop on one image and return the attacked version.

    `delta` is the perturbation we're solving for (starts at zero). `x0` is the
    clean image; `x0 + delta` is the attacked image. Each iteration nudges delta
    in the direction that creates more detections, then clips it back into the
    invisibility budget. Both the perturbation and the loss are restricted to the
    real scene (see the masks above).
    """
    pmask = pixel_mask(x0, content)                  # which pixels we're allowed to change
    model.dense(x0)                                  # one run to populate the anchor grid...
    amask = anchor_mask(model, content)              # ...so we can build the anchor mask
    delta = torch.zeros_like(x0, requires_grad=True)
    for _ in range(iters):
        # 1. Ask the detector for its confidence at every anchor on the current image.
        boxes, conf = model.dense((x0 + delta).clamp(0, 1))
        # 2. Turn confidences into a smooth, differentiable "is this a detection?" score
        #    (sigmoid is ~1 above the threshold tau, ~0 below; alpha sets the sharpness).
        #    Multiplying by amask zeroes out padding anchors so they don't count.
        w = torch.sigmoid(alpha * (conf - tau)) * amask
        # 3. The objective: "flood" = how many detections we have (sum of the soft scores);
        #    "spread" = how spatially diverse they are. We want both large.
        flood = w.sum(dim=1).mean()
        cx = (boxes[:, 0, :] + boxes[:, 2, :]) / 2 / imgsz   # box center x (boxes are x1y1x2y2)
        cy = (boxes[:, 1, :] + boxes[:, 3, :]) / 2 / imgsz   # box center y
        spread = weighted_var(cx, w) + weighted_var(cy, w)
        # 4. Loss is negated because the optimizer MINIMIZES — minimizing -flood = maximizing it.
        loss = -(flood + lam * spread)
        # 5. Gradient of the loss w.r.t. the perturbation: which way to move each pixel.
        g = torch.autograd.grad(loss, delta)[0]
        # 6. Take a fixed-size step in that direction (sign() = the optimal move under an
        #    L-inf budget), then clip back so no pixel exceeds the eps budget.
        delta = (delta - step * g.sign()).clamp(-eps, eps)
        delta = delta * pmask                            # keep the grey padding unchanged
        # 7. Make sure x0+delta is still a valid image, and reset grad tracking for next loop.
        delta = ((x0 + delta).clamp(0, 1) - x0).detach().requires_grad_(True)
    return (x0 + delta).clamp(0, 1).detach()


def random_noise(x0, content, eps):
    """The control: random noise of the SAME eps budget (and same content mask),
    but with no gradient guiding it. This should NOT flood the detector."""
    d = (torch.rand_like(x0) * 2 - 1) * eps * pixel_mask(x0, content)
    return (x0 + d).clamp(0, 1).detach()


def tensor_to_bgr(t):
    """Convert a [1,3,H,W] float tensor in [0,1] (RGB) back to a uint8 BGR image
    that OpenCV can save to disk."""
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of input frames to attack")
    ap.add_argument("--out", required=True, help="folder to write the attacked frames to")
    ap.add_argument("--weights", default="yolov10n.pt", help="YOLOv10 weights file")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    ap.add_argument("--imgsz", type=int, default=640, help="model input size (square)")
    ap.add_argument("--mode", choices=["attack", "noise"], default="attack",
                    help="'attack' = the real PGD attack; 'noise' = the random-noise control")
    ap.add_argument("--tau", type=float, default=0.25,
                    help="detection threshold the attack pushes scores above (match the deployed conf)")
    ap.add_argument("--alpha", type=float, default=50.0,
                    help="how sharply the soft detection score flips at the threshold")
    ap.add_argument("--eps", type=float, default=8 / 255,
                    help="invisibility budget: max change per pixel (8/255 ~ imperceptible)")
    ap.add_argument("--step", type=float, default=2 / 255, help="PGD step size per iteration")
    ap.add_argument("--iters", type=int, default=50, help="number of PGD iterations (more = stronger)")
    ap.add_argument("--lam", type=float, default=0.1, help="weight on the spread/diversity term")
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
        # Count detections on the CLEAN frame first (scene anchors only), for the log.
        with torch.no_grad():
            _, c0 = model.dense(x0)
            amask = anchor_mask(model, content)
        before = int(((c0 > args.tau).float() * amask).sum())

        # Craft the attacked frame (or the random-noise control).
        if args.mode == "attack":
            adv = pgd_attack(model, x0, content, args.tau, args.alpha, args.eps,
                             args.step, args.iters, args.lam, args.imgsz)
        else:
            adv = random_noise(x0, content, args.eps)

        # Count detections again on the result, and save it to disk.
        with torch.no_grad():
            _, c1 = model.dense(adv)
        after = int(((c1 > args.tau).float() * amask).sum())

        out_path = os.path.join(args.out, os.path.basename(p))
        cv2.imwrite(out_path, tensor_to_bgr(adv))
        # "above-thresh 6 -> 50" means: 6 detections clean, 50 after the attack.
        print(f"[{i+1}/{len(paths)}] {os.path.basename(p)}: "
              f"above-thresh {before} -> {after}")


if __name__ == "__main__":
    main()
