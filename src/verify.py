"""STEP 0 — a 30-second self-check. Run this before anything else.

The attack relies on a hook into the model (see common.py) that depends on
exactly how your installed Ultralytics version structures YOLOv10. This script
confirms two things on YOUR setup:
  1. `dense()` returns the full per-anchor confidence tensor (~8400 values), and
  2. gradients actually flow from those confidences back to the input pixels.
If either fails, nothing downstream can work — so check here first. A failure
prints what the model actually returned, so you can adapt common.py's hook.

Usage:
    python src/verify.py --weights yolov10n.pt --device cpu
"""
import argparse
import torch

from common import YoloV10Dense, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov10n.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device={device}")
    m = YoloV10Dense(args.weights, device)

    x = torch.rand(1, 3, args.imgsz, args.imgsz, device=device, requires_grad=True)

    # Raw forward, so you can SEE what your version returns.
    with torch.no_grad():
        raw = m.net(x)
    print("raw forward return type:", type(raw))
    if isinstance(raw, (tuple, list)):
        print("  tuple len:", len(raw),
              "| elem0:", getattr(raw[0], "shape", type(raw[0])))

    boxes, conf = m.dense(x)
    print(f"boxes: {tuple(boxes.shape)}  conf: {tuple(conf.shape)}")
    print(f"num anchors (A) = {conf.shape[-1]}  | conf range [{conf.min():.3f}, {conf.max():.3f}]")
    print(f"# anchors above 0.25 on random input: {(conf > 0.25).sum().item()} "
          "(expect ~0; that's the learned suppression we'll attack)")

    # Gradient sanity: does d(conf)/d(input) exist and is it nonzero?
    loss = torch.sigmoid(50.0 * (conf - 0.25)).sum()
    g = torch.autograd.grad(loss, x)[0]
    print(f"input-gradient norm = {g.norm():.4e}  (must be > 0 for the attack to work)")
    print("OK" if g.norm() > 0 else "FAIL: no gradient to the input")


if __name__ == "__main__":
    main()
