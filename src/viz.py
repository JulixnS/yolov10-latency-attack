"""See what the model sees: original vs attacked image, both with the
detector's boxes drawn, stitched side-by-side with detection counts.

This is the project's visual deliverable — the perturbation is invisible to
the eye, but the attacked image is saturated with phantom boxes.

    python src/viz.py --clean data/kitti_0011_clean/000010.png --adv output/kitti_0011_adv/000010.png --device cpu
"""
import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO

from common import get_device


def annotated(yolo, path, conf, device, imgsz):
    """Run the detector on the image and draw ITS OWN predicted boxes on it
    (these are model predictions, not ground-truth labels). Returns the
    boxes-drawn image and the number of detections."""
    r = yolo.predict(path, conf=conf, device=device, imgsz=imgsz, verbose=False)[0]
    return r.plot(), len(r.boxes)        # r.plot() draws the predicted boxes


def banner(img, text, h=34):
    """Stack a black title bar with `text` on top of the image."""
    bar = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.vconcat([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="original image, e.g. data/kitti_0011_clean/000010.png")
    ap.add_argument("--adv", required=True, help="attacked image, e.g. output/kitti_0011_adv/000010.png")
    ap.add_argument("--out", default="output/compare.png")
    ap.add_argument("--weights", default="yolov10n.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25, help="match the attack's --tau")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    device = get_device(args.device)
    yolo = YOLO(args.weights)

    clean_img, cn = annotated(yolo, args.clean, args.conf, device, args.imgsz)
    adv_img, an = annotated(yolo, args.adv, args.conf, device, args.imgsz)

    # Match heights before hconcat (plot() preserves each image's own size).
    h = max(clean_img.shape[0], adv_img.shape[0])
    clean_img = cv2.resize(clean_img, (clean_img.shape[1], h))
    adv_img = cv2.resize(adv_img, (adv_img.shape[1], h))

    left = banner(clean_img, f"ORIGINAL: {cn} detections")
    right = banner(adv_img, f"ATTACKED: {an} detections")
    compare = cv2.hconcat([left, right])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, compare)
    print(f"wrote {args.out}  ({cn} -> {an} detections)")


if __name__ == "__main__":
    main()
