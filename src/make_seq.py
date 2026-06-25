"""Generate a local frame SEQUENCE from a single image by panning a square
crop across it, so the tracker has coherent frame-to-frame motion to
associate. A stand-in for real footage on a laptop — swap in KITTI/MOT/ffmpeg
frames later by just pointing the harness at that folder instead.

    python src/make_seq.py --image data/dev/bus.jpg --out data/seq_clean --frames 16
"""
import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=16)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    h, w = img.shape[:2]
    side = min(h, w)
    os.makedirs(args.out, exist_ok=True)

    n = args.frames
    for i in range(n):
        t = i / max(n - 1, 1)
        if h >= w:                       # portrait -> pan vertically
            y0 = round((h - side) * t)
            crop = img[y0:y0 + side, 0:side]
        else:                            # landscape -> pan horizontally
            x0 = round((w - side) * t)
            crop = img[0:side, x0:x0 + side]
        cv2.imwrite(os.path.join(args.out, f"{i:06d}.jpg"), crop)
    print(f"wrote {n} frames ({side}x{side}) to {args.out}")


if __name__ == "__main__":
    main()
