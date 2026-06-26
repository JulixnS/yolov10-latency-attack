"""Stitch a folder of frames into an mp4 — by default with the detector's boxes
drawn, since that's what makes the attack visible (raw adversarial frames look
identical to clean). Frames are letterboxed to imgsz so a clean run and an adv
run come out the same size and are directly comparable.

    python src/make_videos.py --frames data/kitti_0011_clean --out output/clean.mp4
    python src/make_videos.py --frames output/kitti_0011_adv  --out output/attacked.mp4
    # ...add --raw for no boxes (to confirm the perturbation is invisible)
"""
import argparse
import glob
import os
import shutil
import subprocess

import cv2
from ultralytics import YOLO

from common import get_device, letterbox


def _to_h264(path):
    """Re-encode in place to H.264/yuv420p so it plays in VSCode/browsers
    (OpenCV writes mp4v, which their Chromium players can't decode). No-op if
    ffmpeg is unavailable."""
    if not shutil.which("ffmpeg"):
        print("  (ffmpeg not found — leaving mp4v; plays in VLC, not VSCode/browser)")
        return
    tmp = path + ".h264.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, path)


def _frames(d):
    return sorted(
        p for ext in ("png", "jpg", "jpeg", "bmp")
        for p in glob.glob(os.path.join(d, f"*.{ext}"))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder of consecutive frames")
    ap.add_argument("--out", required=True, help="output .mp4 path")
    ap.add_argument("--fps", type=float, default=10.0, help="KITTI native ≈10")
    ap.add_argument("--raw", action="store_true", help="no detector boxes (invisibility check)")
    ap.add_argument("--weights", default="yolov10n.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    paths = _frames(args.frames)
    if not paths:
        raise SystemExit(f"no frames in {args.frames}")
    yolo = None if args.raw else YOLO(args.weights)
    device = get_device(args.device)

    writer, W, H = None, None, None
    for p in paths:
        sq, _, _ = letterbox(cv2.imread(p), args.imgsz)     # common 640×640 view
        if args.raw:
            frame = sq
        else:
            r = yolo.predict(sq, conf=args.conf, device=device, imgsz=args.imgsz, verbose=False)[0]
            frame = r.plot()
        if writer is None:
            H, W = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (W, H))
        if (frame.shape[1], frame.shape[0]) != (W, H):
            frame = cv2.resize(frame, (W, H))
        writer.write(frame)
    writer.release()
    _to_h264(args.out)
    print(f"wrote {args.out}  ({len(paths)} frames @ {args.fps} fps, {len(paths)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
