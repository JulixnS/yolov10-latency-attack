"""Latency measurement harness — the part that produces the evidence.

Thesis to demonstrate:
  * DETECTOR latency stays ~flat clean vs adversarial (NMS-free forward is
    fixed compute) while detection COUNT jumps to ~max_det.
  * TRACKER latency EXPLODES on the adversarial stream (association cost
    scales with detection count).

Measure on the laptop CPU — the tracker is CPU-bound, so this is the
representative place to run it, not a compromise.

    # detector-side (any images): flat latency, exploding count
    python src/measure.py detector --clean data/dev --adv out/adv --device cpu

    # tracker-side (needs CONSECUTIVE frames / a sequence dir):
    python src/measure.py tracker --clean data/seq_clean --adv out/seq_adv --device cpu
"""
import argparse
import glob
import os
import time
from types import SimpleNamespace

import yaml
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import ROOT

from common import get_device


def _new_tracker():
    """Fresh ByteTracker built from Ultralytics' bundled bytetrack.yaml.
    Swap this for your SlowTrack tracker — the harness only needs an object
    with `.update(boxes_numpy, orig_img)`."""
    cfg = yaml.safe_load(open(ROOT / "cfg/trackers/bytetrack.yaml"))
    cfg.setdefault("fuse_score", True)
    return BYTETracker(SimpleNamespace(**cfg))


def _frames(d):
    return sorted(
        p for ext in ("jpg", "jpeg", "png", "bmp")
        for p in glob.glob(os.path.join(d, f"*.{ext}"))
    )


def measure_detector(yolo, folder, conf, device, imgsz):
    paths = _frames(folder)
    lat, counts = [], []
    for p in paths:
        t = time.perf_counter()
        r = yolo.predict(p, conf=conf, device=device, imgsz=imgsz, verbose=False)[0]
        lat.append((time.perf_counter() - t) * 1000)
        counts.append(len(r.boxes))
    n = len(paths)
    return {
        "n": n,
        "lat_ms_mean": sum(lat) / n,
        "det_count_mean": sum(counts) / n,
        "det_count_max": max(counts),
    }


def measure_tracker(yolo, folder, conf, device, imgsz):
    """Runs the detector per frame, then times the tracker.update() call in
    ISOLATION (not detector+tracker minus detector — that buries the few-ms
    tracker signal in detector noise). Reports detector and tracker latency
    separately so the multiplier is clean."""
    paths = _frames(folder)
    tracker = _new_tracker()              # fresh state per sequence
    det_lat, trk_lat, ndet, ntracks = [], [], [], []
    for p in paths:
        t = time.perf_counter()
        r = yolo.predict(p, conf=conf, device=device, imgsz=imgsz, verbose=False)[0]
        det_lat.append((time.perf_counter() - t) * 1000)
        det = r.boxes.cpu().numpy()       # what Ultralytics feeds the tracker
        ndet.append(len(det))
        t = time.perf_counter()
        tracks = tracker.update(det, r.orig_img)
        trk_lat.append((time.perf_counter() - t) * 1000)
        ntracks.append(len(tracks))
    n = len(paths)
    return {
        "n": n,
        "det_ms_mean": round(sum(det_lat) / n, 2),
        "tracker_ms_mean": round(sum(trk_lat) / n, 3),
        "tracker_ms_max": round(max(trk_lat), 3),
        "det_count_mean": round(sum(ndet) / n, 1),
        "tracks_mean": round(sum(ntracks) / n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["detector", "tracker"])
    ap.add_argument("--clean", required=True)
    ap.add_argument("--adv", required=True)
    ap.add_argument("--weights", default="yolov10n.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    device = get_device(args.device)
    yolo = YOLO(args.weights)
    fn = measure_detector if args.mode == "detector" else measure_tracker

    print(f"== {args.mode} | device={device} ==")
    for label, folder in (("CLEAN", args.clean), ("ADVERSARIAL", args.adv)):
        stats = fn(yolo, folder, args.conf, device, args.imgsz)
        print(f"{label:12s} {stats}")
    print("Expect: detector lat ~flat & count up  |  tracker lat up sharply on ADVERSARIAL")


if __name__ == "__main__":
    main()
