"""Latency measurement harness — the part that produces the evidence.

Thesis to demonstrate:
  * DETECTOR latency stays ~flat clean vs adversarial (NMS-free forward is
    fixed compute) while detection COUNT jumps to ~max_det.
  * TRACKER latency EXPLODES on the adversarial stream (association cost
    scales with detection count).

Measure on the laptop CPU — the tracker is CPU-bound, so this is the
representative place to run it, not a compromise.

    # detector-side (any images): flat latency, exploding count
    python src/measure.py detector --clean data/dev --adv output/adv --device cpu

    # tracker-side (needs CONSECUTIVE frames / a sequence dir):
    python src/measure.py tracker --clean data/seq_clean --adv output/seq_adv --device cpu
    # ...with SlowTrack's vendored tracker instead of bundled ByteTrack:
    python src/measure.py tracker --clean data/seq_clean --adv output/seq_adv \
        --device cpu --tracker slowtrack
"""
import argparse
import glob
import os
import time
from statistics import median
from types import SimpleNamespace

import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import ROOT

from common import get_device


class _ByteTrackAdapter:
    """Ultralytics' bundled ByteTrack. update(result) -> #tracks."""

    def __init__(self):
        cfg = yaml.safe_load(open(ROOT / "cfg/trackers/bytetrack.yaml"))
        cfg.setdefault("fuse_score", True)
        self.t = BYTETracker(SimpleNamespace(**cfg))

    def update(self, r):
        det = r.boxes.cpu().numpy()              # what Ultralytics feeds the tracker
        return len(self.t.update(det, r.orig_img))


class _SlowTrackAdapter:
    """SlowTrack's vendored ByteTrack-lineage tracker. update(result) -> #tracks.
    Its update() wants [N,5]=[x1,y1,x2,y2,score] plus img_info/img_size (equal
    here so no rescale) and a stage timer."""

    def __init__(self, track_thresh):
        from slowtrack_tracker import BYTETracker as STBYTETracker, Timer
        args = SimpleNamespace(track_thresh=track_thresh, track_buffer=30,
                               match_thresh=0.8, mot20=False)
        self.t = STBYTETracker(args, frame_rate=30)
        self.timer = Timer()

    def update(self, r):
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        out = np.concatenate([xyxy, conf[:, None]], axis=1).astype(float)  # [N,5]
        hw = tuple(int(v) for v in r.orig_shape)                            # (H, W)
        tracks, _total, _avg = self.t.update(out, hw, hw, self.timer)
        return len(tracks)


def _make_tracker(kind, track_thresh):
    """Fresh tracker adapter. Both expose update(result) -> #tracks."""
    if kind == "slowtrack":
        return _SlowTrackAdapter(track_thresh)
    return _ByteTrackAdapter()


def _frames(d):
    return sorted(
        p for ext in ("jpg", "jpeg", "png", "bmp")
        for p in glob.glob(os.path.join(d, f"*.{ext}"))
    )


def _warmup(yolo, path, conf, device, imgsz, tracker_kind=None, reps=3):
    """Pay one-time init costs (model warmup, lap/numpy JIT, first Kalman setup)
    BEFORE timing, so they don't contaminate the first frame. Uses throwaway
    tracker instances so the measured tracker's state stays clean."""
    for _ in range(reps):
        r = yolo.predict(path, conf=conf, device=device, imgsz=imgsz, verbose=False)[0]
        if tracker_kind:
            _make_tracker(tracker_kind, conf).update(r)


def measure_detector(yolo, folder, conf, device, imgsz):
    paths = _frames(folder)
    _warmup(yolo, paths[0], conf, device, imgsz)
    lat, counts = [], []
    for p in paths:
        t = time.perf_counter()
        r = yolo.predict(p, conf=conf, device=device, imgsz=imgsz, verbose=False)[0]
        lat.append((time.perf_counter() - t) * 1000)
        counts.append(len(r.boxes))
    n = len(paths)
    return {
        "n": n,
        "lat_ms_mean": round(sum(lat) / n, 2),
        "lat_ms_median": round(median(lat), 2),
        "det_count_mean": round(sum(counts) / n, 1),
        "det_count_max": max(counts),
    }


def measure_tracker(yolo, folder, conf, device, imgsz, tracker_kind="bytetrack"):
    """Runs the detector per frame, then times the tracker.update() call in
    ISOLATION (not detector+tracker minus detector — that buries the few-ms
    tracker signal in detector noise). Reports detector and tracker latency
    separately so the multiplier is clean. Warms up first; reports median too
    (robust to the occasional JIT/GC outlier)."""
    paths = _frames(folder)
    _warmup(yolo, paths[0], conf, device, imgsz, tracker_kind)
    tracker = _make_tracker(tracker_kind, conf)   # fresh state per sequence
    det_lat, trk_lat, ndet, ntracks = [], [], [], []
    for p in paths:
        t = time.perf_counter()
        r = yolo.predict(p, conf=conf, device=device, imgsz=imgsz, verbose=False)[0]
        det_lat.append((time.perf_counter() - t) * 1000)
        ndet.append(len(r.boxes))
        t = time.perf_counter()
        n_tracks = tracker.update(r)              # adapter normalizes both backends
        trk_lat.append((time.perf_counter() - t) * 1000)
        ntracks.append(n_tracks)
    n = len(paths)
    return {
        "n": n,
        "det_ms_median": round(median(det_lat), 2),
        "tracker_ms_median": round(median(trk_lat), 3),
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
    ap.add_argument("--tracker", choices=["bytetrack", "slowtrack"], default="bytetrack")
    args = ap.parse_args()

    device = get_device(args.device)
    yolo = YOLO(args.weights)

    print(f"== {args.mode} | device={device}"
          + (f" | tracker={args.tracker}" if args.mode == "tracker" else "") + " ==")
    for label, folder in (("CLEAN", args.clean), ("ADVERSARIAL", args.adv)):
        if args.mode == "detector":
            stats = measure_detector(yolo, folder, args.conf, device, args.imgsz)
        else:
            stats = measure_tracker(yolo, folder, args.conf, device, args.imgsz, args.tracker)
        print(f"{label:12s} {stats}")
    print("Expect: detector lat ~flat & count up  |  tracker lat up sharply on ADVERSARIAL")


if __name__ == "__main__":
    main()
