"""Measures how much the attack costs in wall-clock time — this is where the
result numbers come from. It compares a folder of clean frames against the
matching folder of attacked frames, two ways:

  * `detector` mode: time the detector and count its detections per frame. The
    count jumps under attack; the forward's compute is fixed (NMS-free), so any
    latency rise here is a CPU artifact, not a real slowdown (see README).
  * `tracker` mode: feed each frame's detections into a tracker and time just the
    tracker's update() call. THIS is the payload — the tracker's per-frame cost
    grows with the number of detections, so it slows sharply under the flood.

Run on a CPU: the tracker is CPU-bound, so the laptop is the representative
place to measure it. Two tracker backends are available via --tracker.

    # detector side (any images): count up
    python src/measure.py detector --clean data/kitti_0011_clean --adv output/kitti_0011_adv --device cpu

    # tracker side (needs a real sequence of consecutive frames):
    python src/measure.py tracker --clean data/kitti_0011_clean --adv output/kitti_0011_adv \
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


# The two trackers expect detections in different formats, so each gets a small
# adapter exposing the same simple method: update(detector_result) -> number of
# tracks. That lets the timing loop treat them interchangeably.

class _ByteTrackAdapter:
    """Ultralytics' built-in ByteTrack."""

    def __init__(self):
        cfg = yaml.safe_load(open(ROOT / "cfg/trackers/bytetrack.yaml"))
        cfg.setdefault("fuse_score", True)
        self.t = BYTETracker(SimpleNamespace(**cfg))

    def update(self, r):
        det = r.boxes.cpu().numpy()              # the boxes, in the format ByteTrack wants
        return len(self.t.update(det, r.orig_img))


class _SlowTrackAdapter:
    """SlowTrack's vendored tracker (in src/slowtrack_tracker/). Its update() has
    a different signature — it wants an [N,5] array of [x1,y1,x2,y2,score], the
    image size twice (so it does no rescaling), and a timer object."""

    def __init__(self, track_thresh):
        from slowtrack_tracker import BYTETracker as STBYTETracker, Timer
        args = SimpleNamespace(track_thresh=track_thresh, track_buffer=30,
                               match_thresh=0.8, mot20=False)
        self.t = STBYTETracker(args, frame_rate=30)
        self.timer = Timer()

    def update(self, r):
        xyxy = r.boxes.xyxy.cpu().numpy()        # box corners
        conf = r.boxes.conf.cpu().numpy()        # and their confidences
        out = np.concatenate([xyxy, conf[:, None]], axis=1).astype(float)  # -> [N,5]
        hw = tuple(int(v) for v in r.orig_shape)                            # (height, width)
        tracks, _total, _avg = self.t.update(out, hw, hw, self.timer)
        return len(tracks)


def _make_tracker(kind, track_thresh):
    """Build a fresh tracker of the requested kind (state resets per sequence)."""
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
    """Time the detector and count its detections over every frame in `folder`.
    Run it on the clean folder and the attacked folder and compare the two."""
    paths = _frames(folder)
    _warmup(yolo, paths[0], conf, device, imgsz)         # discard first-call init cost
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
    """For every frame: run the detector, then time ONLY the tracker's update()
    call. Timing the tracker alone (rather than detector+tracker and subtracting)
    matters because the tracker takes a few ms while the detector takes hundreds —
    subtracting would drown the tracker signal in detector noise. We warm up first
    and report the median, which ignores the occasional one-frame timing spike."""
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
    ap.add_argument("mode", choices=["detector", "tracker"],
                    help="what to measure: the detector, or the downstream tracker")
    ap.add_argument("--clean", required=True, help="folder of clean frames")
    ap.add_argument("--adv", required=True, help="folder of attacked frames (same filenames)")
    ap.add_argument("--weights", default="yolov10n.pt")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="model input size")
    ap.add_argument("--tracker", choices=["bytetrack", "slowtrack"], default="bytetrack",
                    help="which tracker to time (tracker mode only)")
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
