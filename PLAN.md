# Implementation plan — NMS-free YOLOv10 latency attack on CPU

Goal: get the attack running end-to-end on a laptop CPU and **see what the
model sees** — the original image with its handful of boxes, side-by-side
with the attacked image flooded with phantom boxes.

Each phase has a concrete command and an **acceptance check** — don't move
on until it passes. All files referenced (`src/*.py`, including
`viz.py`) already exist.

---

## Phase 0 — Environment (10 min)

```bash
cd /home/julia/LLM-Defense/yolov10-latency-attack
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU torch comes in automatically
mkdir -p data/dev out
curl -L https://ultralytics.com/images/bus.jpg -o data/dev/bus.jpg
curl -L https://ultralytics.com/images/zidane.jpg -o data/dev/zidane.jpg
```

**Acceptance:** `python -c "import torch, ultralytics; print('ok')"` prints `ok`,
and `data/dev/` holds 2 images. The first `yolov10n.pt` reference auto-downloads
the weights (~5 MB).

---

## Phase 1 — Confirm model internals (the gate)

The attack depends on reading the **dense one2one confidence tensor** (pre
top-k) and getting gradients back to the input. Ultralytics versions differ,
so verify before trusting anything.

```bash
python src/verify.py --device cpu
```

**Acceptance:**
- `conf` shape is `[1, A]` with **A in the low thousands** (~8400 at 640px).
- "# anchors above 0.25 on random input" is ~0 (the learned suppression we attack).
- `input-gradient norm` is **> 0** and prints `OK`.

**If it fails** (shape looks like `[1, 300, 6]`, or gradient is zero): the
forward returns a post-top-k tensor on your version. Switch `common.dense()`
to a forward hook on the one2one head — capture the script's printed return
structure and adapt. Stop here until this passes.

---

## Phase 2 — First attack on one image

Run a short attack (low iters for fast CPU iteration) on the dev images.

```bash
python src/attack.py --images data/dev --out out/adv --device cpu --iters 20
```

**Acceptance:** per-image log shows `above-thresh N -> M` with **M ≫ N**
(e.g. `8 -> 300`). `out/adv/` contains the perturbed images, which look
near-identical to the originals by eye (that's the eps budget working).

**If M doesn't rise:** raise `--iters` (try 50), check `--tau` matches the
conf threshold, or increase `--eps` toward `16/255` temporarily to confirm
the mechanism, then dial back.

---

## Phase 3 — Noise control (load-bearing, not optional)

```bash
python src/attack.py --images data/dev --out out/noise --device cpu --mode noise
```

**Acceptance:** `above-thresh N -> ~N` (noise at the **same eps** does NOT
flood). This contrast is the proof that the flood is adversarial, not just
"any perturbation." Keep these numbers.

---

## Phase 4 — Detector-side measurement (flat latency, count up)

```bash
python src/measure.py detector --clean data/dev --adv out/adv --device cpu
```

**Acceptance:** `lat_ms_mean` is **~the same** for CLEAN vs ADVERSARIAL
(detector is fixed compute), while `det_count_mean` jumps toward `max_det`.
This is half the thesis: detector unmoved, output saturated.

---

## Phase 5 — tracker-side measurement

Needs **consecutive frames**, not the 2 dev images. `make_seq.py` builds a
local sequence by panning a crop across one image (stand-in for footage);
swap in real frames later (KITTI tracking dir, or
`ffmpeg -i clip.mp4 data/seq_clean/%06d.jpg`).

```bash
python src/make_seq.py --image data/dev/bus.jpg --out data/seq_clean --frames 16
python src/attack.py   --images data/seq_clean --out out/seq_adv --device cpu --iters 40
python src/measure.py  tracker --clean data/seq_clean --adv out/seq_adv --device cpu
```

`measure.py tracker` runs the detector per frame, then times the
`tracker.update()` call **in isolation** (reports `det_ms_mean` and
`tracker_ms_mean` separately — don't subtract; the detector noise on CPU
dwarfs the tracker signal).

**Acceptance:** `tracker_ms_mean` is **much larger** on ADVERSARIAL while
`det_ms_mean` stays flat. That's the other half of the thesis.

**Caveat (important for the writeup):** on this CPU laptop the detector
(~240 ms) dwarfs the tracker (sub-ms clean). The tracker *multiplier* is the
real result, because in a real deployment the detector runs on GPU (~5 ms)
and the tracker becomes the bottleneck — that's the scenario the multiplier
represents.

---

## Phase 6 — See what the model sees (the deliverable)

Create `src/viz.py` to render **original vs attacked, both with the model's
boxes drawn**, side by side, annotated with detection counts. Use Ultralytics'
`results.plot()` (returns an annotated BGR array) and `cv2.hconcat`.

```python
# src/viz.py
import argparse, os, cv2, numpy as np
from ultralytics import YOLO
from common import get_device

def annotated(yolo, path, conf, device, imgsz):
    r = yolo.predict(path, conf=conf, device=device, imgsz=imgsz, verbose=False)[0]
    img = r.plot()                      # BGR with boxes drawn
    return img, len(r.boxes)

def banner(img, text, h=34):
    bar = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return cv2.vconcat([bar, img])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)      # e.g. data/dev/bus.jpg
    ap.add_argument("--adv", required=True)        # e.g. out/adv/bus.jpg
    ap.add_argument("--out", default="out/compare.png")
    ap.add_argument("--weights", default="yolov10n.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    yolo = YOLO(a.weights); dev = get_device(a.device)
    ci, cn = annotated(yolo, a.clean, a.conf, dev, a.imgsz)
    ai, an = annotated(yolo, a.adv,   a.conf, dev, a.imgsz)
    h = max(ci.shape[0], ai.shape[0])               # match heights before hconcat
    ci = cv2.resize(ci, (ci.shape[1], h)); ai = cv2.resize(ai, (ai.shape[1], h))
    left  = banner(ci, f"ORIGINAL: {cn} detections")
    right = banner(ai, f"ATTACKED: {an} detections")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(a.out, cv2.hconcat([left, right]))
    print(f"wrote {a.out}  ({cn} -> {an} detections)")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python src/viz.py --clean data/dev/bus.jpg --adv out/adv/bus.jpg --device cpu
```

**Acceptance / done:** `out/compare.png` shows the **original with a few
boxes** on the left and the **attacked image saturated with boxes** on the
right (perturbation invisible to the eye, boxes everywhere), with the
`N -> M` counts in the banners. That is the model's-eye view of the attack.

---

## Summary checklist
- [ ] P0 env + dev images
- [ ] P1 verify.py prints OK (dense shape, nonzero gradient)
- [ ] P2 attack floods (`N -> M`, M≫N)
- [ ] P3 noise control does NOT flood
- [ ] P4 detector latency flat, count up
- [ ] P5 (later) tracker latency explodes on a sequence
- [ ] P6 `out/compare.png`: original vs attacked, boxes drawn ✅
```