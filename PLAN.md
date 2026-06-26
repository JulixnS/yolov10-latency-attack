# Implementation plan — NMS-free YOLOv10 latency attack on CPU

Goal: get the attack running end-to-end on a laptop CPU against real KITTI
Tracking footage, ending in the visual deliverable — the original frame with a
handful of boxes side-by-side with the attacked frame flooded with phantom
boxes, plus the SlowTrack latency multiplier.

Each phase has a concrete command and an **acceptance check** — don't move on
until it passes. All referenced `src/*.py` files already exist.

---

## Phase 0 — Environment + data

```bash
cd /home/julia/LLM-Defense/yolov10-latency-attack
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU torch + lap + cython_bbox

# KITTI Tracking (one-time ~14.7 GB download; kitti/ is gitignored)
mkdir -p kitti && (cd kitti && \
  curl -L -C - -o data_tracking_image_2.zip \
    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip && \
  unzip -q data_tracking_image_2.zip 'training/image_02/0011/*')

# 30-frame clean subset of consecutive frames (CPU-tractable)
mkdir -p data/kitti_0011_clean
ls kitti/training/image_02/0011/*.png | sort | head -30 \
  | xargs -I{} cp {} data/kitti_0011_clean/
```

**Acceptance:** `python -c "import torch, ultralytics; print('ok')"` prints
`ok`, and `data/kitti_0011_clean/` holds 30 PNGs. The first `yolov10n.pt`
reference auto-downloads the weights (~5 MB).

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
forward returns a post-top-k tensor on your version. `common.py`'s forward-hook
path on the one2one head is what recovers it (already handled for ultralytics
8.4.x). Stop here until this passes.

---

## Phase 2 — Craft the attack (feeds every later phase)

```bash
python src/attack.py --images data/kitti_0011_clean --out output/kitti_0011_adv \
    --device cpu --iters 40
```
~10–12 min on a laptop CPU (30 frames × 40 iters).

**Acceptance:** per-frame log shows `above-thresh N -> M` with **M ≫ N**
(e.g. `4 -> ~100`). `output/kitti_0011_adv/` holds the perturbed frames, which
look near-identical to the originals by eye (the eps budget working).

**If M doesn't rise:** raise `--iters`, check `--tau` matches the conf
threshold, or bump `--eps` toward `16/255` temporarily to confirm the
mechanism, then dial back.

---

## Phase 3 — Noise control (load-bearing, not optional)

```bash
python src/attack.py --images data/kitti_0011_clean --out output/kitti_0011_noise \
    --device cpu --mode noise
```

**Acceptance:** `above-thresh N -> ~N` (noise at the **same eps** does NOT
flood). This contrast is the proof the flood is adversarial, not just "any
perturbation." Keep these numbers.

---

## Phase 4 — Detector-side measurement

```bash
python src/measure.py detector --clean data/kitti_0011_clean \
    --adv output/kitti_0011_adv --device cpu
```

**Acceptance:** `det_count_mean` jumps toward `max_det` (clean ~6 → adv ~95).
The detector's *forward* is fixed-FLOP; note that end-to-end CPU `predict()`
latency may rise ~1.4× — that is a denormal-float artifact (see `results.md`),
not a real detector-side attack.

---

## Phase 5 — Tracker-side measurement (SlowTrack, the headline)

```bash
python src/measure.py tracker --clean data/kitti_0011_clean \
    --adv output/kitti_0011_adv --device cpu --tracker slowtrack
```

`measure.py tracker` runs the detector per frame, then times the SlowTrack
`update()` call **in isolation** (warmed up, median reported — don't subtract
detector time; CPU detector noise dwarfs the sub-ms tracker signal).

**Acceptance:** `tracker_ms_median` is **much larger** on ADVERSARIAL
(≈ 0.7 → 3.7 ms, ~5×) while detection count is up ~16×. That's the payload:
the flood overloads the downstream tracker.

---

## Phase 6 — See what the model sees (the deliverable)

```bash
python src/viz.py --clean data/kitti_0011_clean/000010.png \
    --adv output/kitti_0011_adv/000010.png \
    --device cpu --out output/compare_kitti_0011.png
```

**Acceptance / done:** `output/compare_kitti_0011.png` shows the **original
with a few boxes** on the left and the **attacked frame saturated with boxes**
on the right (perturbation invisible to the eye), with the `N -> M` counts in
the banners. That is the model's-eye view of the attack.

---

## Summary checklist
- [ ] P0 env + KITTI 30-frame subset
- [ ] P1 verify.py prints OK (dense shape, nonzero gradient)
- [ ] P2 attack floods (`N -> M`, M≫N)
- [ ] P3 noise control does NOT flood
- [ ] P4 detector count up
- [ ] P5 SlowTrack tracker latency up ~5×
- [ ] P6 `output/compare_kitti_0011.png`: original vs attacked, boxes drawn ✅
