# YOLOv10 NMS-free latency (sponge / overload) attack

A white-box PGD attack that adds an **invisible perturbation** to a camera
frame, making an **NMS-free YOLOv10** detector emit a flood of phantom
detections. The detector's own latency stays flat (its forward pass is fixed
compute), but the flood overloads the **downstream tracker** — so the attack
demonstrates that *removing NMS closes the detector-side latency surface but
relocates the damage one layer downstream.*

- **Target:** YOLOv10 (one-to-one head, no NMS). The attack defeats the head's
  *learned* duplicate suppression instead of exploiting an NMS stage.
- **Payload:** a multi-object tracker (SlowTrack / ByteTrack), whose per-frame
  association cost scales super-linearly with the number of detections.
- **Runs on a laptop CPU** — no GPU required.

---

## Thesis, in four numbers

A correct run shows this chain (detector flat, output flooded, noise control
fails, tracker latency explodes):

1. detector latency: ~unchanged (fixed compute)
2. detections/frame: up ~15–30×
3. same-budget random noise: does **not** flood (the control)
4. tracker latency: up several × while the detector stays flat

---

## Repository layout

```
src/
  common.py          model wrapper + the dense-tensor hook, letterbox, device flag
  verify.py          STEP 0 gate — confirms tensor layout + gradient flow
  attack.py          the PGD attack (+ random-noise control)
  measure.py         latency harness: detector mode + tracker mode (--tracker)
  viz.py             original-vs-attacked comparison image (boxes drawn)
  slowtrack_tracker/ vendored SlowTrack tracker (see its SOURCE_AND_CHANGES doc)
PLAN.md              phase-by-phase runbook with acceptance checks
results.md           KITTI results + comparison image
requirements.txt     ultralytics + slowtrack tracker deps
```

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

This pulls a CPU build of torch automatically, plus `lap` and `cython_bbox`
(needed by the vendored SlowTrack tracker; `cython_bbox` compiles at install
time and builds on x86_64 and aarch64).

---

## Data

A real autonomous-driving **sequence** is the only input needed — the tracker
measurement requires consecutive frames, and individual frames from it double
as the still-image inputs for the detector-side and visualization steps. We use
the **KITTI Tracking** benchmark:
```bash
mkdir -p kitti && cd kitti
curl -L -C - -o data_tracking_image_2.zip \
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip   # 14.7 GB
unzip -q data_tracking_image_2.zip 'training/image_02/0011/*'                 # one busy seq
cd ..
# build a CPU-tractable 30-frame clean subset of consecutive frames
mkdir -p data/kitti_0011_clean
ls kitti/training/image_02/0011/*.png | sort | head -30 \
  | xargs -I{} cp {} data/kitti_0011_clean/
```
(`kitti/` is gitignored — the dataset is never committed.) Any other KITTI
sequence, MOT17/20, or `ffmpeg`-extracted dashcam clip works the same way —
the pipeline just consumes a folder of consecutive frames.

---

## Reproduce the results

### Step 0 — verify the model wiring (mandatory gate)
```bash
python src/verify.py --device cpu
```
Pass = `conf` shape `[1, 8400]`, ~0 anchors above threshold on random input,
and a nonzero input-gradient (`OK`). If it instead reports a post-top-k
`[1,300,6]` shape, your Ultralytics version hides the dense tensor and
`common.py`'s forward-hook path is what recovers it (already handled for
ultralytics 8.4.x).

### Step 1 — craft the attack + the noise control
```bash
# the real KITTI sequence (used for every measurement below)
python src/attack.py --images data/kitti_0011_clean --out output/kitti_0011_adv \
    --device cpu --iters 40

# same-budget random-noise control (must NOT flood)
python src/attack.py --images data/kitti_0011_clean --out output/kitti_0011_noise \
    --device cpu --mode noise
```
Each line logs `above-thresh N -> M`. The attack should give `M ≫ N`; the
noise control should give `M ≈ N`.

### Step 2 — measure the detector (count up; latency caveat below)
```bash
python src/measure.py detector --clean data/kitti_0011_clean \
    --adv output/kitti_0011_adv --device cpu
```

### Step 3 — measure the tracker on the real sequence (SlowTrack)
```bash
python src/measure.py tracker \
    --clean data/kitti_0011_clean --adv output/kitti_0011_adv \
    --device cpu --tracker slowtrack
```
This runs the detector per frame, then times the SlowTrack `update()` call **in
isolation** (reporting `det_ms_mean` and `tracker_ms_mean` separately — don't
subtract them; the CPU detector's noise dwarfs the sub-ms tracker signal).

### Step 4 — see what the model sees
```bash
python src/viz.py --clean data/kitti_0011_clean/000010.png \
    --adv output/kitti_0011_adv/000010.png \
    --device cpu --out output/compare_kitti_0011.png
```
Produces a side-by-side image: original (a few boxes) vs attacked (saturated
with phantom boxes), perturbation invisible, with detection-count banners.

---

## Results

**Headline (KITTI Tracking seq 0011, 30 real frames, CPU, warmed, median):**
the attack floods the detector `5.8 → 94.6` detections/frame (max 123) and the
**SlowTrack tracker latency goes `0.709 → 3.663 ms = 5.2×`** (active tracks
`4.7 → 27.4`). See `output/compare_kitti_0011.png` — 6 real detections vs 72
phantom boxes, perturbation invisible.

**Tracker latency multiplier** (the headline payload):

| sequence                    | tracker   | clean ms | adv ms | multiplier |
|-----------------------------|-----------|----------|--------|------------|
| **KITTI seq 0011 (real)**   | slowtrack | 0.709    | 3.663  | **5.2×**   |
| synthetic 16-frame (legacy) | slowtrack | 0.685    | 2.814  | 4.1×       |
| synthetic 16-frame (legacy) | bytetrack | 0.556    | 3.133  | 5.6×       |

The primary downstream consumer is **SlowTrack** (`--tracker slowtrack`);
`--tracker bytetrack` remains available.

**On the detector latency — an honest nuance.** The detector's forward is
fixed-FLOP (verified: clean vs adversarial forward ≈ 193 vs 196 ms in
isolation), so algorithmically it can't be slowed — that's the NMS-free thesis.
But end-to-end CPU `predict()` *does* show inference `~113 → 250 ms (≈1.4×)` on
the flood. That extra cost is a **denormal-float artifact**: the perturbation
drives subnormal values into the fused Conv+BN path, which is slow on CPU.
`torch.set_flush_denormal(True)` cuts it from 1.42× to 1.11×, and GPUs flush
denormals by default — so this is a CPU-test-rig effect, not a real detector
latency attack. The robust, deployment-relevant payload is the **tracker**.

---

## Two pitfalls (see code comments)

1. **Craft in the model's 640×640 letterboxed [0,1] space**, not on the raw
   JPEG — re-letterboxing a perturbed JPEG resamples and washes out delta.
   `common.image_to_tensor` keeps this consistent.
2. **`--tau` must equal the deployed `conf` threshold** (default 0.25) — the
   attack parks detections just above that line.

## Honest bounds

- `max_det` (default 300) caps the per-frame flood; the attack pins the
  pipeline at that worst case *every frame*, it does not grow unbounded.
- On a CPU laptop the detector (~hundreds of ms) dwarfs the tracker (sub-ms).
  The tracker **multiplier** is the result — in a real deployment the detector
  runs on GPU (~ms) and the tracker is the bottleneck, which is the regime the
  multiplier represents.
- The random-noise control must FAIL to flood — that contrast is the proof the
  NMS-free head's learned suppression is robust to noise but not to adversarial
  input.

## Vendored SlowTrack tracker

`src/slowtrack_tracker/` is SlowTrack's ByteTrack-lineage MOT, vendored from
https://github.com/ershang2/SlowTrack with only mechanical patches (relative
imports, `np.float`→`float`, debug prints removed). Full provenance, per-file
change log, and the upstream MIT license are in
`src/slowtrack_tracker/Slowtrack_source_and_changes.txt`.

---

## What changed during development (summary)

- Re-scoped from YOLOv11 (uses NMS) to **YOLOv10** (NMS-free) — the attack must
  defeat the head's *learned* suppression and push damage downstream.
- Added a **forward-hook dense-tensor extractor** in `common.py` because
  ultralytics 8.4.x returns a post-top-k tensor and builds its one2one branch
  from detached features (neither is differentiable to the input).
- Built the **isolated-timing** tracker harness (don't subtract detector time).
- Added a **`--tracker {bytetrack,slowtrack}`** flag and vendored the real
  **SlowTrack** tracker as the downstream consumer.
- Settled on **real KITTI Tracking** frames as the evaluation sequence.
