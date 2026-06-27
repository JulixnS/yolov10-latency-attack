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

## What the attack looks like

Original (left) vs attacked (right) on four KITTI clips — the perturbation is
imperceptible, but the detector jumps from a handful of detections to a flood
(boxes are the model's own predictions, confined to the scene content):

<table>
  <tr>
    <td align="center"><img src="output/compare_kitti_0011.png" width="420"><br><sub><b>seq 0011</b> — 6 → 50 detections</sub></td>
    <td align="center"><img src="output/compare_kitti_0005.png" width="420"><br><sub><b>seq 0005</b> — 6 → 62 detections</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="output/compare_kitti_0013.png" width="420"><br><sub><b>seq 0013</b> — 2 → 47 detections</sub></td>
    <td align="center"><img src="output/compare_kitti_0020.png" width="420"><br><sub><b>seq 0020</b> — 7 → 56 detections</sub></td>
  </tr>
</table>

---

## Repository layout

```
src/
  common.py          model wrapper + the dense-tensor hook, letterbox, device flag
  verify.py          STEP 0 gate — confirms tensor layout + gradient flow
  attack.py          the PGD attack (+ random-noise control)
  measure.py         latency harness: detector mode + tracker mode (--tracker)
  viz.py             original-vs-attacked comparison image (boxes drawn)
  make_videos.py     stitch a frame folder into an mp4 (detector boxes drawn)
  slowtrack_tracker/ vendored SlowTrack tracker (see its SOURCE_AND_CHANGES doc)
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

`attack.py` requires you to **point it at a folder of input frames** with
`--images`, and writes the perturbed copies to `--out`. Below it's the KITTI
subset built in the Data step, but `--images` can be any folder of `.png`/`.jpg`
frames you want to attack — it processes every image in that folder.

```bash
# the real KITTI sequence (used for every measurement below)
python src/attack.py --images data/kitti_0011_clean --out output/kitti_0011_adv \
    --device cpu --iters 40

# same epsilon value, random-noise (should not flood the detector, detections stay roughly the same)
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
the attack floods the detector `5.8 → 51.7` detections/frame (max 69) and the
**SlowTrack tracker latency goes `0.684 → 2.136 ms = ~3.1×`** (active tracks
`4.7 → 17.5`). See `output/compare_kitti_0011.png` — 6 real detections vs 50
phantom boxes, perturbation invisible, all in the scene content (grey letterbox
padding is left clean).

**Threat model — content-region only.** The perturbation and the loss are
masked to the **scene-content region**, excluding the grey letterbox padding the
preprocessing pipeline inserts. A real camera attacker controls the scene, not
that padding, so flooding it would be cheating. (Without the mask the same setup
reads `5.8 → 94.6` detections / `5.2×`; ~45% of that flood was un-realizable
padding detections — the content-only numbers below are the honest result.)

**Tracker latency multiplier across four KITTI clips** (30 frames each,
SlowTrack, CPU, median):

| KITTI seq | clean detects | adv detects | tracker clean ms | tracker adv ms | multiplier |
|---|---|---|---|---|---|
| 0013 | 2.4 | 46.9 | 0.517 | 1.875 | **3.6×** |
| 0005 | 3.4 | 53.5 | 0.543 | 2.12  | **3.9×** |
| 0011 | 5.8 | 51.7 | 0.684 | 2.136 | **3.1×** |
| 0020 | 6.7 | 41.5 | 0.684 | 1.74  | **2.5×** |

The attack holds across all four clips (**2.5–3.9×**). The multiplier is *larger
when the scene starts with fewer real objects*: the flood saturates at a similar
absolute level (~42–53 detections) regardless, so a clip with a low clean
detection count — and thus a tiny clean tracker baseline — sees a bigger relative
jump, while a busier clip already keeps the tracker working and has less headroom.

The primary downstream consumer is **SlowTrack** (`--tracker slowtrack`);
`--tracker bytetrack` remains available.

**On the detector latency — an honest nuance.** The detector's forward is
fixed-FLOP (verified: clean vs adversarial forward ≈ 193 vs 196 ms in
isolation), so algorithmically it can't be slowed — that's the NMS-free thesis.
But end-to-end CPU `predict()` *does* show inference rising on the flood
(~106 → 229 ms here). That extra cost is a **denormal-float artifact**: the
perturbation drives subnormal values into the fused Conv+BN path, which is slow
on CPU. In a controlled same-frame test the denormal-attributable slowdown is
≈1.4×, and `torch.set_flush_denormal(True)` cuts it to ≈1.1×; GPUs flush
denormals by default — so this is a CPU-test-rig effect, not a real detector
latency attack. The robust, deployment-relevant payload is the **tracker**.

## SlowTrack tracker

`src/slowtrack_tracker/` is SlowTrack's ByteTrack-lineage MOT, vendored from
https://github.com/ershang2/SlowTrack with only mechanical patches (relative
imports, `np.float`→`float`, debug prints removed). Full provenance, per-file
change log, and the upstream MIT license are in
`src/slowtrack_tracker/Slowtrack_source_and_changes.txt`.

---
