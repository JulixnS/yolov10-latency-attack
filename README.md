# YOLOv10 NMS-free latency (sponge/overload) attack

White-box PGD attack that floods an NMS-free YOLOv10 detector with diverse
phantom detections to overload the downstream tracker. The detector itself
stays flat (fixed compute); the latency damage lands one layer downstream.

CPU-first — runs on a laptop with no GPU. Flip `--device cuda`/`auto` for
bulk crafting on the cluster, then bring the adversarial frames back to the
laptop to measure (the tracker is CPU-bound, so the laptop is the right
place for the headline result).

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # pulls CPU torch automatically
```

## Run, in order
```bash
# 0. CONFIRM the model tensor layout + gradient flow on your Ultralytics version
python src/verify.py --device cpu

# 1. Craft the attack and the noise control (dev: small folder, fewer iters)
python src/attack.py --images data/dev --out out/adv   --device cpu --iters 20
python src/attack.py --images data/dev --out out/noise --device cpu --mode noise

# 2. Measure: detector flat + count up; tracker latency explodes
python src/measure.py detector --clean data/dev --adv out/adv   --device cpu
python src/measure.py tracker  --clean data/seq_clean --adv out/seq_adv --device cpu
```

## Where to get images

**Dev / detector-side flooding** — any images work:
- Zero setup: `data/dev` with a few JPEGs. Quick grab:
  `curl -L https://ultralytics.com/images/bus.jpg -o data/dev/bus.jpg`
- 128-image set: `curl -L https://ultralytics.com/assets/coco128.zip -o coco128.zip && unzip`
- Full COCO val2017 (5k imgs, ~1GB): `http://images.cocodataset.org/zips/val2017.zip`

**Tracker-side measurement — needs CONSECUTIVE frames, not random images.**
The tracker payload only shows up on a temporally coherent sequence:
- KITTI tracking sequences (you already have KITTI staged) — ideal AV framing.
- MOT17 / MOT20: https://motchallenge.net
- Any dashcam clip: `ffmpeg -i clip.mp4 data/seq_clean/%06d.jpg`
Then run `attack.py` over `data/seq_clean` to produce `out/seq_adv`, and
measure both with `measure.py tracker`.

## Two pitfalls (see code comments)
1. **Craft in the model's 640x640 letterboxed [0,1] space**, not on the raw
   JPEG — re-letterboxing a perturbed JPEG resamples and washes out delta.
   `common.image_to_tensor` keeps this consistent.
2. **`--tau` must equal the deployed `conf` threshold** (default 0.25).

## Tracker backends
`measure.py tracker` supports two downstream consumers via `--tracker`:
- `bytetrack` (default) — Ultralytics' bundled ByteTrack.
- `slowtrack` — SlowTrack's ByteTrack-lineage MOT, vendored in
  `src/slowtrack_tracker/` from https://github.com/ershang2/SlowTrack
  (patched: relative imports, `np.float`→`float`, debug prints removed).

The harness times `tracker.update()` in isolation and reports detector vs
tracker latency separately. Both backends show the attack overloading the
tracker while the detector stays flat (CPU, 16-frame sequence):

| tracker   | clean ms | adv ms | multiplier | tracks clean→adv |
|-----------|----------|--------|------------|------------------|
| bytetrack | 0.556    | 3.133  | 5.6×       | 5 → 21           |
| slowtrack | 0.685    | 2.814  | 4.1×       | 5 → 19           |

```bash
python src/measure.py tracker --clean data/seq_clean --adv out/seq_adv \
    --device cpu --tracker slowtrack
```

## Honest bounds
- `max_det` (default 300) caps the per-frame flood; the attack pins the
  pipeline at that worst case *every frame*, it does not grow unbounded.
- The random-noise control must FAIL to flood — that contrast is the proof
  the learned suppression is robust to noise but not to adversarial input.
