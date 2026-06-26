# Q&A log — building the YOLOv10 NMS-free latency attack

A record of the questions asked while designing, building, and explaining this
project, with condensed answers.

---

### 1. How do I plan a "confidence threshold attack" on a YOLO model?
After clarifying, the goal was an **overload/sponge latency attack**: keep many
predictions *just above* the confidence threshold so they all survive and flood
the downstream consumers (tracker, fusion, planner). Threat model: **white-box**
(gradient access). This is the Phantom Sponge / Overload family, related to the
SlowTrack line of work.

### 2. Isn't YOLOv11 NMS-free (one-to-one matching)?
No — that's **YOLOv10**. YOLOv10 uses consistent dual assignments with a
one-to-one inference head, so it needs no NMS. **YOLOv11** is anchor-free but
still uses one-to-many assignment + NMS by default. The distinction matters: it
decides whether the detector-side NMS latency lever even exists.

### 3. Re-plan around YOLOv10 (NMS-free).
Thesis becomes: *removing NMS closes the detector-side latency surface but not
the system-level one.* The detector's forward is fixed compute (can't be slowed);
the attack must defeat the head's **learned** suppression and push all damage
downstream into the tracker. Drop the NMS-IoU loss term; keep a flood term plus a
spread (diversity) term motivated by tracker cost.

### 4. How do I implement the attack — do I edit the YOLOv10 code?
No source edits. It's an external PyTorch script: call the model's raw forward
(gradients on), build a differentiable loss on the dense pre-top-k confidences,
and run PGD on the input. The only internal need — the dense confidence tensor —
is obtained via a forward hook, not by editing files. Craft with the raw forward;
*measure* with the real `predict()` so numbers are honest.

### 5–7. How is this code the attack / how does the attack work?
The attack is **gradient ascent on the image instead of the weights**. The model
is frozen (the victim); the loss encodes the goal ("many diverse detections");
autograd turns that goal into a per-pixel nudge; the clamped perturbation is the
weapon. End to end: a tiny perturbation makes the detector emit ~max_det phantom
detections every frame at the *same* detector latency, which overloads the
tracker (association cost scales with detection count), cascading into missed
real-time deadlines.

### 8. Can I test on a laptop CPU without a GPU (nano model)?
Yes. Crafting (PGD) is the only CPU-heavy part (~10–30 s/image at 50 iters); fine
for dev and small runs, slow at full scale. Measurement — especially the tracker
— is CPU-bound anyway, so the laptop is the *right* place for the headline result.
Recommended split: develop on laptop, bulk-craft on a GPU, measure back on the CPU.

### 9. Scaffold CPU-first with a device flag; where do I get images?
Scaffolded `src/` (common, verify, attack, measure, viz). Image sources: any
images for dev/detector-side (COCO val, coco128, sample JPEGs); **consecutive
frames** for the tracker (KITTI tracking, MOT17/20, or ffmpeg from a video) —
random images won't exercise the tracker.

### 10. What is the tracker for, why install one?
The tracker is the **victim**. The detector can't be slowed, so without a
downstream consumer there's no latency damage to measure. A multi-object tracker
links detections across frames (Kalman predict + Hungarian association, ≈O(N³)),
and its per-frame cost explodes with detection count. No separate install needed —
Ultralytics bundles ByteTrack; swap in SlowTrack later.

### 11–12. Create a CPU runbook (PLAN.md) and add viz.py.
Authored `PLAN.md` — a 6-phase plan ending in the original-vs-attacked comparison
image — and added `src/viz.py` to render it.

### 13. Execute the plan to get the comparison photos.
Ran it end to end on CPU. The Phase-1 gate caught a real version difference
(8.4.x returns post-top-k `[1,300,6]` and builds one2one from detached features),
fixed via the forward-hook recompute. Results: bus 5→162, zidane 2→108
above-threshold; noise control 5→5/2→3 (no flood); detector latency flat
(263→222 ms), count 3.5→109. Produced `out/compare_*.png`.

### 14. Set up the tracker-sequence run.
Initially built a synthetic panning sequence (`make_seq.py`, since removed) and
rewrote `measure.py` to time `BYTETracker.update()` in isolation. Result: detector
flat (244→230 ms), detections 5.1→74, **tracker latency 0.556→3.133 ms = 5.6×
multiplier**, tracks 5→21. Later replaced the synthetic sequence with real KITTI
Tracking clips (`make_seq.py` removed).

### 15. What is seq_clean?
The clean (un-attacked) 16-frame sequence (a pan over `bus.jpg`), used both as the
tracker baseline and as the input the attack perturbs into `out/seq_adv/`.

### 16–17. Explain the whole pipeline / explain it to a newcomer.
Two phases: **craft** (raw forward + gradients → adversarial images) and
**measure/view** (real model → latency, counts, comparison image). The four
numbers tell one story: detector flat, count up, noise control fails, tracker up.

### 18. Why multiple iterations?
One gradient step is only locally accurate; PGD takes many small re-aimed steps
within the ε budget. It also *recruits* anchors: only near-threshold anchors have
gradient, so each step pulls a fresh batch over the line (20 iters → 5→64; 150 →
5→162). Iterations search better within a fixed budget; they don't make δ bigger.

### 19. The math, in simple terms.
Maximize a smooth count of detections over a bounded perturbation:
`minimize_δ −Σ σ(α(cᵢ−τ))  s.t. ‖δ‖∞ ≤ ε`, solved by PGD (gradient → sign step →
clip back into the ε-ball). The sigmoid relaxes the non-differentiable
"count above threshold" into something with a usable gradient.

### 20. What are w and α?
- **wᵢ** — per-anchor soft "is this a detection?" in (0,1): `σ(α(cᵢ−τ))`. Summed,
  it approximates the detection count, differentiably.
- **α** — a single scalar (50) setting how sharply w flips from 0→1 across the
  threshold; it sets the width of the near-threshold band that gets gradient.

### 21. Difference between c and w?
**cᵢ** = the model's raw confidence (a measurement, no notion of the threshold).
**wᵢ** = a value we compute from cᵢ relative to τ ("does it cross the line?").
Summing w (not c) makes "maximize the sum" mean "count detections" rather than
"make a few anchors louder."

### 22. Does the model output a tensor of cᵢ?
Not directly. It produces a dense `[1, 84, 8400]` tensor (4 box coords + 80 class
scores per anchor); cᵢ = max over the 80 class channels → `[1, 8400]`. And the
deployed forward hides this behind top-k, returning only `[1, 300, 6]` — which is
why the dense-extraction hook exists.

### 23. How does the code use this math to optimize δ?
Correction: it doesn't *maximize* δ — δ is held minimal by `clamp(-eps, eps)`. It
maximizes the detection count *over* δ: forward → loss, backprop → per-pixel
gradient, sign step in the count-increasing direction, project back into the budget,
repeat.

### 24. What is flood, and why do we need it?
`flood = w.sum()` — a single scalar ≈ the detection count. It's the attack's goal
as one differentiable number (the optimizer needs one scalar). Summing w (not max,
not c) rewards getting *many* anchors over the threshold — the overload payload.

### 25. How is the loss calculated, and how does backprop affect pixels?
Forward chain: image → conf (cᵢ) → w → flood (+ spread) → `loss = −(flood+λ·spread)`.
Backprop applies the chain rule from loss to every pixel (frozen weights still pass
gradient through to the input); the sigmoid slope `α·wᵢ·(1−wᵢ)` routes "blame" to
pixels affecting near-threshold anchors. The sign step then nudges each blamed
pixel a tiny amount in the detection-increasing direction, clipped to stay invisible.
