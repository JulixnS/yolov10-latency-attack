"""Shared helpers used by every other script: pick the device, turn an image
into the exact tensor the model expects, and wrap the YOLOv10 model so the
attack can read its raw, per-location confidence scores.

Background for a first read:
  * The detector looks at ~8400 fixed locations ("anchors") in the image and,
    for each, outputs how confident it is that an object is there. Normally the
    model keeps only its top few hundred best guesses ("top-k") and hands those
    back — the rest are thrown away.
  * The attack needs ALL ~8400 raw confidences, and it needs them in a form
    PyTorch can take gradients of (so it can work out how to nudge pixels). That
    raw, full, gradient-trackable tensor is what `YoloV10Dense.dense()` returns.
  * Ultralytics' normal `.predict()` won't give us that — it discards the raw
    scores and turns off gradient tracking. So we reach inside the model with a
    hook (explained on the class below). `verify.py` checks this still works on
    your Ultralytics version before you trust any results.
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def get_device(flag: str = "cpu") -> str:
    """`--device` flag: 'cpu', 'cuda', or 'auto'."""
    if flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return flag


def letterbox(im: np.ndarray, new: int = 640, color: int = 114):
    """Fit a wide/tall image into a square `new`x`new` box without distorting it:
    scale it down to fit, then pad the leftover space with flat grey (value 114).
    This is exactly what YOLO does internally, and it's how a 1242x375 KITTI frame
    becomes a 640x640 input with grey bars on top and bottom.

    Returns (padded_bgr, ratio, (dw, dh)): the padded image, the scale factor used,
    and the (width, height) padding added on each side. The attack must run in this
    same padded space — perturbing the original then re-letterboxing would resample
    the image and wash out the tiny perturbation (README pitfall #1).
    """
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new - nw) / 2, (new - nh) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(color,) * 3)
    return padded, r, (dw, dh)


def image_to_tensor(path: str, imgsz: int, device: str):
    """Load an image from disk -> letterboxed [1,3,imgsz,imgsz] float tensor in [0,1], RGB.

    Also returns the scene-content rectangle (x0, y0, x1, y1) in imgsz coords —
    the region holding the real (resized) image, excluding the grey letterbox
    padding. The attack masks the perturbation and the loss to this region so it
    can't cheat by flooding the padding (which a real camera attacker can't touch).
    """
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    padded, r, (dw, dh) = letterbox(bgr, imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device).float().div(255.0)
    t = t.permute(2, 0, 1).unsqueeze(0).contiguous()       # [1,3,H,W]
    h, w = bgr.shape[:2]
    nh, nw = round(h * r), round(w * r)                     # resized content size
    x0, y0 = round(dw - 0.1), round(dh - 0.1)              # matches letterbox padding
    content = (x0, y0, x0 + nw, y0 + nh)
    return t, content


class YoloV10Dense:
    """Wraps a YOLOv10 model so the attack can read its raw per-anchor scores
    with gradients attached. `dense(image)` returns a box and a confidence for
    every one of the ~8400 anchors — the numbers the attack tries to push over
    the detection threshold.

    The "hook" trick (the awkward but necessary part):
      YOLOv10 has two output heads. Only the "one2one" head is used at inference
      (it's the one that lets the model skip NMS). But Ultralytics 8.4.x makes
      that head hard to attack two ways at once:
        1. The public forward only returns the final ~300 kept boxes, not the
           raw 8400 — the rest are discarded before we ever see them.
        2. Internally it computes the one2one head from a *detached* copy of the
           features, which severs the gradient path back to the input pixels.
      So we register a "forward pre-hook": a callback that fires just before the
      detection head runs and snapshots its *input* feature maps (which still
      carry gradients). We then re-run the one2one head ourselves on those
      features. Same numbers the deployed model produces, but now we have all
      8400 of them AND a gradient path to the pixels.
    """

    def __init__(self, weights: str = "yolov10n.pt", device: str = "cpu"):
        self.yolo = YOLO(weights)          # high-level API, used later for measurement
        self.net = self.yolo.model.to(device).eval()
        # Freeze the model: the attack optimizes the input image, never the weights.
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.head = self.net.model[-1]     # the detection head (last layer)
        if not getattr(self.head, "end2end", False):
            raise RuntimeError("Head is not end2end — is this really a YOLOv10 model?")
        self._feats = None
        # Install the callback that snapshots the head's input features each forward.
        self.head.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        # Fires right before the head runs. args[0] is the list of feature maps
        # (P3/P4/P5, the three resolution levels) feeding the head — captured here
        # while they still have a gradient path back to the input image.
        self._feats = list(args[0])

    def dense(self, x: torch.Tensor):
        """Run the model on image `x` ([B,3,H,W], pixel values in [0,1]) and
        return the raw, full prediction set:
            boxes: [B, 4, A]  — a box (x1,y1,x2,y2) for each of the A=~8400 anchors
            conf:  [B, A]     — each anchor's confidence (its highest class score)
        The attack reads `conf` and tries to push as many entries as possible
        above the detection threshold.
        """
        self._feats = None
        self.net(x)                        # runs the model; our pre-hook grabs the features
        if self._feats is None:
            raise RuntimeError("head pre-hook did not fire — model structure changed")
        # Re-run the one2one head ourselves on the captured (non-detached) features
        # so the result is differentiable w.r.t. the input pixels.
        preds = self.head.forward_head(self._feats, **self.head.one2one)
        y = self.head._inference(preds)    # decode -> [B, 4 box coords + nc class scores, A]
        boxes = y[:, :4, :]                # first 4 channels = box coordinates
        conf = y[:, 4:, :].amax(dim=1)     # confidence = the single highest class score
        return boxes, conf

    def anchor_centers(self):
        """Where each anchor sits in the image: its (x, y) pixel center, one per
        anchor. Used to tell which anchors fall in the real scene vs. the grey
        padding. Only valid after a dense() call has run (it populates the head's
        cached anchor grid)."""
        a, s = self.head.anchors, self.head.strides   # grid coords (a) and cell size (s)
        return a[0] * s[0], a[1] * s[0]               # grid coords x cell size = pixels
