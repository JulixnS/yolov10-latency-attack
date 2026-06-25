"""Shared helpers: device selection, letterbox preprocessing, and the
YOLOv10 wrapper that exposes the *dense one2one* predictions (pre top-k)
as a differentiable tensor.

The whole attack hinges on getting the dense, pre-selection confidence
tensor out of the model. Ultralytics' `.predict()` hides it behind a
`torch.no_grad()` postprocess, so we call the underlying nn.Module forward
directly. Different Ultralytics versions return slightly different shapes,
so `verify.py` should be run once to confirm the layout before trusting
the attack.
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
    """Resize+pad a BGR uint8 image to a square `new`x`new`, preserving aspect.

    Returns (padded_bgr, ratio, (dw, dh)) so detections can be mapped back.
    Crafting MUST happen in this exact space — see README pitfall #1.
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


def image_to_tensor(path: str, imgsz: int, device: str) -> torch.Tensor:
    """Load an image from disk -> letterboxed [1,3,imgsz,imgsz] float tensor in [0,1], RGB."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    padded, _, _ = letterbox(bgr, imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device).float().div(255.0)
    return t.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,3,H,W]


class YoloV10Dense:
    """Wraps an Ultralytics YOLOv10 model and exposes the dense one2one
    decoded predictions (boxes + confidences) for white-box gradient attacks.

    Why a hook: on ultralytics 8.4.x the model forward returns the *post-top-k*
    `[B, max_det, 6]` tensor, and the head computes its one2one branch from
    DETACHED features (`x_detach`), so neither gives a differentiable dense
    tensor. We instead capture the head's non-detached input feature maps via a
    forward pre-hook and recompute the one2one branch ourselves. The forward
    values are identical to what's deployed; we just keep the graph to the input.
    """

    def __init__(self, weights: str = "yolov10n.pt", device: str = "cpu"):
        self.yolo = YOLO(weights)          # keep the high-level API for measurement
        self.net = self.yolo.model.to(device).eval()
        for p in self.net.parameters():    # freeze the victim: we attack the input, not the weights
            p.requires_grad_(False)
        self.device = device
        self.head = self.net.model[-1]     # v10Detect head
        if not getattr(self.head, "end2end", False):
            raise RuntimeError("Head is not end2end — is this really a YOLOv10 model?")
        self._feats = None
        self.head.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        # args[0] is the list of P3/P4/P5 feature maps feeding the head, with the
        # graph back to the input intact (the detach inside the head is on a copy).
        self._feats = list(args[0])

    def dense(self, x: torch.Tensor):
        """x: [B,3,H,W] in [0,1]. Returns (boxes, conf):
        boxes: [B, 4, A]   (xyxy in pixels at imgsz scale, decoded across all strides)
        conf:  [B, A]      (max class probability per anchor; the value NMS-free top-k filters on)
        """
        self._feats = None
        self.net(x)                        # triggers the pre-hook; also caches anchors/shape
        if self._feats is None:
            raise RuntimeError("head pre-hook did not fire — model structure changed")
        # Recompute the one2one branch from the NON-detached features -> differentiable.
        preds = self.head.forward_head(self._feats, **self.head.one2one)
        y = self.head._inference(preds)    # [B, 4+nc, A]
        boxes = y[:, :4, :]
        conf = y[:, 4:, :].amax(dim=1)     # max over class channels -> per-anchor confidence
        return boxes, conf
