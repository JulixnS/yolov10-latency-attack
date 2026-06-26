"""SlowTrack's ByteTrack-lineage MOT, vendored from
https://github.com/ershang2/SlowTrack (yolox/tracker/).

`BYTETracker.update(output_results, img_info, img_size, stage_track_timer)`
expects an [N,5] array of [x1,y1,x2,y2,score] and a timer object exposing
tic()/toc()/average_time. `Timer` below is a minimal drop-in for that.
"""
import time

from .byte_tracker import BYTETracker

__all__ = ["BYTETracker", "Timer"]


class Timer:
    """Minimal stand-in for yolox's stage timer (tic/toc/average_time)."""

    def __init__(self):
        self.total = 0.0
        self.calls = 0
        self._t = None

    def tic(self):
        self._t = time.perf_counter()

    def toc(self):
        if self._t is not None:
            self.total += time.perf_counter() - self._t
            self.calls += 1
        return self.total

    @property
    def average_time(self):
        return self.total / max(self.calls, 1)
