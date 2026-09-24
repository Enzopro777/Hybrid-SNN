from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class TemporalFrame:
    epoch: int
    frame_index: int
    watermark_ms: float

class TemporalCore:
    """IA 1.6.27: orden causal desacoplado del reloj físico.

    El runtime conserva current_time por compatibilidad, pero el progreso
    lógico del cerebro se registra como eventos, epochs y watermarks.
    """
    VERSION = "1.6.27"

    def __init__(self):
        self.event_seq = 0
        self.causal_epoch = 0
        self.logical_frame = -1
        self.frame_watermark_ms = 0.0
        self.current_event_time_ms = 0.0
        self.last_event_origin = None
        self.trial_active = False
        self.trial_id = None

    def next_event(self, t_ms: float, origin=None, trial_id=None) -> int:
        self.event_seq += 1
        self.current_event_time_ms = max(float(self.current_event_time_ms), float(t_ms))
        self.last_event_origin = origin
        if trial_id is not None:
            self.trial_id = str(trial_id)
        return self.event_seq

    def observe_event(self, seq: int, t_ms: float, origin=None):
        self.event_seq = max(int(self.event_seq), int(seq))
        self.current_event_time_ms = max(float(self.current_event_time_ms), float(t_ms))
        self.last_event_origin = origin
        return int(seq)

    def begin_trial(self, trial_id, start_time_ms: float):
        self.causal_epoch += 1
        self.trial_active = True
        self.trial_id = str(trial_id)
        self.logical_frame = -1
        self.frame_watermark_ms = float(start_time_ms)
        self.current_event_time_ms = float(start_time_ms)

    def begin_frame(self, frame_index: int, watermark_ms: float) -> TemporalFrame:
        self.logical_frame = int(frame_index)
        self.frame_watermark_ms = float(watermark_ms)
        return TemporalFrame(self.causal_epoch, self.logical_frame, self.frame_watermark_ms)

    def end_trial(self):
        self.trial_active = False
        self.logical_frame = -1
        self.trial_id = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "event_seq": int(self.event_seq),
            "causal_epoch": int(self.causal_epoch),
            "logical_frame": int(self.logical_frame),
            "frame_watermark_ms": float(self.frame_watermark_ms),
            "current_event_time_ms": float(self.current_event_time_ms),
            "last_event_origin": self.last_event_origin,
            "trial_active": bool(self.trial_active),
            "trial_id": self.trial_id,
        }

    def restore(self, data: Optional[Dict[str, Any]]):
        data = data or {}
        self.event_seq = int(data.get("event_seq", 0))
        self.causal_epoch = int(data.get("causal_epoch", 0))
        self.logical_frame = int(data.get("logical_frame", -1))
        self.frame_watermark_ms = float(data.get("frame_watermark_ms", 0.0))
        self.current_event_time_ms = float(data.get("current_event_time_ms", 0.0))
        self.last_event_origin = data.get("last_event_origin")
        self.trial_active = bool(data.get("trial_active", False))
        self.trial_id = data.get("trial_id")
