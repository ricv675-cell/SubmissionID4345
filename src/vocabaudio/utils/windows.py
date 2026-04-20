from __future__ import annotations

import math


def normalize_windows(windows: list[list[float]], duration: float) -> list[list[float]]:
    out = []
    for start, end in windows:
        start = max(0.0, min(float(start), float(duration)))
        end = max(0.0, min(float(end), float(duration)))
        if end < start:
            start, end = end, start
        if end <= start:
            end = min(duration, start + 1e-3)
        out.append([round(start, 4), round(end, 4)])
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def seconds_to_step_idx(second: float, duration: float, num_steps: int, use_end: bool) -> int:
    if num_steps <= 0:
        return 0
    sec_per_step = max(duration / num_steps, 1e-6)
    if use_end:
        idx = int(math.ceil(second / sec_per_step) - 1)
    else:
        idx = int(math.floor(second / sec_per_step))
    return max(0, min(idx, num_steps - 1))


def step_span_to_seconds(start_idx: int, end_idx: int, duration: float, num_steps: int) -> list[float]:
    if num_steps <= 0:
        return [0.0, 0.0]
    sec_per_step = duration / num_steps
    start = max(0.0, min(start_idx * sec_per_step, duration))
    end = max(0.0, min((end_idx + 1) * sec_per_step, duration))
    if end <= start:
        end = min(duration, start + sec_per_step)
    return [round(start, 4), round(end, 4)]


def temporal_iou(window_a: list[float], window_b: list[float]) -> float:
    a0, a1 = window_a[:2]
    b0, b1 = window_b[:2]
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0

