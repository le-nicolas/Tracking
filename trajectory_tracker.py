"""Track an object and predict its trajectory in real time.

Example:
    python trajectory_tracker.py --source 0
    python trajectory_tracker.py --source ./video.mp4 --predict-steps 40
"""

from __future__ import annotations

import argparse
import collections
import time
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


SOURCE_TYPE = Union[int, str]


@dataclass
class MotionSample:
    t: float
    x: float
    y: float


class TrajectoryPredictor:
    def __init__(self, history_window: int = 18) -> None:
        self.history_window = history_window

    def predict(self, samples: Sequence[MotionSample], steps: int) -> List[Tuple[int, int]]:
        if len(samples) < 3 or steps <= 0:
            return []

        window = samples[-self.history_window :]
        t = np.array([point.t for point in window], dtype=np.float64)
        x = np.array([point.x for point in window], dtype=np.float64)
        y = np.array([point.y for point in window], dtype=np.float64)

        t = t - t[0]
        if np.allclose(t[-1], 0.0):
            return []

        diffs = np.diff(t)
        median_dt = float(np.median(diffs)) if diffs.size else 1.0 / 30.0
        if median_dt <= 0:
            median_dt = 1.0 / 30.0

        degree = 2 if len(window) >= 6 else 1
        degree = min(degree, len(window) - 1)
        x_poly = np.polyfit(t, x, degree)
        y_poly = np.polyfit(t, y, degree)

        future_t = t[-1] + median_dt * np.arange(1, steps + 1, dtype=np.float64)
        pred_x = np.polyval(x_poly, future_t)
        pred_y = np.polyval(y_poly, future_t)

        prediction = [(int(px), int(py)) for px, py in zip(pred_x, pred_y)]
        return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track an object in video and predict its trajectory."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source. Webcam index (e.g. 0) or file path.",
    )
    parser.add_argument(
        "--predict-steps",
        type=int,
        default=25,
        help="Number of future points to predict each frame.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=80,
        help="Maximum number of tracked center points to keep.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=18,
        help="How many recent points to use in prediction.",
    )
    return parser.parse_args()


def parse_source(source: str) -> SOURCE_TYPE:
    source = source.strip()
    if source.isdigit():
        return int(source)
    return source


def choose_target_roi(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    prompt = "Select object then press ENTER / SPACE"
    roi = cv2.selectROI(prompt, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(prompt)

    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def init_color_model(
    frame: np.ndarray, roi: Tuple[int, int, int, int]
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x, y, w, h = roi
    roi_frame = frame[y : y + h, x : x + w]
    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    lower = np.array((0.0, 30.0, 32.0))
    upper = np.array((180.0, 255.0, 255.0))
    mask = cv2.inRange(hsv_roi, lower, upper)

    histogram = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
    return histogram, (x, y, w, h)


def track_object(
    hsv_frame: np.ndarray,
    hue_hist: np.ndarray,
    track_window: Tuple[int, int, int, int],
) -> Tuple[Tuple[float, float], np.ndarray, Tuple[int, int, int, int]]:
    back_projection = cv2.calcBackProject([hsv_frame], [0], hue_hist, [0, 180], 1)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 1)
    rotated_rect, new_window = cv2.CamShift(back_projection, track_window, criteria)
    box = cv2.boxPoints(rotated_rect).astype(np.int32)

    center_x = float(np.mean(box[:, 0]))
    center_y = float(np.mean(box[:, 1]))
    return (center_x, center_y), box, new_window


def draw_overlay(
    frame: np.ndarray,
    box: np.ndarray,
    samples: Iterable[MotionSample],
    prediction: Sequence[Tuple[int, int]],
) -> None:
    cv2.polylines(frame, [box], True, (39, 174, 96), 2, cv2.LINE_AA)

    points = np.array([(int(s.x), int(s.y)) for s in samples], dtype=np.int32)
    if len(points) >= 2:
        cv2.polylines(frame, [points], False, (39, 174, 96), 2, cv2.LINE_AA)
    if len(points) >= 1:
        cv2.circle(frame, tuple(points[-1]), 4, (39, 174, 96), -1, cv2.LINE_AA)

    if prediction and len(points) >= 1:
        first = np.array([points[-1]], dtype=np.int32)
        pred = np.array(prediction, dtype=np.int32)
        chain = np.concatenate([first, pred], axis=0)
        cv2.polylines(frame, [chain], False, (52, 73, 235), 2, cv2.LINE_AA)

        step = max(1, len(prediction) // 8)
        for px, py in prediction[::step]:
            cv2.circle(frame, (px, py), 3, (52, 73, 235), -1, cv2.LINE_AA)

    speed_text = "Speed: n/a"
    if len(points) >= 2:
        speed = estimate_speed(samples)
        if speed is not None:
            speed_text = f"Speed: {speed:.1f} px/s"

    cv2.putText(
        frame,
        "q: quit  r: reselect target",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        speed_text,
        (14, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )


def estimate_speed(samples: Iterable[MotionSample]) -> Optional[float]:
    points = list(samples)
    if len(points) < 2:
        return None

    last = points[-1]
    prev = points[-2]
    dt = last.t - prev.t
    if dt <= 0:
        return None

    dx = last.x - prev.x
    dy = last.y - prev.y
    return float(np.hypot(dx, dy) / dt)


def main() -> None:
    args = parse_args()
    source = parse_source(args.source)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {source!r}")

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("Unable to read the first frame from source.")

    roi = choose_target_roi(first_frame)
    if roi is None:
        capture.release()
        raise RuntimeError("No ROI selected.")

    histogram, track_window = init_color_model(first_frame, roi)
    predictor = TrajectoryPredictor(history_window=args.history_window)
    samples: Deque[MotionSample] = collections.deque(maxlen=args.history_size)

    cv2.namedWindow("Trajectory Tracker", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        timestamp = time.perf_counter()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        center, box, track_window = track_object(hsv, histogram, track_window)

        samples.append(MotionSample(timestamp, center[0], center[1]))
        prediction = predictor.predict(list(samples), steps=args.predict_steps)

        draw_overlay(frame, box, samples, prediction)
        cv2.imshow("Trajectory Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            fresh = choose_target_roi(frame)
            if fresh is not None:
                histogram, track_window = init_color_model(frame, fresh)
                samples.clear()

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
