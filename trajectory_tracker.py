"""Detector-based trajectory tracking with short-horizon prediction.

Examples:
    python trajectory_tracker.py --source 0
    python trajectory_tracker.py --source ./sample.mp4 --target-class person
"""

from __future__ import annotations

import argparse
import collections
import math
import time
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as error:
    raise SystemExit(
        "Missing dependency 'ultralytics'. Install with `pip install -r requirements.txt`."
    ) from error


SOURCE_TYPE = Union[int, str]


@dataclass
class MotionSample:
    t: float
    x: float
    y: float


@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]
    center: Tuple[float, float]
    area: float


class TrajectoryPredictor:
    def __init__(self, history_window: int = 20) -> None:
        self.history_window = history_window

    def predict(self, samples: Sequence[MotionSample], steps: int) -> List[Tuple[int, int]]:
        if len(samples) < 3 or steps <= 0:
            return []

        window = samples[-self.history_window :]
        t = np.array([sample.t for sample in window], dtype=np.float64)
        x = np.array([sample.x for sample in window], dtype=np.float64)
        y = np.array([sample.y for sample in window], dtype=np.float64)

        t = t - t[0]
        if np.allclose(t[-1], 0.0):
            return []

        diffs = np.diff(t)
        median_dt = float(np.median(diffs)) if diffs.size else 1.0 / 30.0
        if median_dt <= 0:
            median_dt = 1.0 / 30.0

        degree = 2 if len(window) >= 6 else 1
        degree = min(degree, len(window) - 1)

        try:
            x_poly = np.polyfit(t, x, degree)
            y_poly = np.polyfit(t, y, degree)
        except np.linalg.LinAlgError:
            return []

        future_t = t[-1] + median_dt * np.arange(1, steps + 1, dtype=np.float64)
        pred_x = np.polyval(x_poly, future_t)
        pred_y = np.polyval(y_poly, future_t)
        return [(int(px), int(py)) for px, py in zip(pred_x, pred_y)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track detected objects and predict future trajectory for a selected target."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. 0) or file path.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics model path/name.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Tracking config for Ultralytics (e.g. bytetrack.yaml, botsort.yaml).",
    )
    parser.add_argument(
        "--target-class",
        default="",
        help="Optional class name or class id to track (example: person or 0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Detection IoU threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size.",
    )
    parser.add_argument(
        "--predict-steps",
        type=int,
        default=25,
        help="Number of future points to predict.",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=90,
        help="Max points kept per track id.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=20,
        help="Recent points used for prediction fit.",
    )
    parser.add_argument(
        "--lost-timeout",
        type=float,
        default=0.9,
        help="Seconds to wait before auto-switching selected target when lost.",
    )
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=6.0,
        help="Drop trajectories not seen for this many seconds.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Ultralytics device (examples: cpu, 0).",
    )
    parser.add_argument(
        "--visual-mode",
        default="surreal",
        choices=("surreal", "classic"),
        help="Overlay style. 'surreal' is intentionally unconventional.",
    )
    return parser.parse_args()


def parse_source(source: str) -> SOURCE_TYPE:
    source = source.strip()
    if source.isdigit():
        return int(source)
    return source


def model_names_map(model: YOLO) -> Dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {idx: str(name) for idx, name in enumerate(names)}


def resolve_class_filter(model: YOLO, target_class: str) -> Optional[List[int]]:
    candidate = target_class.strip()
    if not candidate:
        return None

    names = model_names_map(model)
    if candidate.isdigit():
        class_id = int(candidate)
        if class_id not in names:
            raise ValueError(f"Class id {class_id} not found in model labels.")
        return [class_id]

    class_name = candidate.lower()
    for class_id, label in names.items():
        if label.lower() == class_name:
            return [class_id]

    raise ValueError(f"Class name '{candidate}' not found in model labels.")


def extract_detections(result, names: Dict[int, str]) -> List[Detection]:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []

    ids = boxes.id.int().cpu().tolist()
    xyxy = boxes.xyxy.cpu().tolist()
    xywh = boxes.xywh.cpu().tolist()
    confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(ids)
    classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(ids)

    detections: List[Detection] = []
    for track_id, box_xyxy, box_xywh, conf, class_id in zip(ids, xyxy, xywh, confs, classes):
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        cx, cy, w, h = [float(v) for v in box_xywh]
        detections.append(
            Detection(
                track_id=int(track_id),
                class_id=int(class_id),
                class_name=names.get(int(class_id), str(class_id)),
                confidence=float(conf),
                box_xyxy=(x1, y1, x2, y2),
                center=(cx, cy),
                area=float(w * h),
            )
        )
    return detections


def choose_best_target(detections: Sequence[Detection]) -> Optional[int]:
    if not detections:
        return None
    best = max(detections, key=lambda item: (item.area, item.confidence))
    return best.track_id


def cycle_target(current_id: Optional[int], visible_ids: Sequence[int]) -> Optional[int]:
    if not visible_ids:
        return None

    ordered = sorted(set(int(track_id) for track_id in visible_ids))
    if current_id not in ordered:
        return ordered[0]

    idx = ordered.index(int(current_id))
    return ordered[(idx + 1) % len(ordered)]


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


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def blend_bgr(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = float(clamp(t, 0.0, 1.0))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def color_for_track(track_id: int, active: bool = False) -> Tuple[int, int, int]:
    seed = (track_id * 1103515245 + 12345) & 0x7FFFFFFF
    hue = seed % 180
    sat = 170 if active else 145
    val = 250 if active else 220
    hsv = np.uint8([[[hue, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_corner_brackets(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    corner = max(10, min(34, int(min(width, height) * 0.28)))

    cv2.line(image, (x1, y1), (x1 + corner, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y1), (x1, y1 + corner), color, thickness, cv2.LINE_AA)

    cv2.line(image, (x2, y1), (x2 - corner, y1), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y1), (x2, y1 + corner), color, thickness, cv2.LINE_AA)

    cv2.line(image, (x1, y2), (x1 + corner, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y2), (x1, y2 - corner), color, thickness, cv2.LINE_AA)

    cv2.line(image, (x2, y2), (x2 - corner, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y2), (x2, y2 - corner), color, thickness, cv2.LINE_AA)


def draw_prediction_glyph(
    layer: np.ndarray,
    point: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: int,
) -> None:
    px, py = point
    scale = max(2, scale)
    glyph = np.array(
        [
            [px, py - scale],
            [px + scale, py],
            [px, py + scale],
            [px - scale, py],
        ],
        dtype=np.int32,
    )
    cv2.polylines(layer, [glyph], True, color, 1, cv2.LINE_AA)
    cv2.circle(layer, (px, py), max(1, scale // 2), color, -1, cv2.LINE_AA)


def draw_speed_meter(
    frame: np.ndarray,
    fps: float,
    speed: Optional[float],
    selected_id: Optional[int],
    visible_count: int,
    visual_mode: str,
) -> None:
    h, _w = frame.shape[:2]
    panel_h = 146
    panel_w = 274
    origin = (16, h - panel_h - 16)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        origin,
        (origin[0] + panel_w, origin[1] + panel_h),
        (24, 21, 18),
        -1,
    )
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, dst=frame)
    cv2.rectangle(
        frame,
        origin,
        (origin[0] + panel_w, origin[1] + panel_h),
        (92, 141, 209),
        1,
        cv2.LINE_AA,
    )

    # Vertical bars intentionally offset for a non-standard HUD rhythm.
    speed_value = 0.0 if speed is None else float(speed)
    fps_bar = int(clamp(fps / 75.0, 0.0, 1.0) * 88)
    speed_bar = int(clamp(speed_value / 520.0, 0.0, 1.0) * 88)
    bx = origin[0] + 16
    by = origin[1] + 116
    cv2.rectangle(frame, (bx, by - 88), (bx + 22, by), (58, 58, 58), 1)
    cv2.rectangle(frame, (bx + 32, by - 88), (bx + 54, by), (58, 58, 58), 1)
    cv2.rectangle(frame, (bx + 3, by - fps_bar), (bx + 19, by), (247, 198, 86), -1)
    cv2.rectangle(frame, (bx + 35, by - speed_bar), (bx + 51, by), (110, 221, 243), -1)

    text_color = (224, 232, 240)
    accent = (176, 206, 255)
    cv2.putText(frame, "FPS", (bx - 1, by + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "SPD", (bx + 29, by + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_color, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"mode:{visual_mode}",
        (origin[0] + 80, origin[1] + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        accent,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"tracks:{visible_count}",
        (origin[0] + 80, origin[1] + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        text_color,
        1,
        cv2.LINE_AA,
    )
    selected_text = "none" if selected_id is None else str(selected_id)
    cv2.putText(
        frame,
        f"selected:{selected_text}",
        (origin[0] + 80, origin[1] + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        text_color,
        1,
        cv2.LINE_AA,
    )
    speed_text = "n/a" if speed is None else f"{speed:.1f}px/s"
    cv2.putText(
        frame,
        f"velocity:{speed_text}",
        (origin[0] + 80, origin[1] + 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        text_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "keys: q quit  n next  c clear  v mode",
        (origin[0] + 12, origin[1] + panel_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (164, 179, 196),
        1,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    detections: Sequence[Detection],
    trails: Dict[int, Deque[MotionSample]],
    selected_id: Optional[int],
    prediction: Sequence[Tuple[int, int]],
    fps: float,
    visual_mode: str,
    now: float,
) -> None:
    if visual_mode == "classic":
        draw_overlay_classic(frame, detections, trails, selected_id, prediction, fps=fps)
        return
    draw_overlay_surreal(frame, detections, trails, selected_id, prediction, fps=fps, now=now)


def draw_overlay_surreal(
    frame: np.ndarray,
    detections: Sequence[Detection],
    trails: Dict[int, Deque[MotionSample]],
    selected_id: Optional[int],
    prediction: Sequence[Tuple[int, int]],
    fps: float,
    now: float,
) -> None:
    h, w = frame.shape[:2]
    tint = np.full_like(frame, (16, 22, 30))
    cv2.addWeighted(frame, 0.83, tint, 0.17, 0, dst=frame)

    layer = np.zeros_like(frame)

    scan_step = 26
    drift = int((now * 40) % scan_step)
    for y in range(-drift, h, scan_step):
        cv2.line(layer, (0, y), (w, y), (22, 18, 14), 1, cv2.LINE_AA)

    for detection in detections:
        active = detection.track_id == selected_id
        color = color_for_track(detection.track_id, active=active)
        x1, y1, x2, y2 = detection.box_xyxy

        draw_corner_brackets(layer, (x1, y1, x2, y2), color, 2 if active else 1)
        cx, cy = int(detection.center[0]), int(detection.center[1])
        base_radius = int(clamp(math.sqrt(max(1.0, detection.area)) * 0.1, 9, 56))
        cv2.circle(layer, (cx, cy), base_radius, color, 1, cv2.LINE_AA)
        if active:
            pulse = int(6 + 6 * (math.sin(now * 4.2) * 0.5 + 0.5))
            cv2.circle(layer, (cx, cy), base_radius + pulse, color, 2, cv2.LINE_AA)
            cv2.line(layer, (cx - 12, cy), (cx + 12, cy), color, 1, cv2.LINE_AA)
            cv2.line(layer, (cx, cy - 12), (cx, cy + 12), color, 1, cv2.LINE_AA)

        label = (
            f"#{detection.track_id} {detection.class_name} {detection.confidence:.2f}"
            if active
            else f"#{detection.track_id}:{detection.class_name}"
        )
        offset = 16 if detection.track_id % 2 == 0 else -168
        tx = int(clamp(x1 + offset, 6, w - 188))
        ty = int(clamp(y1 - 10 if y1 > 28 else y2 + 18, 22, h - 10))
        cv2.putText(
            layer,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )

    speed: Optional[float] = None
    if selected_id is not None and selected_id in trails:
        selected_trail = list(trails[selected_id])
        speed = estimate_speed(selected_trail)
        points = np.array([(int(item.x), int(item.y)) for item in selected_trail], dtype=np.int32)

        if len(points) >= 2:
            trail_color = color_for_track(selected_id, active=True)
            for idx in range(1, len(points)):
                t = idx / max(1, len(points) - 1)
                seg_color = blend_bgr((58, 73, 94), trail_color, t)
                thickness = 1 + int(round(t * 3))
                cv2.line(
                    layer,
                    tuple(points[idx - 1]),
                    tuple(points[idx]),
                    seg_color,
                    thickness,
                    cv2.LINE_AA,
                )

            head = tuple(points[-1])
            orbit_base = int(clamp((0.0 if speed is None else speed) * 0.04, 16, 64))
            for ring in range(3):
                radius = orbit_base + ring * 14
                start = int((now * 90 + ring * 120) % 360)
                cv2.ellipse(
                    layer,
                    head,
                    (radius, radius),
                    0,
                    start,
                    start + 230,
                    trail_color,
                    1,
                    cv2.LINE_AA,
                )
            cv2.circle(layer, head, 5, trail_color, -1, cv2.LINE_AA)

            for idx, point in enumerate(prediction):
                if idx % 2 == 1 and idx != len(prediction) - 1:
                    continue
                t = idx / max(1, len(prediction) - 1)
                glyph_color = blend_bgr(trail_color, (255, 255, 255), 0.18 + 0.5 * t)
                glyph_size = 3 + int(round(t * 4))
                draw_prediction_glyph(layer, (int(point[0]), int(point[1])), glyph_color, glyph_size)
    cv2.addWeighted(frame, 1.0, layer, 0.82, 0, dst=frame)
    draw_speed_meter(
        frame,
        fps=fps,
        speed=speed,
        selected_id=selected_id,
        visible_count=len(detections),
        visual_mode="surreal",
    )


def draw_overlay_classic(
    frame: np.ndarray,
    detections: Sequence[Detection],
    trails: Dict[int, Deque[MotionSample]],
    selected_id: Optional[int],
    prediction: Sequence[Tuple[int, int]],
    fps: float,
) -> None:
    for detection in detections:
        active = detection.track_id == selected_id
        color = (39, 174, 96) if active else (140, 140, 140)
        thickness = 2 if active else 1
        x1, y1, x2, y2 = detection.box_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = (
            f"#{detection.track_id} {detection.class_name} {detection.confidence:.2f}"
            if active
            else f"#{detection.track_id} {detection.class_name}"
        )
        text_y = y1 - 8 if y1 > 16 else y1 + 18
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    if selected_id is not None and selected_id in trails:
        selected_trail = trails[selected_id]
        points = np.array([(int(item.x), int(item.y)) for item in selected_trail], dtype=np.int32)

        if len(points) >= 2:
            cv2.polylines(frame, [points], False, (39, 174, 96), 2, cv2.LINE_AA)
        if len(points) >= 1:
            cv2.circle(frame, tuple(points[-1]), 4, (39, 174, 96), -1, cv2.LINE_AA)

            if prediction:
                pred = np.array(prediction, dtype=np.int32)
                full_path = np.concatenate([points[-1:].copy(), pred], axis=0)
                cv2.polylines(frame, [full_path], False, (52, 73, 235), 2, cv2.LINE_AA)
                step = max(1, len(prediction) // 8)
                for px, py in prediction[::step]:
                    cv2.circle(frame, (int(px), int(py)), 3, (52, 73, 235), -1, cv2.LINE_AA)

        speed = estimate_speed(selected_trail)
    else:
        speed = None

    draw_speed_meter(
        frame,
        fps=fps,
        speed=speed,
        selected_id=selected_id,
        visible_count=len(detections),
        visual_mode="classic",
    )


def main() -> None:
    args = parse_args()
    source = parse_source(args.source)

    model = YOLO(args.model)
    names = model_names_map(model)
    classes_filter = resolve_class_filter(model, args.target_class)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {source!r}")

    predictor = TrajectoryPredictor(history_window=args.history_window)
    trails: Dict[int, Deque[MotionSample]] = {}
    last_seen: Dict[int, float] = {}

    selected_id: Optional[int] = None
    lost_since: Optional[float] = None
    fps_smooth = 0.0
    prev_frame_time = time.perf_counter()

    visual_mode = args.visual_mode
    cv2.namedWindow("Trajectory Tracker", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        now = time.perf_counter()
        frame_dt = max(1e-6, now - prev_frame_time)
        current_fps = 1.0 / frame_dt
        fps_smooth = current_fps if fps_smooth == 0.0 else (fps_smooth * 0.9 + current_fps * 0.1)
        prev_frame_time = now

        tracking_results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=classes_filter,
            device=None if not args.device.strip() else args.device.strip(),
            verbose=False,
        )

        detections = extract_detections(tracking_results[0], names)
        visible_ids = [item.track_id for item in detections]

        for detection in detections:
            if detection.track_id not in trails:
                trails[detection.track_id] = collections.deque(maxlen=args.history_size)
            trails[detection.track_id].append(MotionSample(now, detection.center[0], detection.center[1]))
            last_seen[detection.track_id] = now

        stale_ids = [track_id for track_id, ts in last_seen.items() if now - ts > args.stale_seconds]
        for track_id in stale_ids:
            last_seen.pop(track_id, None)
            trails.pop(track_id, None)
            if selected_id == track_id:
                selected_id = None
                lost_since = None

        if selected_id is None:
            selected_id = choose_best_target(detections)
            lost_since = None
        elif selected_id not in visible_ids:
            if lost_since is None:
                lost_since = now
            elif now - lost_since >= args.lost_timeout:
                selected_id = choose_best_target(detections)
                lost_since = None
        else:
            lost_since = None

        selected_trail = list(trails[selected_id]) if selected_id is not None and selected_id in trails else []
        prediction = predictor.predict(selected_trail, steps=args.predict_steps)

        draw_overlay(
            frame,
            detections,
            trails,
            selected_id,
            prediction,
            fps=fps_smooth,
            visual_mode=visual_mode,
            now=now,
        )
        cv2.imshow("Trajectory Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n"):
            selected_id = cycle_target(selected_id, visible_ids)
            lost_since = None
        if key == ord("c"):
            selected_id = None
            lost_since = None
        if key == ord("v"):
            visual_mode = "classic" if visual_mode == "surreal" else "surreal"

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
