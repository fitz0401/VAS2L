from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class GripperDetection:
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    score: float
    label: str
    predicted: bool = False


class GripperBBoxTracker:
    """YOLO-based gripper tracker for a single-class finetuned model."""

    def __init__(
        self,
        model_path: Path = Path("ckpts/yolo_best.pt"),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        kalman_process_noise: float = 1e-2,
        kalman_measurement_noise: float = 1e-1,
        max_missing_frames: int = 8,
    ):
        self.model_path = self._resolve_model_path(Path(model_path))
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_missing_frames = int(max_missing_frames)
        self._missing_frames = 0
        self._last_bbox_size: Optional[Tuple[int, int]] = None
        self._last_label = "gripper"
        self.kalman = self._build_kalman_filter(kalman_process_noise, kalman_measurement_noise)
        self._kalman_initialized = False

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for GripperBBoxTracker. Install with: pip install ultralytics") from exc

        self.model = YOLO(str(self.model_path))

    @staticmethod
    def _build_kalman_filter(process_noise: float, measurement_noise: float) -> cv2.KalmanFilter:
        # State: [x, y, vx, vy], Measurement: [x, y]
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * max(1e-6, float(process_noise))
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * max(1e-6, float(measurement_noise))
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        return kalman

    def _init_kalman(self, center: Tuple[int, int]) -> None:
        cx, cy = center
        self.kalman.statePre = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        self.kalman.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        self._kalman_initialized = True
        self._missing_frames = 0

    @staticmethod
    def _bbox_from_center_and_size(center: Tuple[int, int], size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        cx, cy = center
        w, h = size
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return x1, y1, x2, y2

    def _stabilize_detection(self, detection: Optional[GripperDetection]) -> Optional[GripperDetection]:
        if detection is not None:
            measurement = np.array([[float(detection.center[0])], [float(detection.center[1])]], dtype=np.float32)
            if not self._kalman_initialized:
                self._init_kalman(detection.center)
                filtered_center = detection.center
            else:
                self.kalman.predict()
                corrected = self.kalman.correct(measurement)
                filtered_center = (int(round(float(corrected[0]))), int(round(float(corrected[1]))))

            w = max(1, int(detection.bbox[2] - detection.bbox[0]))
            h = max(1, int(detection.bbox[3] - detection.bbox[1]))
            self._last_bbox_size = (w, h)
            self._last_label = detection.label
            self._missing_frames = 0

            return GripperDetection(
                bbox=self._bbox_from_center_and_size(filtered_center, (w, h)),
                center=filtered_center,
                score=detection.score,
                label=detection.label,
                predicted=False,
            )

        if not self._kalman_initialized or self._last_bbox_size is None:
            return None

        if self._missing_frames >= self.max_missing_frames:
            return None

        predicted = self.kalman.predict()
        pred_center = (int(round(float(predicted[0]))), int(round(float(predicted[1]))))
        self._missing_frames += 1

        return GripperDetection(
            bbox=self._bbox_from_center_and_size(pred_center, self._last_bbox_size),
            center=pred_center,
            score=0.0,
            label=self._last_label,
            predicted=True,
        )

    @staticmethod
    def _resolve_model_path(model_path: Path) -> Path:
        if model_path.exists():
            return model_path
        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root / model_path
        if candidate.exists():
            return candidate
        return model_path

    @staticmethod
    def center_from_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))

    def detect(self, frame_rgb: np.ndarray) -> Optional[GripperDetection]:
        if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
            raise ValueError("Expected RGB frame with shape HxWx3")

        results = self.model.predict(
            source=frame_rgb,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        if not results:
            return self._stabilize_detection(None)

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return self._stabilize_detection(None)

        confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones(len(boxes))
        best_idx = int(np.argmax(confs))

        xyxy = boxes.xyxy[best_idx].detach().cpu().numpy().tolist()
        bbox = tuple(int(round(v)) for v in xyxy)
        center = self.center_from_bbox(bbox)
        cls_id = int(boxes.cls[best_idx].item()) if getattr(boxes, "cls", None) is not None else -1
        names = result.names if hasattr(result, "names") else {}
        if isinstance(names, dict):
            label = str(names.get(cls_id, "gripper"))
        elif isinstance(names, list) and 0 <= cls_id < len(names):
            label = str(names[cls_id])
        else:
            label = "gripper"

        raw_detection = GripperDetection(
            bbox=bbox,
            center=center,
            score=float(confs[best_idx]),
            label=label,
        )
        return self._stabilize_detection(raw_detection)

    @staticmethod
    def _segment_color(index: int, total: int) -> Tuple[int, int, int]:
        if total <= 1:
            return 255, 80, 70
        start = np.array([70, 120, 255], dtype=np.float32)
        end = np.array([255, 80, 70], dtype=np.float32)
        ratio = float(index) / float(max(1, total - 1))
        color = (1.0 - ratio) * start + ratio * end
        return tuple(int(round(v)) for v in color)

    def draw_overlay(
        self,
        frame_rgb: np.ndarray,
        detection: Optional[GripperDetection],
        history_centers: Sequence[Tuple[int, int]] = (),
        show_text: bool = True,
    ) -> np.ndarray:
        canvas = frame_rgb.copy()

        centers = [tuple(map(int, center)) for center in history_centers]
        if len(centers) >= 2:
            total = len(centers) - 1
            for idx in range(1, len(centers)):
                color = self._segment_color(idx - 1, total)
                thickness = 5 if idx < len(centers) - 1 else 7
                start_pt = centers[idx - 1]
                end_pt = centers[idx]
                if idx == len(centers) - 1:
                    cv2.arrowedLine(canvas, start_pt, end_pt, color, thickness, tipLength=0.25)
                else:
                    cv2.line(canvas, start_pt, end_pt, color, thickness)
            latest_center = centers[-1]
            cv2.circle(canvas, latest_center, 7, (255, 80, 70), -1)
            cv2.circle(canvas, latest_center, 11, (255, 255, 255), 2)

        if detection is not None:
            x1, y1, x2, y2 = detection.bbox
            box_color = (255, 255, 255) if not detection.predicted else (170, 170, 170)
            center_color = (255, 80, 70) if not detection.predicted else (70, 170, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(canvas, detection.center, 4, center_color, -1)
            if show_text:
                label_text = f"{detection.label} {detection.score:.2f}"
                if detection.predicted:
                    label_text += " pred"
                text_pos = (max(0, x1), max(24, y1 - 8))
                cv2.putText(
                    canvas,
                    label_text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                status_text = f"bbox=({x1}, {y1}, {x2}, {y2}) center=({detection.center[0]}, {detection.center[1]})"
                if detection.predicted:
                    status_text += " [pred]"
            else:
                status_text = None
        else:
            status_text = "bbox=none center=none" if show_text else None

        if show_text and status_text is not None:
            cv2.putText(
                canvas,
                status_text,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 80, 70),
                2,
                cv2.LINE_AA,
            )
        return canvas
