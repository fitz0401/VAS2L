from __future__ import annotations

import argparse
import csv
import gc
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image, ImageDraw
import zmq

from VA2L.utils.gripper_tracker import GripperBBoxTracker, GripperDetection
from VA2L.vlm_inference import VLMInference


_IGNORED_OBJECT_TOKENS = {
    "table",
    "table top",
    "tabletop",
    "floor",
    "wall",
    "room",
    "background",
    "person",
    "human",
    "robot",
    "robot arm",
    "robot body",
    "robot base",
    "arm",
    "gripper",
    "end effector",
    "end-effector",
    "manipulator",
}


def _compose_unified_prompt(object_set: List[str], visual_prompt_mode: str, gripper_track_seconds: float) -> str:
    object_text = ", ".join(object_set) if object_set else "none"
    if visual_prompt_mode == "gripper":
        prompt = (
            f"Infer the task the robotic arm is performing from the image, in which the gripper's past {gripper_track_seconds:g}-second trajectory are visible.\n"
            "Format: <action> <target object> (<relation> <relation object>)\n"
            "Rule 1: choose the action from pick, place, insert, open, close, move, turn.\n"
            "Rule 2: relation and relation object are optional, relation can be on, into, from, next to.\n"
            "Rule 3: Output one short sentence only, starting with the action. No explanation.\n"
            "Rule 4: If the gripper is not holding an object, prioritize 'pick' action.\n"
        )
    else:
        prompt = (
            "Infer the task the robotic arm is performing from the image, in which the motion in the last 2 seconds is labeled.\n"
            "Format: <action> <target object> (<relation> <relation object>)\n"
            "Rule 1: choose the action from pick, place, insert, open, close, move, turn.\n"
            "Rule 2: relation and relation object are optional, relation can be on, into, from, next to.\n"
            "Rule 3: Output one short sentence only, starting with the action. No explanation.\n"
            "Rule 4: If the gripper is not holding an object, prioritize 'pick' action.\n"
            "Rule 5: The motion overlay marks the last 2 seconds of image difference.\n"
        )
    return prompt + f"Hint: Object set: {object_text}\n"


def _normalize_object_list(text: str) -> List[str]:
    if not text:
        return []
    cleaned = text.strip().lower()
    cleaned = re.sub(r"^(detected objects|objects|final objects)\s*[:=]\s*", "", cleaned).strip()
    parts = re.split(r"\n|\||,|;", cleaned)
    out: List[str] = []
    seen = set()
    for part in parts:
        token = re.sub(r"\s+", " ", part).strip(" .-\t")
        if not token or token in {"none", "n/a", "na"}:
            continue
        if token in _IGNORED_OBJECT_TOKENS:
            continue
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _initialize_object_set(first_rgb: np.ndarray, vlm: VLMInference, enable_ram: bool) -> Tuple[List[str], str]:
    first_image = Image.fromarray(first_rgb)

    if not enable_ram:
        prompt = (
            "List tabletop manipulable objects visible in this image.\n"
            "Ignore table/floor/wall/background/person/robot body and gripper.\n"
            "Output one line only: comma-separated lowercase object names. If none, output exactly: none"
        )
        merged_text = vlm.infer(first_image, prompt).strip()
        return _normalize_object_list(merged_text), merged_text

    from vision_module.recognize_anything import RAMRecognizer

    ram = RAMRecognizer()
    try:
        raw_tags, _, _ = ram.recognize(first_image)
    finally:
        del ram
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    prompt = (
        "You are filtering object detections for tabletop robot manipulation.\n"
        f"RAM detected tags: {raw_tags}\n"
        "Task: merge synonyms and keep only tabletop manipulable objects visible in the image.\n"
        "Ignore background words (table/floor/wall/room), person/human, and robot body/end-effector/gripper/arm.\n"
        "Output format: one line comma-separated object names in lowercase. If none, output exactly: none"
    )
    merged_text = vlm.infer(first_image, prompt).strip()
    normalized = _normalize_object_list(merged_text)
    if not normalized:
        normalized = _normalize_object_list(", ".join(raw_tags))
    return normalized, merged_text


def _save_debug_step(
    debug_dir: Path,
    step_idx: int,
    overlay: Image.Image,
    action_text: str,
    prompt_mode: str,
    gripper_bbox: Optional[Tuple[int, int, int, int]] = None,
    gripper_center: Optional[Tuple[int, int]] = None,
) -> None:
    step_dir = debug_dir / f"f{step_idx:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    overlay.save(step_dir / "vlm_input.png")
    (step_dir / "vlm_output.txt").write_text(action_text, encoding="utf-8")
    (step_dir / "prompt_mode.txt").write_text(prompt_mode, encoding="utf-8")
    if gripper_bbox is not None:
        (step_dir / "gripper_bbox.txt").write_text(", ".join(map(str, gripper_bbox)), encoding="utf-8")
    if gripper_center is not None:
        (step_dir / "gripper_center.txt").write_text(", ".join(map(str, gripper_center)), encoding="utf-8")


def _draw_debug_caption(frame_rgb: np.ndarray, caption: str) -> np.ndarray:
    canvas = frame_rgb.copy()
    lines = [line.strip() for line in caption.splitlines() if line.strip()]
    if not lines:
        return canvas

    top = 12
    left = 12
    line_height = 28
    padding_y = 10
    padding_x = 14
    max_width = 0
    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        max_width = max(max_width, text_width)

    box_right = min(canvas.shape[1] - 8, left + max_width + padding_x * 2)
    box_bottom = min(canvas.shape[0] - 8, top + len(lines) * line_height + padding_y * 2)
    cv2.rectangle(canvas, (left - 6, top - 6), (box_right, box_bottom), (0, 0, 0), -1)
    cv2.rectangle(canvas, (left - 6, top - 6), (box_right, box_bottom), (255, 255, 255), 2)

    y = top + 18
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (left, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += line_height
    return canvas


def _build_debug_video_writer(debug_dir: Path, fps: float, frame_size: Tuple[int, int]) -> Tuple[cv2.VideoWriter, Path]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    video_path = debug_dir / f"debug_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps)),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open debug video writer at {video_path}")
    return writer, video_path


def _build_diff_overlay(
    current_rgb: np.ndarray,
    one_sec_rgb: np.ndarray,
    two_sec_rgb: np.ndarray,
    diff_threshold: int,
    min_diff_area: int,
) -> Image.Image:
    def compute_bbox(a_rgb: np.ndarray, b_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        diff = cv2.absdiff(a_rgb, b_rgb)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, diff_threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.medianBlur(binary, 5)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        best = max(contours, key=cv2.contourArea)
        if float(cv2.contourArea(best)) < float(min_diff_area):
            return None
        x, y, w, h = cv2.boundingRect(best)
        return x, y, x + w, y + h

    def bbox_center(bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int]]:
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))

    canvas = Image.fromarray(current_rgb.copy())
    draw = ImageDraw.Draw(canvas)
    cur_bbox = compute_bbox(current_rgb, one_sec_rgb)
    prev_bbox = compute_bbox(one_sec_rgb, two_sec_rgb)
    start_pt = bbox_center(prev_bbox)
    end_pt = bbox_center(cur_bbox)

    if cur_bbox is not None:
        draw.rectangle(cur_bbox, outline=(255, 255, 255), width=3)

    if start_pt is not None and end_pt is not None:
        draw.line([start_pt, end_pt], fill=(255, 80, 70), width=4)
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        norm = float(np.hypot(dx, dy))
        if norm > 1e-6:
            ux, uy = dx / norm, dy / norm
            head_len = 12.0
            head_w = 7.0
            tip = np.array([end_pt[0], end_pt[1]], dtype=np.float64)
            base = tip - head_len * np.array([ux, uy], dtype=np.float64)
            perp = np.array([-uy, ux], dtype=np.float64)
            left = base + head_w * perp
            right = base - head_w * perp
            triangle = [tuple(tip.astype(int)), tuple(left.astype(int)), tuple(right.astype(int))]
            draw.polygon(triangle, fill=(255, 80, 70))

    return canvas


def _build_gripper_overlay(
    current_rgb: np.ndarray,
    tracker: GripperBBoxTracker,
    history_centers: Sequence[Tuple[int, int]],
    detection: Optional[GripperDetection],
) -> Image.Image:
    return Image.fromarray(tracker.draw_overlay(current_rgb, detection, history_centers, show_text=False))


def _start_intent_server(
    bind_host: str,
    port: int,
    latest_state: Dict[str, Any],
    state_lock: threading.Lock,
    stop_event: threading.Event,
) -> Tuple[threading.Thread, Any, Any]:
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.REP)
    socket.linger = 0
    endpoint = f"tcp://{bind_host}:{port}"
    socket.bind(endpoint)

    def _server_loop() -> None:
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        while not stop_event.is_set():
            events = dict(poller.poll(timeout=100))
            if socket not in events:
                continue

            try:
                req = socket.recv_string()
            except Exception:
                continue

            cmd = req.strip().lower()
            if cmd == "get_intent":
                with state_lock:
                    has_action = bool(latest_state.get("has_action", False))
                    cached = dict(latest_state)
                if has_action:
                    socket.send_json(
                        {
                            "ok": True,
                            "action": cached.get("action", ""),
                            "frame_index": cached.get("frame_index", -1),
                            "ts": cached.get("ts", 0.0),
                            "object_set": cached.get("object_set", []),
                            "visual_prompt_mode": cached.get("visual_prompt_mode", "diff"),
                            "gripper_center": cached.get("gripper_center", None),
                        }
                    )
                else:
                    socket.send_json({"ok": False, "message": "no cached intent yet"})
            else:
                socket.send_json(
                    {
                        "ok": False,
                        "message": f"unknown command: {req}",
                        "supported": ["get_intent"],
                    }
                )

    worker = threading.Thread(target=_server_loop, daemon=True)
    worker.start()
    return worker, socket, ctx


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-world intent detection with diff/gripper visual prompting.")
    parser.add_argument("--device-id", type=str, default="f1421698")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", type=str, default="qwen-vl-4b")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--precision", type=str, default="auto")
    parser.add_argument("--enable-ram", action="store_true", help="Enable RAM-assisted object-set initialization.")
    parser.add_argument("--visual-prompt-mode", type=str, choices=["diff", "gripper"], default="diff")
    parser.add_argument("--gripper-model-path", type=Path, default=Path("ckpts/yolo_best.pt"))
    parser.add_argument("--gripper-conf-threshold", type=float, default=0.25)
    parser.add_argument("--gripper-iou-threshold", type=float, default=0.45)
    parser.add_argument("--gripper-track-seconds", type=float, default=1.0, help="History window used in gripper visual prompt mode.")
    parser.add_argument("--diff-threshold", type=int, default=24)
    parser.add_argument("--min-diff-area", type=int, default=250)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=Path("dataset/realworld/debug_intent"))
    parser.add_argument("--output", "--ouput", action="store_true", dest="output", help="If set, print per-frame predicted action.")
    parser.add_argument("--server-host", type=str, default="*", help="ZMQ bind host, e.g. * or 127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5562, help="ZMQ REP server port")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    pipeline = rs.pipeline()
    config = rs.config()
    if args.device_id:
        config.enable_device(args.device_id)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    print("Initializing VLM...")
    vlm = VLMInference(
        model=args.model,
        model_id=None,
        device=args.device,
        precision=args.precision,
    )
    print(f"Visual prompt mode: {args.visual_prompt_mode}")
    print(f"Object init mode: {'RAM+VLM' if args.enable_ram else 'VLM-only'}")

    gripper_tracker: Optional[GripperBBoxTracker] = None
    if args.visual_prompt_mode == "gripper":
        gripper_tracker = GripperBBoxTracker(
            model_path=args.gripper_model_path,
            conf_threshold=args.gripper_conf_threshold,
            iou_threshold=args.gripper_iou_threshold,
        )
        print(f"Gripper tracker model: {gripper_tracker.model_path}")

    latest_state: Dict[str, Any] = {
        "has_action": False,
        "action": "",
        "frame_index": -1,
        "ts": 0.0,
        "object_set": [],
        "visual_prompt_mode": args.visual_prompt_mode,
        "gripper_center": None,
    }
    state_lock = threading.Lock()
    stop_event = threading.Event()
    server_thread, server_socket, _ = _start_intent_server(
        bind_host=args.server_host,
        port=args.server_port,
        latest_state=latest_state,
        state_lock=state_lock,
        stop_event=stop_event,
    )
    print(f"Intent server started at tcp://{args.server_host}:{args.server_port}")

    if args.debug:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    frame_history: Deque[np.ndarray] = deque(maxlen=max(2 * args.fps + 2, 4))
    gripper_history: Deque[Tuple[float, Tuple[int, int]]] = deque()
    debug_writer = None
    debug_video_path = None
    if args.debug:
        debug_writer, debug_video_path = None, None

    print("Starting camera stream...")
    pipeline.start(config)

    object_set: List[str] = []
    frame_idx = 0
    debug_writer = None
    try:
        while True:
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            if not color_frame:
                continue
            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            object_set, init_raw = _initialize_object_set(rgb, vlm, enable_ram=args.enable_ram)
            print(f"[Init] object_set={object_set}")
            if args.debug:
                (args.debug_dir / "object_set.txt").write_text(
                    ", ".join(object_set) if object_set else "none",
                    encoding="utf-8",
                )
                (args.debug_dir / "object_init_raw.txt").write_text(init_raw, encoding="utf-8")
                Image.fromarray(rgb).save(args.debug_dir / "init_frame.png")
            frame_history.append(rgb)
            break

        prompt = _compose_unified_prompt(object_set, args.visual_prompt_mode, args.gripper_track_seconds)
        print("Enter online inference loop. Press Ctrl+C to stop.")

        if args.debug:
            frame_size = (args.width, args.height)
            debug_writer, debug_video_path = _build_debug_video_writer(args.debug_dir, args.fps, frame_size)

        while True:
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            if not color_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_history.append(rgb)
            now_ts = time.time()

            if len(frame_history) < 3:
                continue

            one_sec_back = min(len(frame_history) - 1, max(1, args.fps))
            two_sec_back = min(len(frame_history) - 1, max(1, 2 * args.fps))
            current_rgb = frame_history[-1]

            if args.visual_prompt_mode == "diff":
                one_sec_rgb = frame_history[-1 - one_sec_back]
                two_sec_rgb = frame_history[-1 - two_sec_back]
                overlay = _build_diff_overlay(
                    current_rgb=current_rgb,
                    one_sec_rgb=one_sec_rgb,
                    two_sec_rgb=two_sec_rgb,
                    diff_threshold=args.diff_threshold,
                    min_diff_area=args.min_diff_area,
                )
                gripper_bbox = None
                gripper_center = None
            else:
                assert gripper_tracker is not None
                detection = gripper_tracker.detect(current_rgb)
                if detection is not None:
                    gripper_history.append((now_ts, detection.center))
                    gripper_bbox = detection.bbox
                    gripper_center = detection.center
                else:
                    gripper_bbox = None
                    gripper_center = None
                cutoff = now_ts - max(0.0, float(args.gripper_track_seconds))
                while gripper_history and gripper_history[0][0] < cutoff:
                    gripper_history.popleft()
                overlay = _build_gripper_overlay(
                    current_rgb=current_rgb,
                    tracker=gripper_tracker,
                    history_centers=[center for _, center in gripper_history],
                    detection=detection,
                )

            action = vlm.infer(overlay, prompt).strip()
            debug_frame_rgb = None
            if args.debug:
                caption_lines = [
                    f"frame={frame_idx:06d}",
                    f"mode={args.visual_prompt_mode}",
                    f"action={action if action else 'none'}",
                ]
                if args.visual_prompt_mode == "gripper":
                    bbox_text = "none" if gripper_bbox is None else f"({gripper_bbox[0]}, {gripper_bbox[1]}, {gripper_bbox[2]}, {gripper_bbox[3]})"
                    center_text = "none" if gripper_center is None else f"({gripper_center[0]}, {gripper_center[1]})"
                    caption_lines.append(f"bbox={bbox_text}")
                    caption_lines.append(f"center={center_text}")
                else:
                    caption_lines.append(f"objects={', '.join(object_set) if object_set else 'none'}")
                debug_frame_rgb = _draw_debug_caption(np.asarray(overlay), "\n".join(caption_lines))
                if debug_writer is not None:
                    debug_writer.write(cv2.cvtColor(debug_frame_rgb, cv2.COLOR_RGB2BGR))

            if action:
                ts = time.time()
                with state_lock:
                    latest_state["has_action"] = True
                    latest_state["action"] = action
                    latest_state["frame_index"] = frame_idx
                    latest_state["ts"] = float(ts)
                    latest_state["object_set"] = list(object_set)
                    latest_state["visual_prompt_mode"] = args.visual_prompt_mode
                    latest_state["gripper_center"] = gripper_center

                if args.output:
                    if args.visual_prompt_mode == "gripper":
                        center_text = "none" if gripper_center is None else f"({gripper_center[0]}, {gripper_center[1]})"
                        print(f"[{frame_idx:06d}] center={center_text} action={action}")
                    else:
                        print(f"[{frame_idx:06d}] action={action}")

            frame_idx += 1

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        stop_event.set()
        if server_socket is not None:
            server_socket.close(0)
        if server_thread is not None:
            server_thread.join(timeout=1.0)
        pipeline.stop()
        if debug_writer is not None:
            debug_writer.release()
            print(f"Saved debug video to: {debug_video_path}")
        if args.debug:
            print(f"Debug artifacts kept: only {debug_video_path.name if debug_video_path else 'debug video'}")
        print("Stopped.")


if __name__ == "__main__":
    main()
