from __future__ import annotations

import argparse
import base64
import json
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
import zmq

from VA2L.lang_rephrase import rephrase_instruction
from VA2L.utils.gripper_tracker import GripperBBoxTracker, GripperDetection
from VA2L.vlm_inference import VLMInference


_STARTUP_SKIP_FRAMES = 20
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


def _compose_gripper_prompt(
    gripper_track_seconds: float,
    detected_objects: List[str],
    possible_tasks: str = "",
) -> str:
    base_prompt = (
        f"Infer the task the robotic arm is performing from the image, in which the gripper's past {gripper_track_seconds:g}-second trajectory are visible.\n"
        "Format: <action> <target object> (<relation> <relation object>)\n"
        "Rule 1: choose the action from pick, place, insert, open, close, move, turn.\n"
        "Rule 2: relation and relation object are optional, relation can be on, into, from, next to.\n"
        "Rule 3: Output one short sentence only, starting with the action. No explanation.\n"
        "Rule 4: If the gripper is not holding an object, prioritize 'pick' action.\n"
    )
    if not detected_objects:
        return base_prompt

    prompt = base_prompt + f"Hint: detected tabletop objects: {', '.join(detected_objects)}.\n"
    if possible_tasks.strip():
        prompt += f"Hint: possible manipulation tasks by object affordance: {possible_tasks.strip()}\n"
    return prompt


def _normalize_detected_objects(text: str) -> List[str]:
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


def _refine_action(vlm: VLMInference, action: str) -> str:
    refined = rephrase_instruction(action, vlm=vlm)
    return refined.strip() if refined else action.strip()


def _run_online_object_detection(
    first_rgb: np.ndarray,
    model: str,
    timeout_sec: float,
) -> Tuple[List[str], str, str]:
    from openai import OpenAI
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set. Cannot run online object detection.")

    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode first frame for online object detection.")
    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    # DashScope compatible API uses OpenAI Responses; prompt enforces tabletop-only colored object list.
    prompt_text = (
        "You are a tabletop object detector for robot manipulation. "
        "Ignore robot arm/gripper/body, table, environment background, walls, floor, and room context. "
        "Focus only on manipulable tabletop objects. "
        "First, output one line only: comma-separated object descriptions with color, e.g., 'red mug, blue box'. "
        "Second, output one line only describing possible tasks that the robotics arm can complete according to object affordance. "
        "Use the format: <action> <target object> (<relation> <relation object>) . "
        "Choose the action from pick, place, insert, open, close, move, turn. "
        "Relation and relation object are optional; relation can be on, into, from, next to. "
        "If no valid tabletop objects are visible, output exactly: none"
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        timeout=max(1.0, float(timeout_sec)),
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt_text,
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                    },
                ],
            }
        ],
        extra_body={"enable_thinking": False},
    )

    raw_text = ""
    for item in getattr(response, "output", []):
        if getattr(item, "type", "") == "message":
            content = getattr(item, "content", [])
            if content and hasattr(content[0], "text"):
                raw_text = str(content[0].text).strip()
                break
    if not raw_text:
        raw_text = "none"

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        lines = ["none"]

    detected_objects = _normalize_detected_objects(lines[0])
    possible_tasks = ""
    if len(lines) > 1 and detected_objects:
        possible_tasks = " ".join(lines[1:]).strip()

    return detected_objects, possible_tasks, raw_text


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
    vlm: VLMInference,
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
                req = socket.recv_multipart()
            except Exception:
                continue

            cmd = ""
            payload: Dict[str, Any] = {}
            if len(req) == 1:
                raw_req = req[0].decode("utf-8", errors="ignore").strip()
                try:
                    payload = json.loads(raw_req)
                    cmd = str(payload.get("command", "")).strip().lower()
                except Exception:
                    cmd = raw_req.strip().lower()
            elif len(req) >= 2:
                cmd = req[0].decode("utf-8", errors="ignore").strip().lower()
                try:
                    payload = json.loads(req[1].decode("utf-8", errors="ignore"))
                except Exception:
                    payload = {}

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
            elif cmd == "refine_instruction":
                instruction = str(payload.get("instruction", "")).strip()
                if not instruction:
                    socket.send_json({"ok": False, "message": "instruction is required"})
                    continue

                refined = _refine_action(vlm, instruction)
                socket.send_json({"ok": True, "instruction": refined})
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
    parser = argparse.ArgumentParser(description="Real-world intent detection with gripper visual prompting.")
    parser.add_argument("--device-id", type=str, default="f1421698")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model", type=str, default="qwen-vl-4b")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--precision", type=str, default="auto")
    parser.add_argument("--enable_refine", action="store_true", help="Refine each predicted intent with lang rephrase.")
    parser.add_argument("--gripper-model-path", type=Path, default=Path("ckpts/yolo_best.pt"))
    parser.add_argument("--gripper-conf-threshold", type=float, default=0.25)
    parser.add_argument("--gripper-iou-threshold", type=float, default=0.45)
    parser.add_argument("--gripper-track-seconds", type=float, default=2.0, help="History window used in gripper tracking.")
    parser.add_argument("--enable-online-vlm", action="store_true", help="Enable one-time online VLM object detection after startup skip.")
    parser.add_argument("--online-object-model", type=str, default="qwen3.5-plus", help="Online VLM model name for first-frame object detection.")
    parser.add_argument("--online-object-save-path", type=Path, default=Path("dataset/realworld/debug_intent/online_detected_objects.txt"), help="Path to save online detected objects raw output.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=Path("dataset/realworld/debug_intent"))
    parser.add_argument("--output", action="store_true", dest="output", help="If set, print per-frame predicted action.")
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
    print("Visual prompt mode: gripper")

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
        "visual_prompt_mode": "gripper",
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
        vlm=vlm,
    )
    print(f"Intent server started at tcp://{args.server_host}:{args.server_port}")

    if args.debug:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    gripper_history: Deque[Tuple[float, Tuple[int, int]]] = deque()
    debug_writer: Optional[cv2.VideoWriter] = None
    debug_video_path: Optional[Path] = None

    print("Starting camera stream...")
    pipeline.start(config)

    frame_idx = 0
    detected_objects: List[str] = []
    possible_tasks = ""
    prompt = _compose_gripper_prompt(args.gripper_track_seconds, detected_objects, possible_tasks)
    online_detection_done = not bool(args.enable_online_vlm)
    try:
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
            now_ts = time.time()

            if frame_idx < _STARTUP_SKIP_FRAMES:
                frame_idx += 1
                continue

            if not online_detection_done:
                args.debug_dir.mkdir(parents=True, exist_ok=True)
                init_frame_path = args.debug_dir / "init_frame.png"
                Image.fromarray(rgb).save(init_frame_path)

                t_start = time.perf_counter()
                detected_objects, possible_tasks, raw_detected_text = _run_online_object_detection(
                    first_rgb=rgb,
                    model=args.online_object_model,
                    timeout_sec=20.0,
                )

                elapsed = time.perf_counter() - t_start
                print(f"[online vlm object detection] elapsed={elapsed:.3f}s")
                print(f"[Detected objects] {detected_objects if detected_objects else ['none']}")
                print(f"Saved init frame to: {init_frame_path}")
                if possible_tasks:
                    print(f"[Possible tasks] {possible_tasks}")

                args.online_object_save_path.parent.mkdir(parents=True, exist_ok=True)
                save_lines = [
                    f"objects_raw: {raw_detected_text}",
                    f"objects_norm: {', '.join(detected_objects) if detected_objects else 'none'}",
                ]
                save_lines.append(f"possible_tasks: {possible_tasks if possible_tasks else 'none'}")
                args.online_object_save_path.write_text("\n".join(save_lines) + "\n", encoding="utf-8")
                print(f"Saved object metadata to: {args.online_object_save_path}")

                prompt = _compose_gripper_prompt(
                    args.gripper_track_seconds,
                    detected_objects,
                    possible_tasks,
                )
                online_detection_done = True

            detection = gripper_tracker.detect(rgb)
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
                current_rgb=rgb,
                tracker=gripper_tracker,
                history_centers=[center for _, center in gripper_history],
                detection=detection,
            )

            action = vlm.infer(overlay, prompt).strip()
            if args.enable_refine and action:
                action = _refine_action(vlm, action)
            if args.debug:
                caption_lines = [
                    f"frame={frame_idx:06d}",
                    "mode=gripper",
                    f"action={action if action else 'none'}",
                ]
                caption_lines.append(f"objects={', '.join(detected_objects) if detected_objects else 'none'}")
                bbox_text = "none" if gripper_bbox is None else f"({gripper_bbox[0]}, {gripper_bbox[1]}, {gripper_bbox[2]}, {gripper_bbox[3]})"
                center_text = "none" if gripper_center is None else f"({gripper_center[0]}, {gripper_center[1]})"
                caption_lines.append(f"bbox={bbox_text}")
                caption_lines.append(f"center={center_text}")
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
                    latest_state["object_set"] = []
                    latest_state["visual_prompt_mode"] = "gripper"
                    latest_state["gripper_center"] = gripper_center

                if args.output:
                    center_text = "none" if gripper_center is None else f"({gripper_center[0]}, {gripper_center[1]})"
                    print(f"[{frame_idx:06d}] center={center_text} action={action}")

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
