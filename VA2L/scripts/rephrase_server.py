from __future__ import annotations

import argparse
import base64
import json
import os
import time
from typing import Any, Dict, List, Tuple

import zmq

from VA2L.lang_rephrase import (
	_parse_detected_objects,
	rephrase_instruction,
	DEFAULT_OBJECT_SET,
	DEFAULT_TARGET_SET,
)
from VA2L.vlm_inference import VLMInference


def _run_online_detected_objects(
	device_id: str,
	width: int,
	height: int,
	fps: int,
	model: str,
	timeout_sec: float,
) -> Tuple[List[str], List[str]]:
	import cv2
	import numpy as np
	import pyrealsense2 as rs
	from openai import OpenAI

	api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
	if not api_key:
		raise RuntimeError("DASHSCOPE_API_KEY is not set")

	pipeline = rs.pipeline()
	config = rs.config()
	if device_id:
		config.enable_device(device_id)
	config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

	frame_rgb = None
	try:
		pipeline.start(config)
		for _ in range(20):
			pipeline.wait_for_frames()

		frameset = pipeline.wait_for_frames()
		color_frame = frameset.get_color_frame()
		if not color_frame:
			raise RuntimeError("Failed to read color frame from camera")

		bgr = np.asanyarray(color_frame.get_data())
		frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
	finally:
		pipeline.stop()

	ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
	if not ok:
		raise RuntimeError("Failed to encode camera frame")
	image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

	prompt_text = (
		"You are a tabletop object detector for robot manipulation. "
		"Ignore robot arm/gripper/body, table, environment background, walls, floor, and room context. "
		"Output exactly two lines only. "
		"Line 1: comma-separated object descriptions with color, such as 'red mug, blue box'. "
		"Line 2: comma-separated target descriptions that can be interacted with, or none if there is no target set. "
		"If no valid object is visible, output exactly: none on line 1 and none on line 2"
	)

	client = OpenAI(
		api_key=api_key,
		base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
		timeout=max(1.0, float(timeout_sec)),
	)

	t_start = time.perf_counter()
	response = client.responses.create(
		model=model,
		input=[
			{
				"role": "user",
				"content": [
					{"type": "input_text", "text": prompt_text},
					{"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
				],
			}
		],
		extra_body={"enable_thinking": False},
	)
	elapsed = time.perf_counter() - t_start

	raw_text = "none"
	for item in getattr(response, "output", []):
		if getattr(item, "type", "") != "message":
			continue
		for content in getattr(item, "content", []):
			text = getattr(content, "text", None)
			if text:
				raw_text = str(text).strip()
				break
		if raw_text != "none":
			break

	lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
	if not lines:
		lines = ["none", "none"]
	elif len(lines) == 1:
		lines.append("none")

	object_set = _parse_detected_objects(lines[0])
	target_set = _parse_detected_objects(lines[1])
	print(f"[online-detected] elapsed={elapsed:.3f}s raw={raw_text}")
	return object_set, target_set


def _parse_request(req_parts: List[bytes]) -> Tuple[str, Dict[str, Any]]:
	cmd = ""
	payload: Dict[str, Any] = {}

	if len(req_parts) == 1:
		raw = req_parts[0].decode("utf-8", errors="ignore").strip()
		try:
			payload = json.loads(raw)
			cmd = str(payload.get("command", "")).strip().lower()
		except Exception:
			cmd = "rephrase"
			payload = {"instruction": raw}
	elif len(req_parts) >= 2:
		cmd = req_parts[0].decode("utf-8", errors="ignore").strip().lower()
		try:
			payload = json.loads(req_parts[1].decode("utf-8", errors="ignore"))
		except Exception:
			payload = {}

	return cmd, payload


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="ZMQ server for language rephrase.")
	parser.add_argument("--host", type=str, default="*", help="ZMQ bind host, e.g. * or 127.0.0.1")
	parser.add_argument("--port", type=int, default=5562, help="ZMQ REP bind port")
	parser.add_argument("--device", type=str, default="cuda:1", help="Model device")
	parser.add_argument(
		"--model",
		type=str,
		default="qwen-vl-4b",
		choices=["qwen-vl-4b", "qwen-vl-8b", "qwen-vl-2b"],
		help="Model to use for rephrase",
	)
	parser.add_argument(
		"--precision",
		type=str,
		default="auto",
		choices=["auto", "fp16", "bf16", "fp32"],
		help="Model precision",
	)
	parser.add_argument(
		"--object-set",
		type=str,
		default=", ".join(DEFAULT_OBJECT_SET),
		help="Default comma-separated object candidates used by rephrase",
	)
	parser.add_argument(
		"--target-set",
		type=str,
		default=", ".join(DEFAULT_TARGET_SET),
		help="Default comma-separated target candidates used by rephrase; can be empty",
	)
	parser.add_argument(
		"--online-detect",
		action="store_true",
		help="Use one-time camera + online VLM object detection at startup",
	)
	parser.add_argument("--online-object-model", type=str, default="qwen3.5-plus", help="Online VLM model name")
	parser.add_argument("--device-id", type=str, default="f1421698", help="RealSense device id")
	parser.add_argument("--width", type=int, default=1280, help="Camera width")
	parser.add_argument("--height", type=int, default=720, help="Camera height")
	parser.add_argument("--fps", type=int, default=30, help="Camera fps")
	parser.add_argument("--online-timeout-sec", type=float, default=20.0, help="Online VLM timeout in seconds")
	return parser


def main() -> None:
	args = _build_parser().parse_args()

	object_set = _parse_detected_objects(args.object_set)
	target_set = _parse_detected_objects(args.target_set)

	if args.online_detect:
		try:
			online_object_set, online_target_set = _run_online_detected_objects(
				device_id=args.device_id,
				width=args.width,
				height=args.height,
				fps=args.fps,
				model=args.online_object_model,
				timeout_sec=args.online_timeout_sec,
			)
			if online_object_set:
				object_set = online_object_set
			if online_target_set:
				target_set = online_target_set
		except Exception as exc:
			print(f"[online-detected] warning: {exc}. fallback to current defaults.")

	print("Initializing local VLM model for rephrase...")
	vlm = VLMInference(
		model=args.model,
		model_id=None,
		device=args.device,
		precision=args.precision,
	)
	print(f"Detected object set: {object_set}")
	print(f"Detected target set: {target_set}")

	endpoint = f"tcp://{args.host}:{args.port}"
	ctx = zmq.Context.instance()
	socket = ctx.socket(zmq.REP)
	socket.linger = 0
	socket.bind(endpoint)
	poller = zmq.Poller()
	poller.register(socket, zmq.POLLIN)
	print(f"Rephrase server started at {endpoint}")

	try:
		while True:
			events = dict(poller.poll(timeout=100))
			if socket not in events:
				continue

			req_parts = socket.recv_multipart()
			cmd, payload = _parse_request(req_parts)

			if cmd == "rephrase":
				instruction = str(payload.get("instruction", "")).strip()
				result = rephrase_instruction(
					instruction=instruction,
					object_set=object_set,
					target_set=target_set,
					model=args.model,
					device=args.device,
					precision=args.precision,
					vlm=vlm,
				)
				print(f"[rephrase] input={instruction if instruction else '<empty>'}")
				print(f"[rephrase] output={result}")
				socket.send_json({"ok": True, "result": result, "object_set": object_set, "target_set": target_set})
			elif cmd == "set_object_set":
				objects_text = payload.get("object_set", "")
				if isinstance(objects_text, list):
					merged = ", ".join(str(x) for x in objects_text)
				else:
					merged = str(objects_text)
				parsed = _parse_detected_objects(merged)
				if not parsed:
					socket.send_json({"ok": False, "message": "object_set is empty after parsing"})
					continue
				object_set = parsed
				socket.send_json({"ok": True, "object_set": object_set})
			elif cmd == "set_target_set":
				target_text = payload.get("target_set", "")
				if isinstance(target_text, list):
					merged = ", ".join(str(x) for x in target_text)
				else:
					merged = str(target_text)
				target_set = _parse_detected_objects(merged)
				socket.send_json({"ok": True, "target_set": target_set})
			elif cmd == "get_sets":
				socket.send_json({"ok": True, "object_set": object_set, "target_set": target_set})
			else:
				socket.send_json(
					{
						"ok": False,
						"message": f"unknown command: {cmd}",
						"supported": ["rephrase", "set_object_set", "set_target_set", "get_sets"],
					}
				)
	except KeyboardInterrupt:
		print("Interrupted by user.")
	finally:
		socket.close(0)
		print("Rephrase server stopped.")


if __name__ == "__main__":
	main()
