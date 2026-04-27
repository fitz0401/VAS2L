from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class RephraseResult:
	request_id: int
	instruction: str
	result: str
	ok: bool
	error: str = ""


class RephraseClient:
	"""Async ZMQ client for rephrase server.

	`request_rephrase` is non-blocking: it schedules work and returns immediately.
	The latest result is written to `latest_result` when the server reply arrives.
	"""

	def __init__(self, host: str = "127.0.0.1", port: int = 5562, timeout_ms: int = 8000) -> None:
		self._host = host
		self._port = int(port)
		self._timeout_ms = int(timeout_ms)

		self._queue: "queue.Queue[tuple[int, str]]" = queue.Queue()
		self._lock = threading.Lock()
		self._stop_event = threading.Event()

		self._next_request_id = 1
		self._pending_request_id: Optional[int] = None
		self.latest_result: Optional[RephraseResult] = None

		self._worker = threading.Thread(target=self._worker_loop, daemon=True)
		self._worker.start()

	def request_rephrase(self, instruction: str) -> int:
		"""Send a rephrase request asynchronously and return request_id immediately."""
		clean_instruction = instruction.strip()
		with self._lock:
			request_id = self._next_request_id
			self._next_request_id += 1
			self._pending_request_id = request_id
		self._queue.put((request_id, clean_instruction))
		return request_id

	@property
	def pending(self) -> bool:
		with self._lock:
			return self._pending_request_id is not None

	def _worker_loop(self) -> None:
		import zmq

		ctx = zmq.Context.instance()
		sock = ctx.socket(zmq.REQ)
		sock.linger = 0
		sock.rcvtimeo = self._timeout_ms
		sock.sndtimeo = self._timeout_ms
		sock.connect(f"tcp://{self._host}:{self._port}")

		try:
			while not self._stop_event.is_set():
				try:
					request_id, instruction = self._queue.get(timeout=0.1)
				except queue.Empty:
					continue

				ok = False
				result = "none"
				error = ""

				try:
					sock.send_json({"command": "rephrase", "instruction": instruction})
					reply = sock.recv_json()
					ok = bool(reply.get("ok", False))
					if ok:
						result = str(reply.get("result", "none")).strip() or "none"
					else:
						error = str(reply.get("message", "unknown error"))
				except Exception as exc:
					error = str(exc)

				with self._lock:
					self.latest_result = RephraseResult(
						request_id=request_id,
						instruction=instruction,
						result=result,
						ok=ok,
						error=error,
					)
					if self._pending_request_id == request_id:
						self._pending_request_id = None
		finally:
			sock.close(0)

	def close(self) -> None:
		self._stop_event.set()
		self._worker.join(timeout=1.0)

	def __enter__(self) -> "RephraseClient":
		return self

	def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
		self.close()


def main() -> None:
	with RephraseClient(host="127.0.0.1", port=5562, timeout_ms=8000) as client:
		request_id = client.request_rephrase("put the gray in the jar")
		print(f"sent request_id={request_id}")

		# Non-blocking behavior: request is sent, caller can continue immediately.
		print(f"before update latest_result={client.latest_result}")

		while client.pending:
			time.sleep(0.05)

		print(f"after update latest_result={client.latest_result}")


if __name__ == "__main__":
	main()
