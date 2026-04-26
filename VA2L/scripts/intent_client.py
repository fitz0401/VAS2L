from __future__ import annotations

import json
from typing import Any


class IntentClient:
    """ZMQ client for VA2L intent server. Example usage:
        client = IntentClient(host="127.0.0.1", port=5562)
        intent = client.get_intent()
        refined = client.refine_instruction("put the red marker in the cup")
        client.close()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5562, timeout_ms: int = 3000) -> None:
        import zmq

        self._zmq = zmq
        self._endpoint = f"tcp://{host}:{port}"
        self._timeout_ms = int(timeout_ms)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.linger = 0
        self._sock.rcvtimeo = self._timeout_ms
        self._sock.sndtimeo = self._timeout_ms
        self._sock.connect(self._endpoint)
        self._closed = False

    def _request_json(self, command: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_open()
        if payload is None:
            self._sock.send_string(command)
        else:
            body = dict(payload)
            body["command"] = command
            self._sock.send_multipart(
                [
                    command.encode("utf-8"),
                    json.dumps(body, ensure_ascii=False).encode("utf-8"),
                ]
            )
        return self._sock.recv_json()

    def get_intent(self) -> dict[str, Any]:
        """Request latest intent from server."""
        return self._request_json("get_intent")

    def refine_instruction(self, instruction: str) -> dict[str, Any]:
        """Store a temporary instruction to be appended to the next VLM prompt."""
        return self._request_json(
            "refine_instruction",
            {
                "instruction": instruction.strip(),
            },
        )

    def close(self) -> None:
        """Close underlying socket."""
        if self._closed:
            return
        self._sock.close(0)
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("IntentClient is closed. Create a new client instance.")

    def __enter__(self) -> "IntentClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.close()


def main() -> None:
    # Minimal usage example for direct execution.
    with IntentClient() as client:
        # print(json.dumps(client.get_intent(), ensure_ascii=False, indent=2))
        print(json.dumps(client.refine_instruction("break the drawer"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
