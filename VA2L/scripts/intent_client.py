from __future__ import annotations

import argparse
import json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intent client for VA2L ZMQ intent server.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5562)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument(
        "--command",
        type=str,
        default="get_intent",
        choices=["get_intent", "refine_instruction"],
        help="Server command.",
    )
    parser.add_argument("--instruction", type=str, default="", help="Instruction to refine.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    try:
        import zmq
    except ImportError as exc:
        raise ImportError("pyzmq is required for intent client. Install with: pip install pyzmq") from exc

    endpoint = f"tcp://{args.host}:{args.port}"
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.linger = 0
    sock.rcvtimeo = int(args.timeout_ms)
    sock.sndtimeo = int(args.timeout_ms)
    sock.connect(endpoint)

    try:
        if args.command == "refine_instruction":
            payload = {"command": args.command, "instruction": args.instruction.strip()}
            sock.send_multipart([args.command.encode("utf-8"), json.dumps(payload, ensure_ascii=False).encode("utf-8")])
        else:
            sock.send_string(args.command)
        reply = sock.recv_json()
        print(json.dumps(reply, ensure_ascii=False))
    finally:
        sock.close(0)


if __name__ == "__main__":
    main()
