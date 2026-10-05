"""HTTP server for the frozen-base image decision release.

  POST /v1/systemone   TypeSafe-style request (see ``predictor``), JSON response
  GET  /health         model, pinned base revision and calibration identity

Single-threaded on purpose: one GPU, one request at a time, as published image
submissions are served.  Any ``Authorization`` header is ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Mapping

from open_decisions.image_jev.release.predictor import CONTEXT_POLICIES, RequestError

MAX_BODY_BYTES = 128 * 1024 * 1024
DEFAULT_NAME = "image-hopper"


def handler(predict: Callable[[Any], dict], health: Mapping[str, Any], *, phase_timing=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet; the harness measures latency itself
            pass

        def _send(self, status: int, value: Mapping[str, Any]) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, dict(health))
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if not 0 < length <= MAX_BODY_BYTES:
                self._send(400, {"error": "missing or oversized request body"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                if phase_timing:
                    seconds = {}
                    result = predict(body, phase_seconds=seconds)
                    result["phase_seconds"] = seconds
                else:
                    result = predict(body)
                self._send(200, result)
            except (json.JSONDecodeError, RequestError) as error:
                self._send(400, {"error": str(error)})
            except Exception as error:  # noqa: BLE001 - report, keep serving
                self._send(500, {"error": f"{type(error).__name__}: {error}"})

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", default="calibration.json")
    parser.add_argument("--adapter", default=None,
                        help="optional local adapter directory; calibration must bind its tree hash")
    parser.add_argument("--adapters", default=None,
                        help="router v3: directory with default/ and screen_geometry/ adapters; "
                             "calibration must be router-calibration/v2 bound to both")
    parser.add_argument("--serving-form", choices=("U", "M"), default="U",
                        help="router v3: U keeps both adapters attached (set_adapter on route "
                             "change); M merges default and serves an exact difference adapter")
    parser.add_argument("--base-hashes", default="base-model-files.json",
                        help="hash listing of the pinned base snapshot; '' skips the check")
    parser.add_argument("--model-cache", default=None, help="Hugging Face cache directory")
    parser.add_argument("--offline", action="store_true", help="never download; use the cache")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-policy", choices=CONTEXT_POLICIES, default="prepend",
                        help="how request 'state' text reaches the prompt")
    parser.add_argument("--max-input-tokens", type=int, default=None,
                        help="refuse (HTTP 400) requests above this many processed tokens")
    parser.add_argument("--allow-slow-path", action="store_true")
    parser.add_argument("--serving-options", default="",
                        help="comma-separated release.fastpath options (default: none, the "
                             "published path); recorded in /health")
    parser.add_argument("--phase-timing", action="store_true",
                        help="diagnostic synchronized phase_seconds in responses; default off")
    return parser


def main(argv=None) -> int:
    from open_decisions.image_jev.release.predictor import load_predictor

    args = build_parser().parse_args(argv)
    predictor, provenance = load_predictor(
        calibration_path=args.calibration, model_name=args.name, model_cache=args.model_cache,
        adapter_path=args.adapter, adapters_path=args.adapters, serving_form=args.serving_form,
        offline=args.offline, base_hashes_path=args.base_hashes or None, device=args.device,
        context_policy=args.context_policy, max_input_tokens=args.max_input_tokens,
        allow_slow_path=args.allow_slow_path, serving_options=args.serving_options or None,
    )
    health = {"model": args.name, **provenance}
    print(json.dumps(health, indent=2, sort_keys=True), file=sys.stderr)
    server = HTTPServer((args.host, args.port), handler(predictor, health, phase_timing=args.phase_timing))
    print(f"serving {args.name} on http://{args.host}:{args.port}/v1/systemone", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
