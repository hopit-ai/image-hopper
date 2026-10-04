"""Local smoke test with a fake processor and model (CPU, no weights, no network).

Checks the request contract end to end through the real HTTP handler: native image
preparation, the one-prefill readout, the per-answer-type temperature, the response
shape, and refusal of malformed requests.  Run from the bundle root::

    python -m open_decisions.image_jev.release.smoke --calibration calibration.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path

LETTER_BASE = 1000
VOCABULARY = 1100


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [LETTER_BASE + ord(text) - ord("A")]


class FakeProcessor:
    """Shapes like the real processor's output; one token per prompt character."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs.get("enable_thinking") is False
        return "\n".join(message["content"] for message in messages)

    def __call__(self, *, text, images, padding, return_tensors):
        import torch

        ids = torch.tensor([[ord(char) % 256 for char in text[0]]], dtype=torch.long)
        pixels = torch.stack([
            torch.tensor(list(image.convert("L").resize((4, 4)).tobytes()), dtype=torch.float32)
            for image in images
        ])
        grid = torch.tensor([[1, 2, 2]] * len(images), dtype=torch.long)
        return {"input_ids": ids, "pixel_values": pixels, "image_grid_thw": grid}


def _fake_model():
    import torch

    class FakeModel(torch.nn.Module):
        """Letter k gets logit -0.5 k: option A always leads, by a fixed margin."""

        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, input_ids, pixel_values, image_grid_thw, use_cache=False,
                    return_dict=True, logits_to_keep=None, **_kwargs):
            assert logits_to_keep == 1 and use_cache is False
            self.calls += 1
            logits = torch.full((input_ids.shape[0], 1, VOCABULARY), -50.0)
            for index in range(26):
                logits[:, 0, LETTER_BASE + index] = -0.5 * index
            return _Output(logits=logits + self.anchor)

    return FakeModel().eval()


class _Output(dict):
    """Model output readable both as a mapping and by attribute, like transformers' outputs."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _png() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (90, 140, 200)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _expected(n_options: int, temperature: float) -> list[float]:
    weights = [math.exp(-0.5 * index / temperature) for index in range(n_options)]
    total = sum(weights)
    return [weight / total for weight in weights]


class _MemoryConnection:
    """Minimal socket facade for one BaseHTTPRequestHandler exchange."""

    def __init__(self, request: bytes):
        self.request = io.BytesIO(request)
        self.response = io.BytesIO()

    def makefile(self, mode, _buffering=None):
        return self.request if "r" in mode else self.response

    def sendall(self, value):
        self.response.write(value)

    def close(self):
        pass


def _exchange(handler_type, method: str, path: str, body=None) -> tuple[int, dict]:
    payload = b"" if body is None else json.dumps(body).encode("utf-8")
    headers = [f"{method} {path} HTTP/1.1", "Host: smoke"]
    if body is not None:
        headers += ["Content-Type: application/json", f"Content-Length: {len(payload)}"]
    request = ("\r\n".join(headers) + "\r\n\r\n").encode("ascii") + payload
    connection = _MemoryConnection(request)
    handler_type(connection, ("127.0.0.1", 1), object())
    head, raw_body = connection.response.getvalue().split(b"\r\n\r\n", 1)
    status = int(head.splitlines()[0].split()[1])
    return status, json.loads(raw_body)


def run(calibration_path: str | Path) -> dict:
    from open_decisions.image_jev import router
    from open_decisions.image_jev.release.predictor import (
        Predictor, RoutedCalibration, serving_calibration_from_artifact,
    )
    from open_decisions.image_jev.release.server import handler

    # The fake model validates the readout for either frozen-base or adapter bundles. The
    # bundle builder/server perform the real checkpoint-binding check.
    artifact = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    calibration = serving_calibration_from_artifact(
        artifact, checkpoint_sha256=None, adapter_sha256=None, base=None
    )
    model = _fake_model()
    predictor = Predictor(model, FakeProcessor(), calibration, model_name="smoke")
    handler_type = handler(predictor, {"model": "smoke"})

    def post(body) -> tuple[int, dict]:
        return _exchange(handler_type, "POST", "/v1/systemone", body)

    image = _png()
    checks = {}
    try:
        status, health = _exchange(handler_type, "GET", "/health")
        assert status == 200 and health == {"model": "smoke"}
        checks["health_shape"] = True

        status, body = post({
            "state": "", "images": [image],
            "questions": {
                "four": {"type": "choice", "instructions": "Which shelf is fullest?",
                         "criteria": {"top": "Top shelf", "middle": "Middle shelf",
                                      "bottom": "Bottom shelf", "none": "All equal"}},
                "three": {"type": "choice", "instructions": "Pick one.",
                          "criteria": ["red", "green", "blue"]},
                "binary": {"type": "noul", "instructions": "Is the item damaged?"},
            },
        })
        assert status == 200, body
        assert set(body) == {"model", "usage", "answers"} and body["model"] == "smoke"
        assert set(body["usage"]) == {"input_tokens", "output_tokens"}
        answers = body["answers"]
        assert set(answers) == {"four", "three", "binary"}
        assert set(answers["four"]) == {"type", "choice", "probabilities"}
        assert set(answers["binary"]) == {"type", "noul"}
        four = answers["four"]["probabilities"]
        assert list(four) == ["top", "middle", "bottom", "none"]
        assert answers["four"]["choice"] == "top"
        four_temperature = (
            calibration.temperature(4, router.DEFAULT)
            if isinstance(calibration, RoutedCalibration) else calibration.temperature(4)
        )
        for name, got, want in zip(four, four.values(), _expected(4, four_temperature)):
            assert abs(got - want) < 1e-6, (name, got, want)
        three = answers["three"]["probabilities"]
        for got, want in zip(three.values(), _expected(3, calibration.global_temperature)):
            assert abs(got - want) < 1e-6
        binary_temperature = (
            calibration.temperature(2, router.DEFAULT)
            if isinstance(calibration, RoutedCalibration) else calibration.temperature(2)
        )
        want_yes = _expected(2, binary_temperature)[0]
        assert abs(answers["binary"]["noul"] - want_yes) < 1e-6
        assert body["usage"]["output_tokens"] == 0 and body["usage"]["input_tokens"] > 0
        assert model.calls == 3
        checks["calibrated_probabilities"] = True
        checks["systemone_response_shape"] = True

        bare = image.split(",", 1)[1]
        status, body = post({"images": [bare], "questions": {"d": {
            "instructions": "Pick.", "criteria": {"a": "A", "b": "B"}}}})
        assert status == 200 and set(body["answers"]["d"]["probabilities"]) == {"a", "b"}
        checks["bare_base64_image"] = True

        for bad in ({"questions": {"d": {"instructions": "x", "criteria": ["a", "b"]}}},
                    {"images": [image], "questions": {"d": {"type": "score", "instructions": "x"}}},
                    {"images": [image], "questions": {"d": {"instructions": "x", "criteria": ["a"]}}}):
            status, body = post(bad)
            assert status == 400 and "error" in body, (bad, status, body)
        checks["malformed_requests_refused"] = True
    finally:
        pass
    return {"ok": True, "checks": checks,
            "calibration_identity_sha256": calibration.identity_sha256}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", default="calibration.json")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.calibration), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
