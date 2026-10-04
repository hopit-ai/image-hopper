"""Request contract, calibration and readout for Image Hopper releases.

The model path is exactly the one evaluated in stage 2 (``deploy/modal_image_jev.py``
``TorchPredictor`` on the native arm): :func:`vision.prepare_native` (processor-default
resolution, no pixel budget), one prefill with a last-position vocabulary projection
(:func:`vision.last_position_logits`), an FP32 softmax over the displayed option letters, and
a per-answer-type (or route x answer-type) temperature from the bound calibration artifact.
The routed system keeps its LoRA attached and enables it only for the registered text cues.
Temperature scaling never changes the selected option.

Wire format (TypeSafe ``/v1/systemone`` style, as used by published image submissions)::

    {"state": <string or object, optional>,
     "images": ["data:image/png;base64,...", ...],          # or bare base64
     "questions": {"decision": {"type": "choice", "instructions": "...",
                                "criteria": {"<name>": "<description>", ...}}}}

Response::

    {"model": NAME, "usage": {"input_tokens": n, "output_tokens": 0},
     "answers": {"decision": {"type": "choice", "choice": "<name>",
                              "probabilities": {"<name>": p, ...}}}}

``noul`` (yes/no) questions answer ``{"type": "noul", "noul": P(yes)}``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from open_decisions.image_jev import router, vision
from open_decisions.image_jev.speed import AdapterSwitch, NativeCache, PhaseTimer


BASE_REPO = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
IMAGE_ROUTE = "native"
CALIBRATION_VERSION = "image-jev/calibration/v2"
# Same buckets as train.calibration.answer_type for requests without an explicit answer type
# (two options -> yes/no, four -> 4-option, five -> 5-marker); other counts use the global
# temperature.  Every evaluated item had 2, 4 or 5 options.
TYPE_BUCKET_BY_OPTION_COUNT = {2: "yes/no", 4: "4-option", 5: "5-marker"}
MAX_IMAGES = 4
NOUL_OPTIONS = ({"name": "yes", "description": "Yes."}, {"name": "no", "description": "No."})
CONTEXT_POLICIES = ("prepend", "ignore")
_CALIBRATION_FIELDS = {
    "artifact_sha256", "checkpoint_sha256", "global_temperature", "image_route",
    "image_route_config_sha256", "regularization", "routing_config_sha256", "type_counts",
    "type_temperatures", "validation", "version",
}
_ROUTED_CALIBRATION_FIELDS = {
    "adapter_sha256", "artifact_sha256", "base", "global_temperature", "image_route",
    "image_route_config_sha256", "regularization", "route_type_counts",
    "route_type_temperatures", "rules_sha256", "rules_version", "version",
}


class RequestError(ValueError):
    """A malformed or unsupported request (HTTP 400)."""


def _compact(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _temperature(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class Calibration:
    """The route-bound temperature map (``image-jev/calibration/v2``) as served."""

    global_temperature: float
    type_temperatures: Mapping[str, float]
    artifact_sha256: str
    identity_sha256: str

    @classmethod
    def from_artifact(cls, value: Mapping[str, Any], *,
                      checkpoint_sha256: str | None = "") -> "Calibration":
        if not isinstance(value, Mapping) or set(value) != _CALIBRATION_FIELDS:
            raise ValueError("calibration artifact fields differ from image-jev/calibration/v2")
        body = {key: item for key, item in value.items() if key != "artifact_sha256"}
        if value["artifact_sha256"] != hashlib.sha256(_compact(body)).hexdigest():
            raise ValueError("calibration artifact hash differs")
        if value["version"] != CALIBRATION_VERSION:
            raise ValueError("calibration artifact version differs")
        if value["image_route"] != IMAGE_ROUTE:
            raise ValueError("calibration was not fitted on the native image route")
        if value["image_route_config_sha256"] != vision.IMAGE_ROUTE_CONFIG_SHA256[IMAGE_ROUTE]:
            raise ValueError("calibration image-route configuration differs")
        if value["routing_config_sha256"] != vision.ROUTING_CONFIG_SHA256:
            raise ValueError("calibration routing configuration differs")
        checkpoint = value["checkpoint_sha256"]
        if not isinstance(checkpoint, str) or (checkpoint and
                                               (len(checkpoint) != 64 or any(
                                                   char not in "0123456789abcdef"
                                                   for char in checkpoint))):
            raise ValueError("calibration checkpoint_sha256 must be empty or a lowercase SHA-256")
        if checkpoint_sha256 is not None and checkpoint != checkpoint_sha256:
            target = "frozen base" if checkpoint_sha256 == "" else "selected adapter"
            raise ValueError(f"calibration is not bound to the {target}")
        temperatures = value["type_temperatures"]
        if not isinstance(temperatures, Mapping):
            raise ValueError("type_temperatures must be an object")
        return cls(
            global_temperature=_temperature(value["global_temperature"], "global_temperature"),
            type_temperatures={
                str(key): _temperature(item, f"type_temperatures[{key!r}]")
                for key, item in temperatures.items()
            },
            artifact_sha256=str(value["artifact_sha256"]),
            identity_sha256=hashlib.sha256(_compact(dict(value))).hexdigest(),
        )

    @classmethod
    def load(cls, path: str | Path, *, checkpoint_sha256: str | None = "") -> "Calibration":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load calibration {path}: {error}") from error
        return cls.from_artifact(value, checkpoint_sha256=checkpoint_sha256)

    def temperature(self, n_options: int) -> float:
        bucket = TYPE_BUCKET_BY_OPTION_COUNT.get(int(n_options))
        if bucket is None:
            return self.global_temperature
        return self.type_temperatures.get(bucket, self.global_temperature)


@dataclass(frozen=True)
class RoutedCalibration:
    """Serving view of ``image-jev/router-calibration/v1``."""

    global_temperature: float
    route_type_temperatures: Mapping[str, Mapping[str, float]]
    artifact_sha256: str
    identity_sha256: str
    adapter_sha256: str
    base: Mapping[str, str]
    rules_sha256: str

    @classmethod
    def from_artifact(
        cls, value: Mapping[str, Any], *, adapter_sha256: str | None = None,
        base: Mapping[str, str] | None = None,
    ) -> "RoutedCalibration":
        if not isinstance(value, Mapping) or set(value) != _ROUTED_CALIBRATION_FIELDS:
            raise ValueError("routed calibration artifact fields differ")
        body = {key: item for key, item in value.items() if key != "artifact_sha256"}
        if value["artifact_sha256"] != hashlib.sha256(_compact(body)).hexdigest():
            raise ValueError("routed calibration artifact hash differs")
        if value["version"] != "image-jev/router-calibration/v1":
            raise ValueError("routed calibration artifact version differs")
        if value["rules_version"] != router.RULES_VERSION \
                or value["rules_sha256"] != router.RULES_SHA256:
            raise ValueError("routed calibration rules binding differs")
        if value["image_route"] != IMAGE_ROUTE \
                or value["image_route_config_sha256"] != vision.IMAGE_ROUTE_CONFIG_SHA256[IMAGE_ROUTE]:
            raise ValueError("routed calibration image-route binding differs")
        if adapter_sha256 is not None and value["adapter_sha256"] != adapter_sha256:
            raise ValueError("routed calibration adapter binding differs")
        if base is not None and value["base"] != dict(base):
            raise ValueError("routed calibration base binding differs")
        if (
            not isinstance(value["adapter_sha256"], str)
            or len(value["adapter_sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in value["adapter_sha256"])
        ):
            raise ValueError("routed calibration adapter hash is invalid")
        if not isinstance(value["base"], Mapping) or set(value["base"]) != {"repo", "revision"}:
            raise ValueError("routed calibration base pin differs")
        route_temperatures = value["route_type_temperatures"]
        route_counts = value["route_type_counts"]
        if not isinstance(route_temperatures, Mapping) or set(route_temperatures) != set(router.ROUTES):
            raise ValueError("routed calibration routes differ")
        if not isinstance(route_counts, Mapping) or set(route_counts) != set(router.ROUTES):
            raise ValueError("routed calibration count routes differ")
        checked = {}
        for route_name in router.ROUTES:
            values = route_temperatures[route_name]
            counts = route_counts[route_name]
            if (
                not isinstance(values, Mapping)
                or set(values) != set(TYPE_BUCKET_BY_OPTION_COUNT.values())
                or not isinstance(counts, Mapping)
                or set(counts) != set(TYPE_BUCKET_BY_OPTION_COUNT.values())
            ):
                raise ValueError("routed calibration temperatures must be objects")
            if any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in counts.values()
            ):
                raise ValueError("routed calibration counts are invalid")
            checked[route_name] = {
                str(kind): _temperature(item, f"route_type_temperatures[{route_name}][{kind}]")
                for kind, item in values.items()
            }
        return cls(
            global_temperature=_temperature(value["global_temperature"], "global_temperature"),
            route_type_temperatures=checked,
            artifact_sha256=str(value["artifact_sha256"]),
            identity_sha256=hashlib.sha256(_compact(dict(value))).hexdigest(),
            adapter_sha256=str(value["adapter_sha256"]), base=dict(value["base"]),
            rules_sha256=str(value["rules_sha256"]),
        )

    def temperature(self, n_options: int, route_name: str) -> float:
        if route_name not in router.ROUTES:
            raise ValueError("unknown routed calibration route")
        bucket = TYPE_BUCKET_BY_OPTION_COUNT.get(int(n_options))
        if bucket is None:
            return self.global_temperature
        return self.route_type_temperatures[route_name].get(bucket, self.global_temperature)


def serving_calibration_from_artifact(
    value: Mapping[str, Any], *, checkpoint_sha256: str | None = "",
    adapter_sha256: str | None = None, base: Mapping[str, str] | None = None,
) -> Calibration | RoutedCalibration:
    if not isinstance(value, Mapping):
        raise ValueError("calibration artifact must be an object")
    if value.get("version") == "image-jev/router-calibration/v1":
        return RoutedCalibration.from_artifact(
            value, adapter_sha256=adapter_sha256, base=base
        )
    return Calibration.from_artifact(value, checkpoint_sha256=checkpoint_sha256)


@dataclass(frozen=True)
class Question:
    qid: str
    kind: str
    instructions: str
    options: tuple[dict[str, str], ...]
    route_instructions: str | None = None


def _image_source(value: Any, index: int):
    if not isinstance(value, str) or not value:
        raise RequestError(f"image {index} must be a non-empty data URI or base64 string")
    if value.startswith("data:"):
        return value
    if (len(value) // 4) * 3 > vision.MAX_IMAGE_BYTES + 2:
        raise RequestError(f"image {index} exceeds {vision.MAX_IMAGE_BYTES} bytes")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise RequestError(f"image {index} is neither a data URI nor valid base64") from None


def _images(body: Mapping[str, Any]) -> list:
    raw = body.get("images")
    state = body.get("state")
    if raw is None and isinstance(state, Mapping):
        raw = state.get("images", state.get("image"))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise RequestError("request needs at least one image in 'images'")
    if len(raw) > MAX_IMAGES:
        raise RequestError(f"at most {MAX_IMAGES} images per request")
    return [_image_source(value, index) for index, value in enumerate(raw)]


def _context(state: Any) -> str | None:
    if state is None:
        return None
    if isinstance(state, str):
        return state.strip() or None
    if isinstance(state, Mapping):
        rest = {key: value for key, value in state.items() if key not in {"image", "images"}}
        return json.dumps(rest, ensure_ascii=False, sort_keys=True) if rest else None
    raise RequestError("state must be a string or an object")


def _options(criteria: Any) -> tuple[dict[str, str], ...]:
    if isinstance(criteria, Mapping):
        values = [{"name": str(name), "description": str(text)} for name, text in criteria.items()]
    elif isinstance(criteria, list):
        values = []
        for index, option in enumerate(criteria):
            if isinstance(option, str):
                values.append({"name": option, "description": option})
            elif isinstance(option, Mapping) and "name" in option:
                values.append({"name": str(option["name"]),
                               "description": str(option.get("description", option["name"]))})
            else:
                raise RequestError(f"criteria[{index}] must be text or an object with a name")
    else:
        raise RequestError("choice criteria must be an object or a list")
    names = [value["name"] for value in values]
    if not 2 <= len(values) <= len(vision.hopper_prompt.LETTERS):
        raise RequestError("choice questions need 2 to 26 options")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise RequestError("option names must be unique and non-empty")
    return tuple(values)


def parse_request(body: Any, *, context_policy: str = "prepend") -> tuple[list, list[Question]]:
    """Validate a ``/v1/systemone`` body into images and questions (no model work)."""
    if context_policy not in CONTEXT_POLICIES:
        raise ValueError(f"context_policy must be one of {CONTEXT_POLICIES}")
    if not isinstance(body, Mapping):
        raise RequestError("request body must be a JSON object")
    questions = body.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise RequestError("request needs a non-empty 'questions' object")
    images = _images(body)
    context = _context(body.get("state")) if context_policy == "prepend" else None
    parsed = []
    for qid, spec in questions.items():
        if not isinstance(spec, Mapping):
            raise RequestError(f"question {qid!r} must be an object")
        kind = spec.get("type", "choice")
        instructions = spec.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise RequestError(f"question {qid!r} needs non-empty instructions")
        if context:
            instructions = f"{context}\n\n{instructions}"
        if kind == "choice":
            options = _options(spec.get("criteria"))
        elif kind == "noul":
            options = NOUL_OPTIONS
        else:
            raise RequestError(f"question {qid!r}: unsupported type {kind!r} (choice or noul)")
        parsed.append(Question(str(qid), kind, instructions, options, str(spec["instructions"])))
    return images, parsed


def _length(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return int(shape[-1])
    first = value[0] if value and isinstance(value[0], (list, tuple)) else value
    return len(first)


class Predictor:
    """One prefill per question; calibrated option probabilities in the board's shape."""

    def __init__(self, model, processor, calibration: Calibration | RoutedCalibration, *, model_name: str,
                 context_policy: str = "prepend", max_input_tokens: int | None = None,
                 inference_context: Callable[[], Any] | None = None,
                 adapter_context: Callable[[str], Any] | None = None,
                 synchronize: Callable[[], Any] | None = None, forward_call=None) -> None:
        if context_policy not in CONTEXT_POLICIES:
            raise ValueError(f"context_policy must be one of {CONTEXT_POLICIES}")
        self.model, self.processor, self.calibration = model, processor, calibration
        self.model_name = model_name
        self.context_policy = context_policy
        self.max_input_tokens = max_input_tokens
        self._inference_context = inference_context
        self._adapter_context = adapter_context
        self._synchronize = synchronize
        self._forward_call = forward_call
        self._cache = NativeCache(processor)
        try:
            self.device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            self.device = None

    def _read(self, images: Sequence, question: Question, *, loaded_images=None, timer=None) -> tuple[list[float], int]:
        example = {"question": question.instructions, "options": list(question.options),
                   "images": list(images)}
        try:
            prepared = vision.prepare_native(example, self.processor, loaded_images=loaded_images,
                                             cache=self._cache, timer=timer, device=self.device)
        except (TypeError, ValueError) as error:
            raise RequestError(str(error)) from error
        tokens = _length(prepared.input_ids)
        if self.max_input_tokens is not None and tokens > self.max_input_tokens:
            raise RequestError(
                f"processed request has {tokens} tokens, above the configured "
                f"{self.max_input_tokens}-token limit"
            )
        with timer.phase("route_adapter_switch"):
            route_name = None
            if isinstance(self.calibration, RoutedCalibration):
                route_name = router.route(
                    question.route_instructions or question.instructions, question.options
                )
                temperature = self.calibration.temperature(len(question.options), route_name)
            else:
                temperature = self.calibration.temperature(len(question.options))
            route_context = (
                self._adapter_context(route_name) if self._adapter_context is not None
                else nullcontext()
            )
        inference_context = (
            self._inference_context() if self._inference_context is not None else nullcontext()
        )
        with route_context:
            with inference_context:
                logits = vision.last_position_logits(
                    self.model, prepared, timer=timer, forward_call=self._forward_call)
                with timer.phase("readout_calibration"):
                    probabilities = vision.option_probs(
                        logits, prepared.letter_token_ids, temperature=temperature)
        with timer.phase("readout_calibration"):
            total = math.fsum(probabilities)
            return [value / total for value in probabilities], tokens

    def __call__(self, body: Any, *, phase_seconds=None) -> dict[str, Any]:
        timer = PhaseTimer(phase_seconds, self._synchronize)
        with timer.phase("decode_load"):
            images, questions = parse_request(body, context_policy=self.context_policy)
        return self.predict_questions(images, questions, timer=timer)

    def predict_questions(self, images, questions, *, timer=None) -> dict[str, Any]:
        timer = timer or PhaseTimer()
        with timer.phase("decode_load"):
            try:
                loaded = [vision._load_image(source, exif_transpose=False) for source in images]
                loaded = [image if image.mode == "RGB" else image.convert("RGB") for image in loaded]
            except (TypeError, ValueError) as error:
                raise RequestError(str(error)) from error
        answers, input_tokens = {}, 0
        for question in questions:
            probabilities, tokens = self._read(images, question, loaded_images=loaded, timer=timer)
            with timer.phase("response_build"):
                input_tokens += tokens
                names = [option["name"] for option in question.options]
                # Exact ties go to the first option, as the harness's scorer breaks them.
                top = max(range(len(names)), key=lambda index: (probabilities[index], -index))
                if question.kind == "noul":
                    answers[question.qid] = {"type": "noul", "noul": probabilities[0]}
                else:
                    answers[question.qid] = {
                        "type": "choice", "choice": names[top],
                        "probabilities": dict(zip(names, probabilities, strict=True)),
                    }
        with timer.phase("response_build"):
            return {"model": self.model_name,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                    "answers": answers}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: str | Path) -> str:
    """Hash a local adapter as sorted relative paths and file-byte hashes."""
    root = Path(path)
    rows = [
        {"path": item.relative_to(root).as_posix(), "sha256": file_sha256(item)}
        for item in sorted(root.rglob("*")) if item.is_file()
    ]
    if not rows:
        raise ValueError("adapter directory is empty")
    return hashlib.sha256(_compact(rows)).hexdigest()


def verify_snapshot(snapshot: str | Path, expected: Mapping[str, str]) -> dict[str, str]:
    """Hash every file listed for the pinned base revision; refuse any difference."""
    root = Path(snapshot)
    failures = []
    for name, digest in sorted(expected.items()):
        path = root / name
        if not path.is_file():
            failures.append(f"missing {name}")
        elif file_sha256(path) != digest:
            failures.append(f"hash differs: {name}")
    if failures:
        raise RuntimeError("base model snapshot differs from the pinned revision: "
                           + "; ".join(failures))
    return dict(expected)


def check_fast_path() -> None:
    """Qwen3.5's linear-attention layers need these kernels; the fallback is far slower."""
    missing = []
    for module in ("fla", "causal_conv1d"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "fast linear-attention kernels unavailable (" + ", ".join(missing) + "); install "
            "flash-linear-attention and causal-conv1d from requirements.txt, or pass "
            "--allow-slow-path"
        )


def load_predictor(*, calibration_path: str | Path, model_name: str,
                   model_cache: str | None = None, offline: bool = False,
                   adapter_path: str | Path | None = None,
                   base_hashes_path: str | Path | None = None, device: str = "cuda",
                   context_policy: str = "prepend", max_input_tokens: int | None = None,
                   allow_slow_path: bool = False) -> tuple[Predictor, dict[str, Any]]:
    """Load the pinned frozen base in bf16 and return the predictor and its provenance."""
    import platform

    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor

    if not allow_slow_path:
        check_fast_path()
    adapter = None if adapter_path is None else Path(adapter_path)
    adapter_sha256 = "" if adapter is None else tree_sha256(adapter)
    try:
        calibration_value = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load calibration {calibration_path}: {error}") from error
    base_pin = {"repo": BASE_REPO, "revision": BASE_REVISION}
    calibration = serving_calibration_from_artifact(
        calibration_value, checkpoint_sha256=adapter_sha256,
        adapter_sha256=adapter_sha256 or None, base=base_pin,
    )
    routed = isinstance(calibration, RoutedCalibration)
    if routed and adapter is None:
        raise ValueError("routed calibration requires the screen_geometry adapter")
    snapshot = Path(snapshot_download(BASE_REPO, revision=BASE_REVISION, cache_dir=model_cache,
                                      local_files_only=offline))
    verified = None
    if base_hashes_path is not None:
        listing = json.loads(Path(base_hashes_path).read_text(encoding="utf-8"))
        if listing.get("model_id") != BASE_REPO or listing.get("revision") != BASE_REVISION:
            raise RuntimeError("base-model hash listing is for a different model or revision")
        verified = len(verify_snapshot(snapshot, listing["files"]))
    processor = AutoProcessor.from_pretrained(str(snapshot), trust_remote_code=False, use_fast=True)
    auto_model = getattr(transformers, "AutoModelForMultimodalLM", None) or getattr(
        transformers, "AutoModelForImageTextToText")
    model = auto_model.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16,
                                       device_map=device, trust_remote_code=False, attn_implementation="sdpa")
    if adapter is not None:
        from peft import PeftModel

        loaded = PeftModel.from_pretrained(
            model, str(adapter), **({"adapter_name": router.SCREEN_GEOMETRY} if routed else {})
        )
        model = loaded if routed else loaded.merge_and_unload()
    model = model.eval()
    adapter_context = AdapterSwitch(model) if routed else None
    predictor = Predictor(model, processor, calibration, model_name=model_name,
                          context_policy=context_policy, max_input_tokens=max_input_tokens,
                          inference_context=torch.inference_mode,
                          adapter_context=adapter_context,
                          synchronize=(torch.cuda.synchronize if str(device).startswith("cuda") else None))
    provenance = {
        "base_repo": BASE_REPO, "base_revision": BASE_REVISION,
        "base_files_verified": verified,
        "adapter": None if adapter is None else {"path": str(adapter), "sha256": adapter_sha256},
        "image_route": IMAGE_ROUTE,
        "image_route_config_sha256": vision.IMAGE_ROUTE_CONFIG_SHA256[IMAGE_ROUTE],
        "calibration_artifact_sha256": calibration.artifact_sha256,
        "calibration_identity_sha256": calibration.identity_sha256,
        "system_kind": "routed" if routed else "single-route",
        "router_rules_version": router.RULES_VERSION if routed else None,
        "router_rules_sha256": router.RULES_SHA256 if routed else None,
        "adapter_switching": "PEFT layer enable/disable on route transition" if routed else None,
        "precision": "bfloat16", "attention": "sdpa", "processor_use_fast": True, "scorer": "one-prefill-option-letter",
        "context_policy": context_policy, "max_input_tokens": max_input_tokens,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
    }
    return predictor, provenance


__all__ = [
    "BASE_REPO", "BASE_REVISION", "Calibration", "RoutedCalibration", "Predictor", "Question", "RequestError",
    "TYPE_BUCKET_BY_OPTION_COUNT", "check_fast_path", "load_predictor", "parse_request",
    "serving_calibration_from_artifact", "tree_sha256", "verify_snapshot",
]
