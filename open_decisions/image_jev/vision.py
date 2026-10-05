"""Shared image preparation and option readout for Image JevBench.

This module is deliberately the only path from source images to processor inputs.  Training and
serving should both call :func:`prepare_for_route`; keeping resize, prompt and readout decisions
here makes a route or budget change visible and versioned instead of an accidental processor
configuration change.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import hashlib
import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from open_decisions.image_jev.speed import PhaseTimer, move_to_device

# One source for option rendering in training and serving (no fallback path that could drift);
# tests/test_image_jev_vision.py checks it against the public hopper_decisions copy when that is importable.
from open_decisions.scoring import prompt as hopper_prompt


FACTOR = 32
MIN_PIXELS = 65_536
MAX_PIXELS = 16_777_216
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
# Optional native-route image-token cap (a serving option, off by default).  The cap lowers the
# processor's own max pixels to ``cap * FACTOR**2`` so its smart resize keeps at most ``cap``
# merged image tokens per image.  Below 256 tokens the processor's minimum-area rule and its
# one-factor edge clamp can exceed the cap for extreme aspect ratios, so smaller caps are refused.
MIN_IMAGE_TOKEN_CAP = 256
MAX_IMAGE_TOKEN_CAP = MAX_PIXELS // (FACTOR * FACTOR)
BUDGET_VERSION = "b1"
BUDGETS = MappingProxyType({"photo": 160, "scene": 224, "document": 384, "dense": 768})
ROUTING_VERSION = "image-jev/content-routing/v1"
ROUTING_CONFIG = MappingProxyType({
    "version": ROUTING_VERSION,
    "inputs": "decoded-oriented-rgb-pixels-only",
    "policy": "conservative-document",
    "content_class": "document",
})
ROUTING_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(dict(ROUTING_CONFIG), sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
IMAGE_ROUTE_VERSION = "image-jev/image-route/v1"
IMAGE_ROUTES = ("native", "capped")
IMAGE_ROUTE_CONFIG_SHA256 = MappingProxyType({
    route: hashlib.sha256(json.dumps(
        {"version": IMAGE_ROUTE_VERSION, "image_route": route},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    for route in IMAGE_ROUTES
})

IMAGE_SYSTEM = ("You make decisions about supplied images. Inspect every image and reply with the letter of the "
                "correct option and nothing else. Do not produce reasoning.")
VISION_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


@dataclass(frozen=True)
class PreparedInput:
    """The complete model input produced identically for training and serving."""

    input_ids: Any
    pixel_values: Any
    image_grid_thw: Any
    resize_records: list[dict[str, Any]]
    letter_token_ids: list[int]
    prompt: str
    routing_config_sha256: str = ROUTING_CONFIG_SHA256
    # transformers >= 5.x Qwen-VL needs this for multimodal RoPE; returned by the processor beside input_ids.
    mm_token_type_ids: Any = None
    image_route: str = "capped"

    @property
    def ids(self):
        """Short alias used by call sites that name language inputs ``ids``."""
        return self.input_ids

    @property
    def pixels(self):
        """Short alias for the processor's image tensor."""
        return self.pixel_values

    @property
    def records(self):
        """Short alias for the resize audit records."""
        return self.resize_records

    @property
    def image_route_config_sha256(self):
        """Registered identity of the native/capped preparation choice."""
        return IMAGE_ROUTE_CONFIG_SHA256[self.image_route]


def smart_resize(height: int, width: int, factor: int = FACTOR, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Qwen2-VL/Qwen3-VL's ``smart_resize``, with Qwen3.5-4B image defaults.

    The arithmetic and its boundary behavior intentionally track Transformers exactly, including
    its rejection only when the absolute aspect ratio is *greater than* 200.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_tokens(height: int, width: int, factor: int = FACTOR) -> int:
    """Merged image-token count after the model processor's normal smart resize."""
    resized_h, resized_w = smart_resize(height, width, factor=factor)
    return (resized_h // factor) * (resized_w // factor)


def validate_image_token_cap(value: Any) -> int:
    """An explicit native-route image-token cap: an integer in [256, 16384]."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"image_token_cap must be an integer, got {value!r}")
    if not MIN_IMAGE_TOKEN_CAP <= value <= MAX_IMAGE_TOKEN_CAP:
        raise ValueError(f"image_token_cap must be between {MIN_IMAGE_TOKEN_CAP} and "
                         f"{MAX_IMAGE_TOKEN_CAP}, got {value}")
    return value


def native_max_pixels(image_token_cap: int | None = None) -> int:
    """The processor max pixels of the native route: its default, or ``cap * FACTOR**2``."""
    if image_token_cap is None:
        return MAX_PIXELS
    return validate_image_token_cap(image_token_cap) * FACTOR * FACTOR


def native_image_tokens(height: int, width: int, image_token_cap: int | None = None) -> int:
    """Merged image tokens of one native-route image, with the optional cap applied."""
    resized_h, resized_w = smart_resize(height, width, max_pixels=native_max_pixels(image_token_cap))
    return (resized_h // FACTOR) * (resized_w // FACTOR)


def budget_pixels(content_class: str) -> int:
    """Return the explicit pixel-area budget for a named content class."""
    try:
        return BUDGETS[content_class] * FACTOR * FACTOR
    except (KeyError, TypeError):
        raise ValueError(f"unknown image content class {content_class!r}; expected one of {tuple(BUDGETS)}") from None


def route_content(img: Image.Image) -> str:
    """Conservatively route every image to ``document`` for now.

    Content routing is a later measured experiment.  Any future classifier must use only observable
    image content, never benchmark ids, source names, family names or other hidden metadata.
    """
    if not isinstance(img, Image.Image):
        raise TypeError("classify_content expects a PIL.Image.Image")
    return "document"


# Compatibility name for callers that only need the returned class.  All
# consumers route through ``route_content`` and bind ROUTING_CONFIG_SHA256.
classify_content = route_content


def _scaled_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Largest-edge parameterisation of a whole-image, aspect-preserving integer size."""
    if width >= height:
        return long_edge, max(1, min(height, round(height * long_edge / width)))
    return max(1, min(width, round(width * long_edge / height))), long_edge


def _budgeted_size(width: int, height: int, token_budget: int) -> tuple[int, int]:
    """A whole-image size whose default processor result meets ``token_budget``."""
    if image_tokens(height, width) <= token_budget:
        return width, height

    # Prefer the processor's own factor-aligned target.  This makes its subsequent smart resize a
    # no-op and ensures LANCZOS, rather than a second processor interpolation, sets the final pixels.
    target_h, target_w = smart_resize(height, width, max_pixels=token_budget * FACTOR * FACTOR)
    while (target_h // FACTOR) * (target_w // FACTOR) > token_budget:
        if target_w >= target_h and target_w > FACTOR:
            target_w -= FACTOR
        elif target_h > FACTOR:
            target_h -= FACTOR
        else:
            break
    try:
        aligned_tokens = image_tokens(target_h, target_w)
    except ValueError:
        aligned_tokens = token_budget + 1
    if target_w <= width and target_h <= height and aligned_tokens <= token_budget:
        return target_w, target_h

    # A source edge shorter than one factor can make the aligned target an upscale.  Search actual
    # image sizes in that edge case; the saved pixels still enforce the budget even if a later
    # processor is constructed with its native 16M maximum.
    low, high = 1, max(width, height)
    best = None
    while low <= high:
        middle = (low + high) // 2
        candidate = _scaled_size(width, height, middle)
        try:
            fits = image_tokens(candidate[1], candidate[0]) <= token_budget
        except ValueError:
            fits = False
        if fits:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if best is None:  # Unreachable for accepted (<=200:1) inputs and the smallest 160-token budget.
        raise ValueError(f"image aspect ratio cannot fit the {token_budget}-token budget")
    return best


def resize_for_budget(img: Image.Image, content_class: str) -> tuple[Image.Image, dict[str, Any]]:
    """Convert to RGB and, if necessary, LANCZOS-downscale the whole image to its named budget."""
    if not isinstance(img, Image.Image):
        raise TypeError("resize_for_budget expects a PIL.Image.Image")
    token_budget = budget_pixels(content_class) // (FACTOR * FACTOR)
    orig_size = img.size
    new_size = _budgeted_size(*orig_size, token_budget)
    converted = img.convert("RGB")
    resized = converted if new_size == orig_size else converted.resize(new_size, Image.Resampling.LANCZOS)
    tokens = image_tokens(resized.height, resized.width)
    if tokens > token_budget:  # Keep a hard invariant next to the operation that enforces it.
        raise RuntimeError(f"resize produced {tokens} image tokens for a {token_budget}-token budget")
    record = {"content_class": content_class, "budget_version": BUDGET_VERSION,
              "orig_size": orig_size, "new_size": resized.size, "image_tokens": tokens}
    return resized, record


def _normalise_options(options) -> list[dict[str, str]]:
    normalised = []
    for index, option in enumerate(options):
        if isinstance(option, str):
            normalised.append({"name": option, "description": option})
        elif isinstance(option, dict) and "name" in option and "description" in option:
            normalised.append({"name": str(option["name"]), "description": str(option["description"])})
        elif isinstance(option, (tuple, list)) and len(option) == 2:
            normalised.append({"name": str(option[0]), "description": str(option[1])})
        else:
            raise ValueError(f"option {index} must be text, a (name, description) pair, or a mapping with those keys")
    if not 2 <= len(normalised) <= len(hopper_prompt.LETTERS):
        raise ValueError("image decisions require 2 to 26 options")
    return normalised


def render_prompt(question: str, options, n_images: int) -> str:
    """Render the image decision user turn with explicit letter-description pairs and thinking off."""
    if not isinstance(question, str) or not question:
        raise ValueError("question must be non-empty text")
    if not isinstance(n_images, int) or isinstance(n_images, bool) or n_images < 1:
        raise ValueError("n_images must be a positive integer")
    normalised = _normalise_options(options)
    shown = hopper_prompt.option_lines({"kind": "choice", "options": normalised})
    payload = {"question": question,
               "options": [{"letter": hopper_prompt.LETTERS[i], "description": text}
                           for i, (_, text) in enumerate(shown)]}
    images = "\n".join(f"IMAGE {i + 1}\n{VISION_TOKEN}" for i in range(n_images))
    return (f"{images}\n\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            "Reply with one option letter only; do not produce reasoning.")


def _decode_data_uri(uri: str) -> bytes:
    try:
        header, encoded = uri.split(",", 1)
    except ValueError:
        raise ValueError("bad image data URI: missing comma") from None
    if not header.startswith("data:image/") or not header.endswith(";base64"):
        raise ValueError("bad image data URI: expected data:image/...;base64,...")
    # Reject an oversized payload before allocating its decoded representation.  Padding means the
    # estimate can exceed the true size by at most two bytes, so an exact check follows decoding.
    if (len(encoded) // 4) * 3 > MAX_IMAGE_BYTES + 2:
        raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("bad image data URI: invalid base64 payload") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
    return raw


def _image_bytes(source) -> bytes:
    if isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes")
        return raw
    if isinstance(source, str) and source.startswith("data:"):
        return _decode_data_uri(source)
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ValueError(f"cannot read image path {path}: {error}") from error
        if size > MAX_IMAGE_BYTES:
            raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes: {path}")
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read image path {path}: {error}") from error
        if len(raw) > MAX_IMAGE_BYTES:  # The file may have changed between stat and read.
            raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} bytes: {path}")
        return raw
    raise TypeError("each image must be bytes, a filesystem path, or a base64 image data URI")


def _load_image(source, *, exif_transpose: bool = True) -> Image.Image:
    raw = _image_bytes(source)
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            width, height = opened.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"image exceeds {MAX_IMAGE_PIXELS} pixels: {width}x{height}")
            opened.load()
            ready = ImageOps.exif_transpose(opened) if exif_transpose else opened
            # A loaded image owns its pixels after the file closes.
            return ready
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError(f"cannot decode image: {error}") from error


def _output_field(output, name: str):
    if isinstance(output, dict):
        value = output.get(name)
    else:
        value = getattr(output, name, None)
    if value is None:
        raise ValueError(f"processor output is missing {name!r}")
    return value


def _processor_min_pixels(processor) -> int:
    """The processor's own minimum area (kept unchanged when a token cap lowers the maximum)."""
    image_processor = getattr(processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    for getter in (lambda: size["shortest_edge"], lambda: getattr(size, "shortest_edge"),
                   lambda: getattr(image_processor, "min_pixels")):
        try:
            value = getter()
        except (KeyError, TypeError, AttributeError):
            continue
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return MIN_PIXELS


def _grid_rows(grid) -> list[list[int]]:
    rows = grid.tolist() if hasattr(grid, "tolist") else [list(row) for row in grid]
    return [[int(value) for value in row] for row in rows]


def _prepare(example: dict, processor, *, image_route: str,
             content_class: str | None = None, loaded_images=None, cache=None,
             timer=None, device=None, image_token_cap: int | None = None) -> PreparedInput:
    """Prepare one decision after validating an explicit registered image route.

    ``image_token_cap`` (native route only; default None = the processor's own maximum) lowers
    the processor's max pixels so every image keeps at most that many merged image tokens.
    """
    if image_route not in IMAGE_ROUTES:
        raise ValueError(f"image_route must be one of {IMAGE_ROUTES}, got {image_route!r}")
    if image_token_cap is not None:
        if image_route != "native":
            raise ValueError("image_token_cap applies to the native image route only")
        image_token_cap = validate_image_token_cap(image_token_cap)
    try:
        question, options = example["question"], example["options"]
    except (KeyError, TypeError) as error:
        raise ValueError("example must contain question and options") from error
    sources = example.get("images")
    if sources is None and "image" in example:
        sources = [example["image"]]
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError("example must contain a non-empty images list")

    normalised = _normalise_options(options)
    if content_class is not None:
        raise ValueError("content_class overrides are forbidden; routing uses observable image content")
    # The historical native read path used decoded pixel order directly. Keep
    # that exact behavior while capped retains its established EXIF transpose.
    timer = timer or PhaseTimer()
    with timer.phase("decode_load"):
        loaded = loaded_images if loaded_images is not None else [
            _load_image(source, exif_transpose=image_route == "capped") for source in sources
        ]
    with timer.phase("processor_preprocess"):
        ready_images, records = [], []
        for image in loaded:
            routed_class = route_content(image)
            if image_route == "capped":
                ready, record = resize_for_budget(image, routed_class)
            else:
                ready = image if image.mode == "RGB" else image.convert("RGB")
                record = {
                    "image_route": "native",
                    "content_class": routed_class,
                    "orig_size": image.size,
                    "new_size": ready.size,
                    "image_tokens": image_tokens(ready.height, ready.width),
                }
                if image_token_cap is not None:
                    # The processor, not this function, resizes; new_size stays the decoded size.
                    record["image_token_cap"] = image_token_cap
                    record["image_tokens"] = native_image_tokens(ready.height, ready.width,
                                                                 image_token_cap)
            ready_images.append(ready)
            records.append(record)

    with timer.phase("tokenise_chat_template"):
        user = render_prompt(question, normalised, len(ready_images))
        if cache is not None:
            prompt = cache.prompt(user)
        elif hasattr(processor, "apply_chat_template"):
            messages = [{"role": "system", "content": IMAGE_SYSTEM}, {"role": "user", "content": user}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)
        else:
            prompt = user
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("processor must expose its tokenizer for option-letter readout")
        token_ids = (cache.letter_ids(len(normalised)) if cache is not None else
                     hopper_prompt.letter_token_ids(tokenizer)[:len(normalised)])
    # Fast torch image processors accept device; slow/PIL processors do not.
    image_processor = getattr(processor, "image_processor", None)
    fast = image_processor is not None and (
        getattr(image_processor, "backend", None) == "torch" or
        "Fast" in type(image_processor).__name__ or
        any(base.__name__ == "BaseImageProcessorFast" for base in type(image_processor).__mro__))
    images_kwargs = {"device": device} if fast and device is not None else {}
    if image_token_cap is not None:
        # Every spelling the processor's kwargs accept, all consistent: min unchanged, max capped.
        min_pixels, max_pixels = _processor_min_pixels(processor), native_max_pixels(image_token_cap)
        images_kwargs.update(size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
                             min_pixels=min_pixels, max_pixels=max_pixels)
    kwargs = {"images_kwargs": images_kwargs} if images_kwargs else {}
    if cache is not None:
        cache.tokenizer.timer = timer
    token_before = timer.seconds["tokenise_chat_template"] if timer.seconds is not None else 0.0
    with timer.phase("processor_preprocess"):
        output = processor(text=[prompt], images=ready_images, padding=True, return_tensors="pt", **kwargs)
        if image_token_cap is not None:
            # Fail closed if the processor ignored the cap (a library change would do that).
            merge = int(getattr(getattr(processor, "image_processor", None), "merge_size", 2) or 2)
            served = [t * h * w // (merge * merge)
                      for t, h, w in _grid_rows(_output_field(output, "image_grid_thw"))]
            if len(served) != len(records) or any(tokens > image_token_cap for tokens in served):
                raise RuntimeError(f"processor returned {served} image tokens for an "
                                   f"{image_token_cap}-token cap")
            for record, tokens in zip(records, served):
                record["processor_image_tokens"] = tokens
    if timer.seconds is not None:
        timer.seconds["processor_preprocess"] -= timer.seconds["tokenise_chat_template"] - token_before
    return PreparedInput(input_ids=_output_field(output, "input_ids"),
                         pixel_values=_output_field(output, "pixel_values"),
                         image_grid_thw=_output_field(output, "image_grid_thw"),
                         resize_records=records, letter_token_ids=token_ids, prompt=prompt,
                         mm_token_type_ids=(output.get("mm_token_type_ids") if isinstance(output, dict)
                                            else getattr(output, "mm_token_type_ids", None)),
                         image_route=image_route)


def prepare(example: dict, processor, content_class: str | None = None) -> PreparedInput:
    """Prepare the existing capped route shared by capped training and serving."""
    return _prepare(example, processor, image_route="capped", content_class=content_class)


def prepare_native(example: dict, processor, **kwargs) -> PreparedInput:
    """Prepare processor-default full-resolution images without budget resizing.

    ``image_token_cap=N`` (optional, default off) lowers only the processor's max pixels.
    """
    return _prepare(example, processor, image_route="native", **kwargs)


def prepare_for_route(example: dict, processor, route: str) -> PreparedInput:
    """Dispatch one example through exactly one registered image route."""
    if route == "native":
        return prepare_native(example, processor)
    if route == "capped":
        return prepare(example, processor)
    raise ValueError(f"image_route must be one of {IMAGE_ROUTES}, got {route!r}")


def last_position_logits(model, prepared, *, timer=None, forward_call=None):
    """Run exactly one last-position vocabulary projection for every local reader."""
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = None

    timer = timer or PhaseTimer()

    def move(value):
        return move_to_device(value, device)

    with timer.phase("h2d_copy"):
        kwargs = {
            "input_ids": move(prepared.input_ids),
            "pixel_values": move(prepared.pixel_values),
            "image_grid_thw": move(prepared.image_grid_thw),
            "use_cache": False,
            **({"mm_token_type_ids": move(prepared.mm_token_type_ids)}
               if getattr(prepared, "mm_token_type_ids", None) is not None else {}),
            "return_dict": True,
        }
    readout_kwargs = getattr(model, "_image_last_position_kwargs", None)
    if readout_kwargs is None:
        parameters = inspect.signature(model.forward).parameters
        if "logits_to_keep" in parameters or any(
            parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            readout_kwargs = {"logits_to_keep": 1}
        elif "num_logits_to_keep" in parameters:
            readout_kwargs = {"num_logits_to_keep": 1}
        else:
            raise ValueError("model does not expose a last-position logits readout")
        model._image_last_position_kwargs = readout_kwargs
    kwargs.update(readout_kwargs)
    with timer.phase("forward"):
        output = model(**kwargs) if forward_call is None else forward_call(lambda: model(**kwargs))
    logits = output["logits"] if isinstance(output, dict) else output.logits
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != 1:
        raise ValueError("model must return exactly [1, 1, vocabulary] logits")
    return logits[0, 0, :]


def calibrated_read(model, prepared: PreparedInput, *, temperature: float) -> tuple[list[float], list[float]]:
    """Return raw and checkpoint-artifact-calibrated option probabilities."""
    logits = last_position_logits(model, prepared)
    import torch

    raw = option_log_probs(logits, prepared.letter_token_ids, validate_finite=False).exp()
    calibrated = option_log_probs(logits, prepared.letter_token_ids, temperature,
                                  validate_finite=False).exp()
    values = torch.stack((raw, calibrated)).detach().cpu().tolist()
    if not all(math.isfinite(value) for row in values for value in row):
        raise ValueError("option probabilities are non-finite")
    return values[0], values[1]


def option_log_probs(logits_last, letter_token_ids: list[int], temperature: float = 1.0, *,
                     validate_finite: bool = True):
    """Differentiable FP32 log-softmax over the served option-letter rows.

    Training uses this function directly and :func:`option_probs` is only its
    exponentiated, detached presentation form.  Keeping selection, validation,
    dtype conversion and temperature here prevents a train/serve readout fork.
    """
    import torch

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    if not letter_token_ids:
        raise ValueError("letter_token_ids must not be empty")
    logits = torch.as_tensor(logits_last)
    if logits.ndim != 1:
        raise ValueError("logits_last must be a one-dimensional vocabulary vector")
    if any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < logits.numel()
           for index in letter_token_ids):
        raise ValueError("letter_token_ids must be valid vocabulary indices")
    try:
        selected = logits[list(letter_token_ids)].to(dtype=torch.float32) / temperature
    except (IndexError, TypeError) as error:
        raise ValueError("letter_token_ids must be valid vocabulary indices") from error
    log_probabilities = torch.log_softmax(selected, dim=-1)
    if validate_finite and not bool(torch.isfinite(log_probabilities).all()):
        raise ValueError("option probabilities are non-finite")
    return log_probabilities


def option_probs(logits_last, letter_token_ids: list[int], temperature: float = 1.0) -> list[float]:
    """FP32 softmax over only the displayed option-letter logits."""
    values = option_log_probs(logits_last, letter_token_ids, temperature, validate_finite=False).exp().detach().cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("option probabilities are non-finite")
    return values
