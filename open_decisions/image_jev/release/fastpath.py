"""Opt-in serving optimisations for the Image Hopper release server (cost audit, step 1).

Every option is off by default, so the published path is unchanged unless the server is started
with ``--serving-options``.  Options are either

* **exact**: the served function is bit-identical by construction (same operations in the same
  order, only Python overhead, copies or host transfers removed); or
* **tolerance-bound**: numerically close but not bit-identical.  The cost audit registers the
  shipping rule before measuring: argmax identical on every audit item and max |dp| <= 1e-3.

Options (``OPTIONS`` holds the one-line descriptions):

``lean_lora``      exact.  PEFT's LoRA wrappers are replaced by plain modules that call the
                   original base layer, plus the screen_geometry LoRA term computed with PEFT's
                   exact operations (input cast to the adapter dtype, B(A(x)) * scaling, add,
                   cast back) only while that route is active.  Route change flips one flag.
``lora_bf16``      tolerance.  As lean_lora, LoRA factors held in the base dtype (bf16).
``merged_views``   tolerance.  The screen_geometry route reads a second, merged bf16 copy of each
                   targeted weight (W + s*B@A rounded once); the default route reads W.
``uint8_pixels``   exact.  The processor resizes and patchifies as before but returns uint8
                   patches; they are uploaded from pinned memory and normalised on the device with
                   the processor's own fused mean/std arithmetic (elementwise, so identical).
``gpu_preprocess`` tolerance.  uint8_pixels plus the resize itself on the device.
``device_readout`` exact.  Option rows gathered with a cached device index tensor; one host copy.
``cuda_graphs``    tolerance.  The language model (and the last-position vocabulary row) is
                   replayed from CUDA graphs recorded at padded lengths, one set per route.
                   Right padding after the decision position cannot change a causal model's
                   output there; kernels chosen for the padded shape can round differently.
``sdpa_flash``     tolerance.  Scaled-dot-product attention restricted to the flash kernel.
``compile_text``   tolerance.  ``torch.compile(dynamic=True)`` on the language model.

One parameterised option changes outputs by design (it is neither exact nor tolerance-bound):

``image_token_cap_default=N`` output-changing.  As below, but only for questions whose
                   router-rules/v1 text route is ``default``; ``screen_geometry`` images stay
                   native.  The two caps are alternatives.
``image_token_cap=N`` output-changing.  The native route's processor max pixels drop to
                   ``N * 32 * 32`` so every image keeps at most N merged image tokens
                   (``vision.native_image_tokens``); N in [256, 16384].  Weights, router and
                   calibration are unchanged.  Off (no cap) unless given.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

OPTIONS: Mapping[str, str] = {
    "lean_lora": "exact: plain base layers + route-switched LoRA term with PEFT's operations",
    "lora_bf16": "tolerance: route-switched LoRA term with bf16 factors",
    "merged_views": "tolerance: screen_geometry route reads merged bf16 weight copies",
    "uint8_pixels": "exact: uint8 patches, pinned upload, device normalisation",
    "gpu_preprocess": "tolerance: uint8_pixels plus device resize",
    "device_readout": "exact: cached device index for option rows, one host copy",
    "cuda_graphs": "tolerance: language model replayed from CUDA graphs at padded lengths",
    "sdpa_flash": "tolerance: SDPA restricted to the flash kernel",
    "compile_text": "tolerance: torch.compile(dynamic=True) on the language model",
}
EXACT_OPTIONS = frozenset({"lean_lora", "uint8_pixels", "device_readout"})
IMAGE_TOKEN_CAP = "image_token_cap"
IMAGE_TOKEN_CAP_DEFAULT = "image_token_cap_default"
PARAMETERISED_OPTIONS: Mapping[str, str] = {
    IMAGE_TOKEN_CAP: "output-changing: native-route processor max pixels = N x 32 x 32 (N tokens)",
    IMAGE_TOKEN_CAP_DEFAULT: "output-changing: as image_token_cap, only for questions the text "
                             "rule routes to default (screen_geometry stays native)",
}
ROUTE_MODES = ("lean_lora", "lora_bf16", "merged_views")
# Padded lengths for the recorded graphs: 64-token steps to 1,024, then 128, 256 and 512.
GRAPH_LENGTHS = tuple(list(range(64, 1025, 64)) + list(range(1152, 2049, 128))
                      + list(range(2304, 4097, 256)) + list(range(4608, 8193, 512)))


class ServingOptionError(ValueError):
    """Unknown or conflicting serving options."""


def _parameter(item: str) -> tuple[str, int] | None:
    name, sep, raw = item.partition("=")
    if not sep:
        return None
    name, raw = name.strip(), raw.strip()
    if name not in PARAMETERISED_OPTIONS:
        raise ServingOptionError(f"unknown serving option {name!r}; parameterised options: "
                                 f"{sorted(PARAMETERISED_OPTIONS)}")
    if not raw.isdigit():
        raise ServingOptionError(f"{name} needs a positive integer, got {raw!r}")
    from open_decisions.image_jev import vision

    try:
        return name, vision.validate_image_token_cap(int(raw))
    except ValueError as error:
        raise ServingOptionError(str(error)) from None


def parse_options(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Validate options; return them in the canonical (``OPTIONS``) order.

    A parameterised option (``image_token_cap=N``) follows the named options, normalised to
    ``name=N``; giving it twice is refused.
    """
    if value is None:
        return ()
    items = [item.strip() for item in value.split(",")] if isinstance(value, str) else list(value)
    items = [item for item in items if item]
    parameters: dict[str, int] = {}
    named = []
    for item in items:
        parsed = _parameter(item)
        if parsed is None:
            named.append(item)
            continue
        name, number = parsed
        if name in parameters and parameters[name] != number:
            raise ServingOptionError(f"{name} given twice ({parameters[name]} and {number})")
        parameters[name] = number
    if len(parameters) > 1:
        raise ServingOptionError("image_token_cap and image_token_cap_default are alternatives")
    items = named
    unknown = sorted(set(items) - set(OPTIONS))
    if unknown:
        raise ServingOptionError(f"unknown serving options {unknown}; known: {sorted(OPTIONS)}"
                                 f" and {sorted(f'{k}=N' for k in PARAMETERISED_OPTIONS)}")
    chosen = set(items)
    modes = [name for name in ROUTE_MODES if name in chosen]
    if "lora_bf16" in chosen and "merged_views" in chosen:
        raise ServingOptionError("lora_bf16 and merged_views are alternative route modes")
    if "compile_text" in chosen and "cuda_graphs" in chosen:
        raise ServingOptionError("compile_text and cuda_graphs are alternatives")
    if "gpu_preprocess" in chosen:
        chosen.add("uint8_pixels")
    if modes and "lean_lora" not in chosen:
        chosen.add("lean_lora")  # lora_bf16/merged_views use the same route modules
    return (tuple(name for name in OPTIONS if name in chosen)
            + tuple(f"{name}={parameters[name]}" for name in PARAMETERISED_OPTIONS
                    if name in parameters))


def image_token_cap(options: Iterable[str], name: str = IMAGE_TOKEN_CAP) -> int | None:
    """The ``image_token_cap=N`` value among parsed options, or None (no cap: the default)."""
    for item in options:
        parsed = _parameter(item)
        if parsed is not None and parsed[0] == name:
            return parsed[1]
    return None


def image_token_cap_default(options: Iterable[str]) -> int | None:
    """The route-conditioned ``image_token_cap_default=N`` value, or None."""
    return image_token_cap(options, IMAGE_TOKEN_CAP_DEFAULT)


def is_exact(options: Iterable[str]) -> bool:
    """True only for options that keep the served function bit-identical (no cap)."""
    return set(options) <= EXACT_OPTIONS


def route_mode(options: Iterable[str]) -> str | None:
    chosen = set(options)
    if "merged_views" in chosen:
        return "merged"
    if "lora_bf16" in chosen:
        return "bf16"
    if "lean_lora" in chosen:
        return "exact"
    return None


# ---------------------------------------------------------------------------------- route modules
class RouteState:
    """One shared flag: True while the screen_geometry route is served."""

    __slots__ = ("on",)

    def __init__(self, on: bool = True):
        self.on = bool(on)


def _route_linear_class():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class RouteLinear(nn.Module):
        """A LoRA-targeted linear layer with the adapter switched by ``RouteState``.

        ``exact`` replicates PEFT's vanilla LoRA forward operation by operation:
        ``result = base(x); x = x.to(A.dtype); result = result + B(A(x)) * scaling;
        result.to(result_dtype)``.  Off-route it calls the original base layer only.
        """

        def __init__(self, base, lora_a, lora_b, scaling: float, state: RouteState, mode: str):
            super().__init__()
            if mode not in ("exact", "bf16", "merged"):
                raise ValueError("unknown route mode")
            self.base = base
            self.state = state
            self.mode = mode
            self.scaling = float(scaling)
            weight = base.weight
            if mode == "merged":
                delta = (lora_b.detach().float() @ lora_a.detach().float()) * self.scaling
                merged = (weight.detach().float() + delta.to(weight.device)).to(weight.dtype)
                self.register_buffer("merged_weight", merged, persistent=False)
                self.lora_a = self.lora_b = None
            else:
                dtype = weight.dtype if mode == "bf16" else lora_a.dtype
                self.register_buffer("lora_a", lora_a.detach().to(dtype), persistent=False)
                self.register_buffer("lora_b", lora_b.detach().to(dtype), persistent=False)

        @property
        def weight(self):  # modules that inspect .weight keep working
            return self.base.weight

        def forward(self, x):
            if not self.state.on:
                return self.base(x)
            if self.mode == "merged":
                return F.linear(x, self.merged_weight, self.base.bias)
            result = self.base(x)
            result_dtype = result.dtype
            if x.dtype != self.lora_a.dtype:
                x = x.to(dtype=self.lora_a.dtype)
            result = result + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
            return result.to(result_dtype)

    return RouteLinear, torch


def _lora_layers(root) -> list[tuple[str, Any]]:
    return [(name, module) for name, module in root.named_modules()
            if hasattr(module, "base_layer") and hasattr(module, "lora_A")
            and hasattr(module, "lora_B") and hasattr(module, "scaling")]


def install_route_modules(peft_model, *, adapter_name: str, mode: str):
    """Replace PEFT LoRA wrappers by ``RouteLinear``; return (plain model, state, record)."""
    from torch import nn

    RouteLinear, torch = _route_linear_class()
    root = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    state = RouteState(True)
    layers = _lora_layers(root)
    if not layers:
        raise ServingOptionError("no LoRA layers found to replace")
    extra_bytes = 0
    for name, module in layers:
        if not isinstance(module.base_layer, nn.Linear):
            raise ServingOptionError(f"{name}: LoRA on a non-linear layer is not supported")
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            raise ServingOptionError(f"{name}: adapter {adapter_name!r} missing")
        if adapter_name in getattr(module, "lora_variant", {}) or getattr(module, "merged", False):
            raise ServingOptionError(f"{name}: DoRA/variant or merged LoRA is not supported")
        if getattr(module, "fan_in_fan_out", False):
            raise ServingOptionError(f"{name}: fan_in_fan_out is not supported")
        dropout = module.lora_dropout[adapter_name]
        if not isinstance(dropout, nn.Identity) and getattr(dropout, "p", 0.0) != 0.0:
            raise ServingOptionError(f"{name}: LoRA dropout must be zero at serving")
        lora_a, lora_b = module.lora_A[adapter_name], module.lora_B[adapter_name]
        if getattr(lora_a, "bias", None) is not None or getattr(lora_b, "bias", None) is not None:
            raise ServingOptionError(f"{name}: LoRA bias is not supported")
        replacement = RouteLinear(module.base_layer, lora_a.weight, lora_b.weight,
                                  module.scaling[adapter_name], state, mode)
        if mode == "merged":
            extra_bytes += replacement.merged_weight.numel() * replacement.merged_weight.element_size()
        parent_name, _, child = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, child, replacement)
    del peft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    record = {"route_mode": mode, "replaced_layers": len(layers),
              "merged_view_bytes": extra_bytes}
    return root.eval(), state, record


class RouteSwitch:
    """Adapter context for ``RouteLinear`` models: one flag assignment per route change."""

    def __init__(self, state: RouteState):
        self.state = state

    def __call__(self, route):
        from open_decisions.image_jev import router

        if route not in router.ROUTES:
            raise ValueError("unknown adapter route")
        self.state.on = route == router.SCREEN_GEOMETRY
        return contextlib.nullcontext()


# ------------------------------------------------------------------------------------ processor
class PixelProcessor:
    """Processor proxy returning device-normalised pixels from uint8 patches.

    The image processor normalises before patchifying; normalisation is per channel and
    elementwise and patchify is a permutation plus temporal duplication, so normalising the
    uint8 patches with the same fused mean/std tensors gives identical float32 values.
    """

    def __init__(self, processor, device, *, gpu_resize: bool = False):
        object.__setattr__(self, "_inner", processor)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_gpu_resize", bool(gpu_resize))
        object.__setattr__(self, "_norm", None)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)

    def _mean_std(self, device):
        import torch

        if self._norm is None or self._norm[0] != device:
            ip = self._inner.image_processor
            mean = torch.tensor(ip.image_mean, device=device) * (1.0 / ip.rescale_factor)
            std = torch.tensor(ip.image_std, device=device) * (1.0 / ip.rescale_factor)
            per = int(ip.temporal_patch_size) * int(ip.patch_size) * int(ip.patch_size)
            object.__setattr__(self, "_norm", (device, mean.to(torch.float32).repeat_interleave(per),
                                               std.to(torch.float32).repeat_interleave(per)))
        return self._norm[1], self._norm[2]

    def normalize(self, patches):
        import torch

        ip = self._inner.image_processor
        if not (getattr(ip, "do_rescale", True) and getattr(ip, "do_normalize", True)):
            raise RuntimeError("uint8_pixels expects the processor's default rescale+normalize")
        mean, std = self._mean_std(patches.device)
        return patches.to(dtype=torch.float32).sub_(mean).div_(std)

    def __call__(self, *args, **kwargs):
        import torch

        if kwargs.get("images") is None and len(args) < 2:
            return self._inner(*args, **kwargs)
        images_kwargs = dict(kwargs.pop("images_kwargs", None) or {})
        images_kwargs.update(do_rescale=False, do_normalize=False)
        if self._gpu_resize and self._device is not None:
            images_kwargs["device"] = self._device
        output = self._inner(*args, images_kwargs=images_kwargs, **kwargs)
        pixels = output["pixel_values"]
        if pixels.dtype != torch.uint8:
            raise RuntimeError(f"expected uint8 patches, got {pixels.dtype}")
        device = self._device
        if device is not None and str(device).startswith("cuda") and pixels.device.type == "cpu":
            pixels = pixels.pin_memory().to(device, non_blocking=True)
        elif device is not None and pixels.device != torch.device(device):
            pixels = pixels.to(device)
        output["pixel_values"] = self.normalize(pixels)
        return output


# --------------------------------------------------------------------------------------- readout
class DeviceReadout:
    """``vision.option_probs`` with a cached device index (same operations after the gather)."""

    def __init__(self):
        self._index: dict[tuple, Any] = {}

    def __call__(self, logits, letter_token_ids, temperature: float) -> list[float]:
        import math

        import torch

        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        key = (str(logits.device), tuple(letter_token_ids))
        index = self._index.get(key)
        if index is None:
            if any(not 0 <= int(i) < logits.numel() for i in letter_token_ids):
                raise ValueError("letter_token_ids must be valid vocabulary indices")
            index = torch.tensor(list(letter_token_ids), dtype=torch.long, device=logits.device)
            self._index[key] = index
        selected = logits.index_select(0, index).to(dtype=torch.float32) / temperature
        values = torch.log_softmax(selected, dim=-1).exp().detach().cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise ValueError("option probabilities are non-finite")
        return values


# ---------------------------------------------------------------------------------------- graphs
def hf_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _move(value, device):
    from open_decisions.image_jev.speed import move_to_device

    return move_to_device(value, device)


def embeds_positions(model, prepared, device):
    """The multimodal forward up to the language model (the model's own methods, same order)."""
    import torch

    top = hf_model(model)
    core = top.model
    ids = _move(prepared.input_ids, device)
    pixels = _move(prepared.pixel_values, device)
    grid = _move(prepared.image_grid_thw, device)
    mm = getattr(prepared, "mm_token_type_ids", None)
    mm = None if mm is None else _move(mm, device)
    embeds = core.get_input_embeddings()(ids)
    if pixels is not None:
        features = core.get_image_features(pixels, grid, return_dict=True).pooler_output
        features = torch.cat(features, dim=0).to(embeds.device, embeds.dtype)
        mask, _ = core.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=features)
        embeds = embeds.masked_scatter(mask, features)
    positions = core.compute_3d_position_ids(
        input_ids=ids, inputs_embeds=embeds, image_grid_thw=grid, video_grid_thw=None,
        attention_mask=None, past_key_values=None, mm_token_type_ids=mm)
    return embeds, positions


def pad_positions(positions, size: int):
    """Right-pad (3, 1, n) position ids to ``size`` by continuing each axis from its last value."""
    import torch

    n = positions.shape[-1]
    if size < n:
        raise ValueError("padded size is shorter than the sequence")
    if size == n:
        return positions
    tail = positions[..., -1:] + torch.arange(1, size - n + 1, device=positions.device,
                                              dtype=positions.dtype)
    return torch.cat((positions, tail), dim=-1)


def eager_last_logits(model, embeds, positions):
    top = hf_model(model)
    hidden = top.model.language_model(input_ids=None, position_ids=positions, attention_mask=None,
                                      past_key_values=None, inputs_embeds=embeds,
                                      use_cache=False).last_hidden_state
    return top.lm_head(hidden[:, -1:, :])[0, 0]


class DecoderGraphs:
    """CUDA graphs of language model + last-row vocabulary projection at padded lengths."""

    def __init__(self, model, *, lengths: Sequence[int], routes: Sequence[Any],
                 set_route: Callable[[Any], Any] | None, device, log=print):
        import torch

        top = hf_model(model)
        self.model = model
        lm, head = top.model.language_model, top.lm_head
        hidden = int(top.config.text_config.hidden_size)
        dtype = next(lm.parameters()).dtype
        self.graphs: dict[tuple[Any, int], tuple] = {}
        self.lengths = sorted(set(int(n) for n in lengths))
        started = time.monotonic()
        pool = None
        with torch.inference_mode():
            for route in routes:
                if set_route is not None:
                    set_route(route)
                for n in sorted(self.lengths, reverse=True):  # longest first; shorter reuse pool
                    embeds = torch.zeros(1, n, hidden, device=device, dtype=dtype)
                    positions = torch.arange(n, device=device).view(1, 1, n).expand(3, 1, n).contiguous()
                    last = torch.zeros(1, dtype=torch.long, device=device)

                    def step(embeds=embeds, positions=positions, last=last):
                        state = lm(input_ids=None, position_ids=positions, attention_mask=None,
                                   past_key_values=None, inputs_embeds=embeds,
                                   use_cache=False).last_hidden_state
                        return head(state[0].index_select(0, last))[0]

                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            step()
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        out = step()
                    pool = graph.pool()
                    self.graphs[(route, n)] = (graph, embeds, positions, last, out)
        self.capture_seconds = time.monotonic() - started
        self.routes = list(routes)
        log(f"captured {len(self.graphs)} graphs in {self.capture_seconds:.1f} s")

    def fits(self, n: int) -> bool:
        return bool(self.lengths) and n <= self.lengths[-1]

    def run(self, route, embeds, positions):
        n = embeds.shape[1]
        size = next(x for x in self.lengths if x >= n)
        graph, static_embeds, static_positions, last, out = self.graphs[(route, size)]
        static_embeds.zero_()
        static_embeds[:, :n].copy_(embeds)
        static_positions.copy_(pad_positions(positions, size))
        last.fill_(n - 1)
        graph.replay()
        return out.clone()


class GraphLogits:
    """``logits_fn`` for the Predictor: eager multimodal front, graph-replayed language model."""

    def __init__(self, graphs: DecoderGraphs | None, device, *, route_key: Callable[[Any], Any]):
        self.graphs, self.device, self.route_key = graphs, device, route_key
        self.replayed = self.eager = 0

    def __call__(self, model, prepared, *, timer, route=None):
        from open_decisions.image_jev.speed import PhaseTimer

        timer = timer or PhaseTimer()
        with timer.phase("forward"):
            embeds, positions = embeds_positions(model, prepared, self.device)
            n = embeds.shape[1]
            if self.graphs is not None and self.graphs.fits(n):
                self.replayed += 1
                return self.graphs.run(self.route_key(route), embeds, positions)
            self.eager += 1
            return eager_last_logits(model, embeds, positions)


# ------------------------------------------------------------------------------------------ apply
@dataclasses.dataclass
class Serving:
    model: Any
    processor: Any
    adapter_context: Callable | None
    inference_context: Callable[[], Any] | None
    logits_fn: Callable | None
    readout_fn: Callable | None
    record: dict
    image_token_cap: int | None = None
    image_token_cap_default: int | None = None


def _sdpa_flash_context():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.FLASH_ATTENTION])


def apply_serving_options(model, processor, options: Sequence[str], *, device, routed: bool,
                          adapter_name: str | None, adapter_context: Callable | None,
                          inference_context: Callable[[], Any] | None,
                          graph_lengths: Sequence[int] = GRAPH_LENGTHS, log=print) -> Serving:
    """Apply validated ``options`` to a loaded (PEFT or plain) model and its processor."""
    import torch

    options = parse_options(options)
    record: dict[str, Any] = {"serving_options": list(options), "exact": is_exact(options)}
    cap = image_token_cap(options)
    if cap is not None:
        from open_decisions.image_jev import vision

        record["image_token_cap"] = cap
        record["native_max_pixels"] = vision.native_max_pixels(cap)
    cap_default = image_token_cap_default(options)
    if cap_default is not None:
        from open_decisions.image_jev import vision

        record["image_token_cap_default"] = cap_default
        record["native_max_pixels_default_route"] = vision.native_max_pixels(cap_default)
    started = time.monotonic()
    mode = route_mode(options)
    if mode is not None:
        if not routed or adapter_name is None:
            raise ServingOptionError("route modes need the routed single-adapter system")
        model, state, route_record = install_route_modules(model, adapter_name=adapter_name,
                                                           mode=mode)
        adapter_context = RouteSwitch(state)
        record.update(route_record)
    if "uint8_pixels" in options:
        processor = PixelProcessor(processor, device, gpu_resize="gpu_preprocess" in options)
    readout_fn = DeviceReadout() if "device_readout" in options else None
    if "sdpa_flash" in options:
        base_context = inference_context

        @contextlib.contextmanager
        def flash_context():
            with (base_context() if base_context is not None else contextlib.nullcontext()):
                with _sdpa_flash_context():
                    yield

        inference_context = flash_context
    if "compile_text" in options:
        core = hf_model(model).model
        core.language_model = torch.compile(core.language_model, dynamic=True)
        record["compile"] = "torch.compile(language_model, dynamic=True)"
    logits_fn = None
    if "cuda_graphs" in options:
        if not str(device).startswith("cuda"):
            raise ServingOptionError("cuda_graphs needs a CUDA device")
        from open_decisions.image_jev import router

        routes = list(router.ROUTES) if routed else [None]
        set_route = adapter_context if routed else None
        context = inference_context() if inference_context is not None else contextlib.nullcontext()
        with context:
            graphs = DecoderGraphs(model, lengths=graph_lengths, routes=routes,
                                   set_route=set_route, device=device, log=log)
        logits_fn = GraphLogits(graphs, device, route_key=(lambda route: route) if routed
                                else (lambda route: None))
        record.update({"graph_count": len(graphs.graphs), "graph_lengths": list(graphs.lengths),
                       "graph_capture_seconds": round(graphs.capture_seconds, 2)})
    record["apply_seconds"] = round(time.monotonic() - started, 2)
    return Serving(model=model, processor=processor, adapter_context=adapter_context,
                   inference_context=inference_context, logits_fn=logits_fn,
                   readout_fn=readout_fn, record=record, image_token_cap=cap,
                   image_token_cap_default=cap_default)


__all__ = [
    "DecoderGraphs", "DeviceReadout", "EXACT_OPTIONS", "GRAPH_LENGTHS", "GraphLogits",
    "IMAGE_TOKEN_CAP", "IMAGE_TOKEN_CAP_DEFAULT", "OPTIONS", "PARAMETERISED_OPTIONS",
    "image_token_cap", "image_token_cap_default",
    "PixelProcessor", "RouteState", "RouteSwitch", "Serving", "ServingOptionError",
    "apply_serving_options", "install_route_modules", "is_exact", "pad_positions",
    "parse_options", "route_mode",
]
