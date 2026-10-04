"""Serial inference helpers; no model imports or GPU synchronization when untimed."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from time import perf_counter

PHASES = ("decode_load", "processor_preprocess", "tokenise_chat_template",
          "route_adapter_switch", "h2d_copy", "forward", "readout_calibration", "response_build")


class PhaseTimer:
    def __init__(self, seconds=None, synchronize=None):
        self.seconds, self.synchronize = seconds, synchronize
        if seconds is not None:
            seconds.update(dict.fromkeys(PHASES, 0.0))

    def phase(self, name):
        return self._phase(name) if self.seconds is not None else nullcontext()

    @contextmanager
    def _phase(self, name):
        if self.synchronize is not None:
            self.synchronize()
        started = perf_counter()
        try:
            yield
        finally:
            if self.synchronize is not None:
                self.synchronize()
            self.seconds[name] += perf_counter() - started


class AdapterSwitch:
    """Keep PEFT's selected adapter fixed; toggle layers only on route transitions.

    Owns adapter state for a serial reader. Do not mutate adapters externally.
    """
    def __init__(self, model):
        from open_decisions.image_jev import router
        self.model = model
        self.layers = model.base_model
        model.set_adapter(router.SCREEN_GEOMETRY)
        self.route = router.SCREEN_GEOMETRY

    def __call__(self, route):
        from open_decisions.image_jev import router
        if route not in router.ROUTES:
            raise ValueError("unknown adapter route")
        if route != self.route:
            if route == router.SCREEN_GEOMETRY:
                self.layers.enable_adapter_layers()
            else:
                self.layers.disable_adapter_layers()
            self.route = route
        return nullcontext()


def move_to_device(value, device):
    if device is None or not hasattr(value, "to"):
        return value
    if getattr(value, "device", None) == device:
        return value
    # Pin only CUDA-bound host tensors. Already-device pixels need no copy.
    if getattr(device, "type", str(device).split(":")[0]) == "cuda":
        if getattr(getattr(value, "device", None), "type", None) == "cpu":
            value = value.pin_memory()
        return value.to(device, non_blocking=True)
    return value.to(device)


class NativeCache:
    """Per-processor caches for the immutable letter ids and chat shell."""
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = CachedTokenizer(processor.tokenizer)
        processor.tokenizer = self.tokenizer
        self.letters = None
        self.shell = None

    def prompt(self, user):
        from open_decisions.image_jev.vision import IMAGE_SYSTEM
        processor = self.processor
        if not hasattr(processor, "apply_chat_template"):
            return user
        def render(text):
            return processor.apply_chat_template(
                [{"role": "system", "content": IMAGE_SYSTEM}, {"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if self.shell is None:
            marker = "IMAGE_HOPPER_STATIC_SHELL_56"
            sample = render(marker)
            parts = sample.split(marker)
            # Verify the shell against the actual first request (templates may inspect content).
            self.shell = tuple(parts) if len(parts) == 2 and "".join((parts[0], user, parts[1])) == render(user) else False
        if self.shell is False:
            return render(user)
        self.tokenizer.configure(*self.shell)
        return self.shell[0] + user + self.shell[1]

    def letter_ids(self, count):
        from open_decisions.scoring.prompt import letter_token_ids
        if self.letters is None:
            self.letters = letter_token_ids(self.processor.tokenizer)
        return self.letters[:count]


class CachedTokenizer:
    """Cache Qwen chat tokens only at isolated added-special-token boundaries.

    Unknown kwargs, tokenizer families and shapes use the original tokenizer.
    The first full encoding also verifies exact token-id equality.
    """
    def __init__(self, tokenizer):
        self.original = tokenizer
        self.parts = None
        self.timer = PhaseTimer()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def configure(self, prefix, suffix):
        if self.parts is not None or not getattr(self.original, "is_fast", False):
            return
        if "Qwen" not in type(self.original).__name__ or "<|im_start|>" not in prefix:
            return
        boundary = prefix.rfind("<|im_start|>") + len("<|im_start|>")
        prefix = prefix[:boundary]
        if not suffix.startswith("<|im_end|>"):
            return
        tokens = {str(token): token for token in self.original.added_tokens_decoder.values()}
        for key in ("<|im_start|>", "<|im_end|>"):
            token = tokens.get(key)
            if token is None or not token.special or token.lstrip or token.rstrip:
                return
        if self.original.num_special_tokens_to_add(pair=False):
            return
        self.parts = (prefix, suffix, self.original.encode(prefix, add_special_tokens=False),
                      self.original.encode(suffix, add_special_tokens=False), False)

    def __call__(self, text, **kwargs):
        with self.timer.phase("tokenise_chat_template"):
            allowed = {"padding", "return_attention_mask", "return_token_type_ids", "add_special_tokens", "return_tensors"}
            if (not self.parts or set(kwargs) - allowed or kwargs.get("return_token_type_ids") or
                    not isinstance(text, list) or len(text) != 1 or not isinstance(text[0], str)):
                return self.original(text, **kwargs)
            prefix, suffix, before, after, verified = self.parts
            value = text[0]
            if not value.startswith(prefix) or not value.endswith(suffix):
                return self.original(text, **kwargs)
            middle = value[len(prefix):-len(suffix)]
            ids = before + self.original.encode(middle, add_special_tokens=False) + after
            if not verified:
                reference = self.original(text, **kwargs)
                expected = reference["input_ids"]
                if hasattr(expected, "tolist"):
                    expected = expected.tolist()
                if expected != [ids]:
                    self.parts = False
                    return reference
                self.parts = (prefix, suffix, before, after, True)
                return reference
            return self.original.pad({"input_ids": [ids]}, padding=kwargs.get("padding", False),
                                     return_attention_mask=kwargs.get("return_attention_mask"),
                                     return_tensors=kwargs.get("return_tensors"))
