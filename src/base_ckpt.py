"""Map the base ai4bharat/IndicF5 model.safetensors onto the CFM model built by render.py.

The base checkpoint differs from our .pt voices in three ways (TECH_REPORT E29, "Base IndicF5"):
keys carry an `ema_model._orig_mod.` prefix, they may lack the CFM `transformer.` prefix, and GRN
stores `weight`/`bias` where the pinned f5_tts uses `gamma`/`beta`. Loading with strict=False would
silently skip the GRN tensors (they stay zero-init and the text encoder garbles), so every model
tensor must be matched or this raises."""
PREFIXES = ("ema_model.", "_orig_mod.")
GRN = {".grn.weight": ".grn.gamma", ".grn.bias": ".grn.beta"}


class BaseCkptError(RuntimeError):
    pass


def _strip(k):
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if k.startswith(p):
                k = k[len(p):]; changed = True
    return k


def remap(sd, model_keys, param_keys):
    """{ckpt key: tensor} -> {model key: tensor} (model_keys = state_dict keys); raises if any learned
    parameter under `transformer.` (param_keys = named_parameters names) is left unfilled."""
    model_keys = set(model_keys)
    out = {}
    for k, v in sd.items():
        if k in ("initted", "step"):
            continue
        k = _strip(k)
        for old, new in GRN.items():
            if k.endswith(old):
                k = k[: -len(old)] + new
        if k not in model_keys and "transformer." + k in model_keys:
            k = "transformer." + k
        if k in model_keys:
            out[k] = v
    missing = sorted(k for k in param_keys if k.startswith("transformer.") and k not in out)
    if missing:
        raise BaseCkptError(f"base checkpoint is missing {len(missing)} model tensors, e.g. {missing[:3]}")
    return out
