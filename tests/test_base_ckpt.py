"""Base IndicF5 safetensors -> CFM key remap (prefixes, transformer., GRN weight/bias -> gamma/beta)."""
import pytest
import base_ckpt as BC

MODEL = ["transformer.text_embed.text_blocks.0.grn.gamma", "transformer.text_embed.text_blocks.0.grn.beta",
         "transformer.proj_out.weight", "transformer.rotary_embed.inv_freq", "mel_spec.mel_stft.window"]
PARAMS = MODEL[:3]


def test_remaps_prefixes_and_grn():
    sd = {"ema_model._orig_mod.transformer.text_embed.text_blocks.0.grn.weight": 1,
          "ema_model._orig_mod.text_embed.text_blocks.0.grn.bias": 2,
          "ema_model._orig_mod.proj_out.weight": 3, "initted": 0, "step": 9, "unrelated.x": 4}
    assert BC.remap(sd, MODEL, PARAMS) == {"transformer.text_embed.text_blocks.0.grn.gamma": 1,
                                          "transformer.text_embed.text_blocks.0.grn.beta": 2,
                                          "transformer.proj_out.weight": 3}


def test_missing_parameter_raises_instead_of_silent_zero_init():
    sd = {"ema_model._orig_mod.proj_out.weight": 3}
    with pytest.raises(BC.BaseCkptError, match="missing 2 model tensors"):
        BC.remap(sd, MODEL, PARAMS)
