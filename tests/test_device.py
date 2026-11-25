"""--device auto|cuda|mps|cpu: selection order and that render.py hands the device to the model."""
from types import SimpleNamespace
import pytest
import device as DEV
from test_render_tamil import clip, tmp_bank_ta
from conftest import load_render


def fake_torch(cuda, mps):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                           backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


@pytest.mark.parametrize("cuda, mps, want", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_auto_prefers_cuda_then_mps_then_cpu(cuda, mps, want):
    assert DEV.pick_device("auto", fake_torch(cuda, mps)) == want


def test_torch_without_mps_backend_falls_back_to_cpu():
    t = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace())
    assert DEV.pick_device("auto", t) == "cpu"


@pytest.mark.parametrize("req", ["cuda", "mps"])
def test_unavailable_explicit_device_fails_loudly(req):
    with pytest.raises(SystemExit, match=f"--device {req}"):
        DEV.pick_device(req, fake_torch(False, False))


def test_cpu_is_always_allowed():
    assert DEV.pick_device("cpu", fake_torch(True, True)) == "cpu"


def test_render_passes_device_to_model_vocoder_and_infer(tmp_path):
    mod, st = load_render(tmp_path / "r", ["--bank_ta", tmp_bank_ta(tmp_path), "--device", "cpu"])
    assert mod.DEVICE == "cpu"
    assert mod.load_model.call_args.kwargs["device"] == "cpu"
    assert mod.load_vocoder.call_args.kwargs["device"] == "cpu"
    mod.render_clip(clip())
    assert st.infer_devices == ["cpu"]
