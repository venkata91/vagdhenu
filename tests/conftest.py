"""CPU test harness: stub the GPU stack (torch/torchaudio/f5_tts/bigvgan) so src/render.py and
src/render_core.py can be executed without models, and block any *real* import of those packages."""
import os, sys, json, types, itertools, importlib.abc, importlib.util
from unittest.mock import MagicMock
import numpy as np, soundfile as sf

TESTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TESTS)
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

BLOCKED = ("torch", "torchaudio", "f5_tts", "bigvgan")


class _BlockGpuImports(importlib.abc.MetaPathFinder):
    """RN-4: only reached when no stub is in sys.modules, i.e. a real import was attempted."""
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"real import of '{name}' is blocked in tests (CPU-only, RN-4)")
        return None


if not any(isinstance(f, _BlockGpuImports) for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockGpuImports())


class Stubs:
    """Records what the render path hands to the (stubbed) model."""
    def __init__(self):
        self.infer_calls = []
        self.preprocess_calls = []
        self.stitch_calls = []

    def infer_process(self, ref_audio, ref_text, gen_text, model, vocoder, mel_spec_type="vocos", speed=1.0,
                      nfe_step=32, cfg_strength=2.0, device=None, fix_duration=None, **kw):
        self.infer_calls.append({"ref_audio": os.path.basename(ref_audio), "ref_text": ref_text,
                                 "gen_text": gen_text, "fix_duration": fix_duration, "speed": speed})
        return np.full(24000, 0.1, np.float32), 24000, None

    def preprocess_ref_audio_text(self, ref_audio, ref_text, clip_short=True, **kw):
        self.preprocess_calls.append((os.path.basename(ref_audio), ref_text))
        return ref_audio, ref_text

    @staticmethod
    def ta_load(path):
        n = sf.info(path).frames if os.path.exists(path) else 4 * 24000
        return np.zeros((1, n), np.float32), 24000


def install_stubs():
    st = Stubs()
    utils = types.ModuleType("f5_tts.infer.utils_infer")
    utils.load_model = MagicMock(name="load_model")
    utils.load_vocoder = MagicMock(name="load_vocoder")
    utils.infer_process = st.infer_process
    utils.preprocess_ref_audio_text = st.preprocess_ref_audio_text
    f5 = types.ModuleType("f5_tts"); infer = types.ModuleType("f5_tts.infer")
    model = types.ModuleType("f5_tts.model"); model.DiT = MagicMock(name="DiT")
    f5.infer = infer; f5.model = model; infer.utils_infer = utils
    ta = types.ModuleType("torchaudio"); ta.load = st.ta_load
    sys.modules.update({"torch": MagicMock(name="torch"), "torchaudio": ta, "bigvgan": MagicMock(name="bigvgan"),
                        "f5_tts": f5, "f5_tts.infer": infer, "f5_tts.infer.utils_infer": utils,
                        "f5_tts.model": model})
    return st


_seq = itertools.count()


def _exec(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_render(tmp_dir, extra_argv=()):
    """Execute src/render.py (an argv-driven script) on an empty shard under stubs; returns (module, stubs).
    bvgan is replaced by a constant-signal stub and _stitch is wrapped to record gap lengths + gate flags."""
    st = install_stubs()
    tmp_dir = str(tmp_dir); os.makedirs(tmp_dir, exist_ok=True)
    shard = os.path.join(tmp_dir, "empty_shard.json")
    with open(shard, "w") as f:
        json.dump([], f)
    old_argv = sys.argv
    sys.argv = ["render.py", "--shard", shard, "--results", os.path.join(tmp_dir, "r.json"),
                "--outdir", tmp_dir, *extra_argv]
    try:
        mod = _exec(os.path.join(SRC, "render.py"), f"_render_under_test_{next(_seq)}")
    finally:
        sys.argv = old_argv
    mod.bvgan = lambda mel: np.full(12000, 0.2, np.float32)
    real_stitch = mod._stitch

    def _stitch(segs, GAPS, fric=False, halant=False):
        st.stitch_calls.append({"gaps": [len(g) for g in GAPS], "fric": fric, "halant": halant})
        return real_stitch(segs, GAPS, fric=fric, halant=halant)
    mod._stitch = _stitch
    return mod, st


def load_render_core():
    install_stubs()
    return _exec(os.path.join(SRC, "render_core.py"), f"_render_core_under_test_{next(_seq)}")
