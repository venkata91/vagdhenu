"""RN-1..RN-4, BK-3 (render level): Tamil clips through render.render_clip under the stub harness."""
import os, sys, json
import numpy as np, soundfile as sf
import pytest
import tamil_text as TT, tamil_syll as TS
from conftest import load_render

PADA = "‡பல்லாண்டு பல்லாண்டு பல்லாயிரத்தாண்டு ⋆"


def tmp_bank_ta(tmp_path, sps=0.3):
    d = tmp_path / "bank"; d.mkdir(exist_ok=True)
    sf.write(str(d / "ta_santhai.wav"), np.zeros(3 * 24000, np.float32), 24000)
    bank = {"santhai": {"lang": "ta", "mode": "santhai", "wav": "ta_santhai.wav",
                        "ref_text": "பல்லாண்டு பல்லாண்டு ⋆", "sec_per_syll": sps}}
    (d / "bank_ta.json").write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
    return str(d / "bank_ta.json")


def clip(**kw):
    c = {"id": "t_1.1.1_s01_acharya", "meter": "santhai", "padas": [PADA], "seed": 7, "no_sandhi": True,
         "out": "unused.wav", "lang": "ta", "mode": "santhai", "role": "acharya"}
    c.update(kw)
    return c


def test_tamil_clip_gets_model_text_and_tamil_duration(tmp_path):
    mod, st = load_render(tmp_path / "r", ["--bank_ta", tmp_bank_ta(tmp_path)])
    r = mod.render_clip(clip())
    want = TT.model_text(PADA)
    n = TS.n_syllables(want)
    assert n == 12
    assert [c["gen_text"] for c in st.infer_calls] == [want]
    call = st.infer_calls[0]
    assert call["fix_duration"] == pytest.approx(3.0 + n * 0.3)
    assert (call["ref_audio"], call["ref_text"]) == ("ta_santhai.wav", "பல்லாண்டு பல்லாண்டு")
    assert st.preprocess_calls == [("ta_santhai.wav", "பல்லாண்டு பல்லாண்டு")]
    assert st.stitch_calls == [{"gaps": [int(mod.a.gap * 24000)], "fric": False, "halant": False}]
    assert r["lang"] == "ta" and r["pieces"] == 1 and r["seed"] == 7
    assert os.path.exists(r["out"]) and r["out"].endswith("t_1.1.1_s01_acharya.wav")
    assert not any("\u0c80" <= ch <= "\u0cff" or "\u0900" <= ch <= "\u097f" for ch in call["gen_text"])


def test_tamil_text_skips_sanskrit_postprocessing(tmp_path, monkeypatch):
    mod, st = load_render(tmp_path / "r", ["--bank_ta", tmp_bank_ta(tmp_path)])
    spied = []
    for name in ("_danda_fix", "_anusvara_m", "_hna_metathesis", "_vocalic_l", "_satva", "n_aksharas"):
        if hasattr(mod, name):
            monkeypatch.setattr(mod, name, lambda *x, _n=name, **k: spied.append(_n))
    monkeypatch.setattr(mod.PT, "align_slp1", lambda *x, **k: spied.append("align_slp1") or "")
    monkeypatch.setattr(mod.PT, "model_text", lambda *x, **k: spied.append("PT.model_text") or "")
    mod.render_clip(clip(no_sandhi=False))
    assert spied == [] and st.infer_calls[0]["gen_text"] == TT.model_text(PADA)


def test_clip_overrides_reach_infer(tmp_path):
    mod, st = load_render(tmp_path / "r", ["--bank_ta", tmp_bank_ta(tmp_path)])
    mine = tmp_path / "mine.wav"; sf.write(str(mine), np.zeros(2 * 24000, np.float32), 24000)
    mod.render_clip(clip(ref_wav=str(mine), ref_text="வடிவாய் ⋆", sps=0.5, speed=0.8))
    c = st.infer_calls[0]
    assert (c["ref_audio"], c["ref_text"], c["speed"]) == ("mine.wav", "வடிவாய்", 0.8)
    assert c["fix_duration"] == pytest.approx(2.0 + 12 * 0.5)


def test_sps_zero_override_is_speed_based(tmp_path):
    mod, st = load_render(tmp_path / "r", ["--bank_ta", tmp_bank_ta(tmp_path)])
    mod.render_clip(clip(sps=0))
    assert st.infer_calls[0]["fix_duration"] is None


def run_shard(tmp_path, clips, extra=()):
    shard = tmp_path / "shard.json"; shard.write_text(json.dumps(clips, ensure_ascii=False), encoding="utf-8")
    mod, st = load_render(tmp_path / "r", ["--shard", str(shard), *extra])
    return json.load(open(tmp_path / "r" / "r.json", encoding="utf-8")), st


def test_shipped_bank_fails_loudly_without_sanskrit_fallback(tmp_path):
    res, st = run_shard(tmp_path, [clip(), clip(id="g", meter="goshti")])
    assert "santhai" in res[0]["error"] and os.path.join("reference_bank", "ta_santhai.wav") in res[0]["error"]
    assert "goshti" in res[1]["error"] and "ta_goshti.wav" in res[1]["error"]
    assert st.infer_calls == [] and st.preprocess_calls == []


def test_unknown_tamil_key_errors_instead_of_vasantatilaka(tmp_path):
    res, st = run_shard(tmp_path, [clip(meter="vasantatilaka"), clip(id="u", meter="nope")],
                        ["--bank_ta", tmp_bank_ta(tmp_path)])
    assert "tamil ref 'vasantatilaka': no such entry" in res[0]["error"]
    assert "tamil ref 'nope': no such entry" in res[1]["error"]
    assert st.infer_calls == [] and st.preprocess_calls == []


def test_sanskrit_clip_still_uses_sanskrit_path(tmp_path):
    mod, st = load_render(tmp_path / "r", ["--bank_ta", tmp_bank_ta(tmp_path)])
    r = mod.render_clip({"id": "s", "meter": "anushtubh", "padas": ["रामो राजमणिः सदा विजयते"], "seed": 60,
                         "no_sandhi": True, "out": "x.wav"})
    assert "lang" not in r and st.infer_calls[0]["ref_audio"] == "anushtubh.wav"


def test_no_gpu_module_is_really_imported():
    for m in ("torch", "torchaudio", "f5_tts", "bigvgan"):
        assert m not in sys.modules or not hasattr(sys.modules[m], "__file__") or sys.modules[m].__file__ is None


def test_out_dir_is_created_before_inference(tmp_path):
    mod, st = load_render(tmp_path / "r", ["--bank_ta", tmp_bank_ta(tmp_path), "--outdir", ""])
    out = tmp_path / "new" / "units" / "x.wav"
    real = mod.infer_process
    def infer(*a, **k):
        assert out.parent.is_dir()
        return real(*a, **k)
    mod.infer_process = infer
    r = mod.render_clip(clip(out=str(out)))
    assert r["out"] == str(out) and out.exists()


def test_readme_sequence_runs_into_fresh_dir(tmp_path):
    """README "Running it": plan CLI -> render.py (stubbed) -> assemble CLI, nothing pre-created."""
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fresh = tmp_path / "fresh"
    r = subprocess.run([sys.executable, os.path.join(repo, "src", "tamil_plan.py"), "--fixture",
                        os.path.join(repo, "examples", "tamil", "tiruppallandu.txt"), "--mode", "single",
                        "--unit_dir", str(fresh / "units"), "--plan_out", str(fresh / "p.json"),
                        "--shard_out", str(fresh / "s.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    mod, st = load_render(tmp_path / "r", ["--outdir", "", "--bank_ta", tmp_bank_ta(tmp_path)])
    shard = json.load(open(fresh / "s.json", encoding="utf-8"))
    assert len(shard) == 91
    for c in shard:   # the harness patches bvgan after import, so drive render_clip per clip
        mod.render_clip(c)
    r = subprocess.run([sys.executable, os.path.join(repo, "src", "tamil_assemble.py"), "--plan", str(fresh / "p.json"),
                        "--units_dir", str(fresh / "units"), "--outdir", str(fresh / "tamil")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len([f for f in os.listdir(fresh / "tamil") if f.endswith(".wav")]) == 12
