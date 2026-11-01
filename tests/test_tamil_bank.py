"""BK-1..BK-5: Tamil reference bank resolution, sps helper, isolation from the Sanskrit bank."""
import os, sys, json, hashlib, subprocess
import numpy as np, soundfile as sf
import pytest
import tamil_bank as TB, tamil_text as TT
from conftest import load_render

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RB = os.path.join(REPO, "src", "reference_bank")


def wav(path, seconds, sr=24000):
    sf.write(str(path), np.zeros(int(seconds * sr), np.float32), sr)
    return str(path)


def tmp_bank(tmp_path, **over):
    wav(tmp_path / "ta_santhai.wav", 3.0)
    e = {"lang": "ta", "mode": "santhai", "wav": "ta_santhai.wav", "ref_text": "பல்லாண்டு பல்லாண்டு ⋆",
         "sec_per_syll": 0.3}
    e.update(over)
    return {"_doc": "x", "santhai": e, "sk": {"wav": "anushtubh.wav", "ref_text": "राम"}}


def test_resolves_filled_entry(tmp_path):
    r = TB.resolve_ref(tmp_bank(tmp_path), str(tmp_path), {"meter": "santhai"})
    assert r == TB.TaRef("santhai", str(tmp_path / "ta_santhai.wav"), "பல்லாண்டு பல்லாண்டு ⋆",
                         "பல்லாண்டு பல்லாண்டு", 0.3)


def test_clip_overrides_entry(tmp_path):
    other = wav(tmp_path / "mine.wav", 2.0)
    r = TB.resolve_ref(tmp_bank(tmp_path), str(tmp_path),
                       {"meter": "santhai", "ref_wav": other, "ref_text": "வடிவாய்", "sps": 0.25})
    assert (r.wav, r.ref_text, r.model_ref_text, r.sps) == (other, "வடிவாய்", "வடிவாய்", 0.25)


@pytest.mark.parametrize("key, over, clip, pat", [
    ("nope", {}, {}, r"tamil ref 'nope': no such entry in the Tamil bank"),
    ("sk", {}, {}, r"tamil ref 'sk': entry is not lang 'ta'"),
    ("santhai", {"wav": "gone.wav"}, {}, r"tamil ref 'santhai': wav not found: .*gone\.wav"),
    ("santhai", {"ref_text": ""}, {}, r"tamil ref 'santhai': ref_text is empty \(.*ta_santhai\.wav\)"),
    ("santhai", {"ref_text": "राम राम"}, {}, r"tamil ref 'santhai': ref_text is not Tamil.*U\+0930"),
    ("santhai", {"ref_text": "॥ 1 ॥"}, {}, r"tamil ref 'santhai': ref_text is not Tamil.*empty"),
    ("santhai", {"sec_per_syll": None}, {}, r"tamil ref 'santhai': sec_per_syll not measured.*ta_santhai\.wav"),
    ("santhai", {"sec_per_syll": 0}, {}, r"tamil ref 'santhai': sec_per_syll not measured"),
    ("santhai", {}, {"ref_wav": "nowhere.wav"}, r"tamil ref 'santhai': wav not found: nowhere\.wav"),
])
def test_resolve_errors_name_key_and_path(tmp_path, key, over, clip, pat):
    bank = tmp_bank(tmp_path, **over)
    with pytest.raises(TB.TamilBankError, match=pat):
        TB.resolve_ref(bank, str(tmp_path), dict(clip, meter=key))


def test_clip_sps_fills_unmeasured_entry(tmp_path):
    r = TB.resolve_ref(tmp_bank(tmp_path, sec_per_syll=None), str(tmp_path), {"meter": "santhai", "sps": 0.31})
    assert r.sps == 0.31


def test_shipped_bank_parses_and_both_entries_fail_loudly():
    path = os.path.join(RB, "bank_ta.json")
    bank = TB.load_bank(path)
    assert "_template" in bank and "_doc" in bank
    for key in ("santhai", "goshti"):
        e = bank[key]
        assert e["lang"] == "ta" and e["mode"] == key and e["sec_per_syll"] is None
        with pytest.raises(TB.TamilBankError, match=rf"tamil ref '{key}': wav not found: .*ta_{key}\.wav"):
            TB.resolve_ref(bank, RB, {"meter": key})


def test_no_tamil_wavs_ship():
    assert not [f for f in os.listdir(RB) if f.startswith("ta_")]


def test_provisional_sps(tmp_path):
    r = TB.provisional_sps(wav(tmp_path / "a.wav", 6.0), "பல்லாண்டு பல்லாண்டு")
    assert r == {"dur_s": 6.0, "nsyll": 6, "sps": round(6.0 / 6 * 1.10, 3), "warning": None}
    r = TB.provisional_sps(wav(tmp_path / "b.wav", 16.0), "பல்லாண்டு பல்லாண்டு")
    assert r["dur_s"] == 16.0 and "15" in r["warning"]
    with pytest.raises(TB.TamilBankError, match="no Tamil syllables"):
        TB.provisional_sps(str(tmp_path / "a.wav"), "॥ 1 ॥")


def test_sps_cli(tmp_path):
    p = wav(tmp_path / "a.wav", 6.0)
    r = subprocess.run([sys.executable, os.path.join(REPO, "src", "tamil_bank.py"), "sps", "--wav", p,
                        "--ref_text", "பல்லாண்டு பல்லாண்டு"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["sps"] == 1.1


def test_sanskrit_bank_untouched_and_tamil_keys_invisible(tmp_path):
    blob = subprocess.run(["git", "-C", REPO, "rev-parse", "c18927a:src/reference_bank/bank.json"],
                          capture_output=True, text=True, check=True).stdout.strip()
    now = subprocess.run(["git", "-C", REPO, "hash-object", os.path.join(RB, "bank.json")],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert now == blob
    ta_keys = {k for k in TB.load_bank(os.path.join(RB, "bank_ta.json")) if not k.startswith("_")}
    ta_wavs = {"ta_santhai", "ta_goshti"}
    mod, _ = load_render(tmp_path)
    assert not (set(mod._lut) & (ta_keys | ta_wavs))
    bank = json.load(open(os.path.join(RB, "bank.json"), encoding="utf-8"))
    demo_meters = [k for k, v in bank.items() if not k.startswith("_") and isinstance(v, dict) and "wav" in v]
    assert not (set(demo_meters) & ta_keys)   # same filter as demo METERS and Renderer.meters()
    assert not any(isinstance(v, dict) and v.get("lang") == "ta" for v in bank.values())
