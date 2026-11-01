"""AS-1..AS-3: CPU assembly of unit wavs into per-pāsuram wavs + manifest."""
import os, sys, json, subprocess
import numpy as np, soundfile as sf
import pytest
import tamil_text as TT, tamil_plan as TP, tamil_assemble as TA

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALL = ("@title: t\n@work_id: w\n@source: s\n[pasuram 1]\nபல்லாண்டு ⋆\nபலகோடி ⋆\nமணிவண்ணா ॥ 1 ॥\n"
         "[pasuram 2]\nஅடியோமோடும் ⋆\nவடிவாய் ॥ 2 ॥\n")


def make(tmp_path, mode="santhai", skip=None, **cfg):
    config = TP.load_config(None); config.update(cfg)
    plan, _ = TP.build_plan(TT.parse_text(SMALL, "small.txt"), mode, config, str(tmp_path / "u"))
    rng = np.random.default_rng(0); (tmp_path / "u").mkdir(parents=True, exist_ok=True); lens = {}
    for p in plan["pasurams"]:
        for u in p["units"]:
            if u["id"] == skip:
                continue
            n = int(rng.integers(1000, 30000)); lens[u["id"]] = n
            sf.write(str(tmp_path / "u" / (u["id"] + ".wav")), rng.uniform(-.5, .5, n).astype(np.float32), 24000)
    return plan, lens


@pytest.mark.parametrize("mode, cfg", [("santhai", {}), ("goshti", {"alternate": True}), ("single", {})])
def test_pasuram_length_is_units_plus_gaps_and_manifest_matches(tmp_path, mode, cfg):
    plan, lens = make(tmp_path, mode, **cfg)
    man = TA.assemble(plan, str(tmp_path / "u"), str(tmp_path / "o"))
    assert man == json.load(open(tmp_path / "o" / f"w.{mode}.manifest.json", encoding="utf-8"))
    assert [p["num"] for p in man["pasurams"]] == ["1", "2"]
    for p, mp in zip(plan["pasurams"], man["pasurams"]):
        y, sr = sf.read(str(tmp_path / "o" / mp["file"]))
        assert sr == 24000 and y.ndim == 1
        want = sum(lens[u["id"]] + round(u["gap_after_s"] * 24000) for u in p["units"])
        assert abs(len(y) - want) <= 1 and mp["samples"] == len(y)
        assert abs(mp["duration_s"] - len(y) / 24000) < 1e-3
        assert mp["mode"] == mode and mp["display_text"] == p["segments"] and mp["file"] == p["file"]
        pos = 0
        for u, mu in zip(p["units"], mp["units"]):
            assert (mu["id"], mu["role"], mu["group"]) == (u["id"], u["role"], u["group"])
            assert mu["start"] == pos and mu["end"] - mu["start"] == lens[u["id"]]
            assert abs(mu["start_s"] - mu["start"] / 24000) < 1e-3 and abs(mu["end_s"] - mu["end"] / 24000) < 1e-3
            pos = mu["end"] + round(u["gap_after_s"] * 24000)
        assert pos == len(y)


def _no_outputs(d):
    return not d.exists() or not [f for f in os.listdir(d) if f.endswith((".wav", ".partial", ".json"))]


def test_missing_unit_raises_and_writes_nothing(tmp_path):
    plan, _ = make(tmp_path, skip="w_2_s02_shishya_r1")
    with pytest.raises(TA.AssemblyError, match="w_2_s02_shishya_r1") as e:
        TA.assemble(plan, str(tmp_path / "u"), str(tmp_path / "o"))
    assert e.value.unit_id == "w_2_s02_shishya_r1"
    assert _no_outputs(tmp_path / "o")


def test_non_24k_unit_is_rejected(tmp_path):
    plan, _ = make(tmp_path)
    bad = tmp_path / "u" / "w_2_s01_acharya.wav"
    sf.write(str(bad), np.zeros(2205, np.float32), 22050)
    with pytest.raises(TA.AssemblyError, match=r"unit w_2_s01_acharya: sr 22050 != 24000"):
        TA.assemble(plan, str(tmp_path / "u"), str(tmp_path / "o"))
    assert _no_outputs(tmp_path / "o")


def test_stereo_unit_is_rejected(tmp_path):
    plan, _ = make(tmp_path)
    sf.write(str(tmp_path / "u" / "w_1_s01_acharya.wav"), np.zeros((100, 2), np.float32), 24000)
    with pytest.raises(TA.AssemblyError, match="w_1_s01_acharya: 2 channels"):
        TA.assemble(plan, str(tmp_path / "u"), str(tmp_path / "o"))


def test_cli_writes_wavs_and_manifest(tmp_path):
    plan, _ = make(tmp_path)
    (tmp_path / "p.json").write_text(TP.dumps(plan), encoding="utf-8")
    r = subprocess.run([sys.executable, os.path.join(REPO, "src", "tamil_assemble.py"), "--plan",
                        str(tmp_path / "p.json"), "--units_dir", str(tmp_path / "u"), "--outdir", str(tmp_path / "o")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(tmp_path / "o")) == ["w.santhai.manifest.json", "w_1.santhai.wav", "w_2.santhai.wav"]
    os.remove(tmp_path / "u" / "w_1_s01_acharya.wav")
    r = subprocess.run([sys.executable, os.path.join(REPO, "src", "tamil_assemble.py"), "--plan",
                        str(tmp_path / "p.json"), "--units_dir", str(tmp_path / "u"), "--outdir", str(tmp_path / "o2")],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "tamil_assemble: error: unit w_1_s01_acharya" in r.stderr


def test_two_modes_share_an_outdir(tmp_path):
    p1, _ = make(tmp_path / "a", "santhai")
    p2, _ = make(tmp_path / "b", "goshti")
    m1 = TA.assemble(p1, str(tmp_path / "a" / "u"), str(tmp_path / "o"))
    m2 = TA.assemble(p2, str(tmp_path / "b" / "u"), str(tmp_path / "o"))
    assert not {p["file"] for p in m1["pasurams"]} & {p["file"] for p in m2["pasurams"]}
    for m in (m1, m2):
        assert m == json.load(open(tmp_path / "o" / f"w.{m['mode']}.manifest.json", encoding="utf-8"))
        for p in m["pasurams"]:
            assert sf.info(str(tmp_path / "o" / p["file"])).frames == p["samples"]
