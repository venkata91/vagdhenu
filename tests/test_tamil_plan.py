"""RS-1..RS-8: recitation plan builder (single / santhai / goshti [+alternate]) and its CLI."""
import os, sys, json, zlib, subprocess
import pytest
import tamil_text as TT, tamil_syll as TS, tamil_plan as TP

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO, "examples", "tamil", "tiruppallandu.txt")
WORK = TT.parse_file(FIXTURE)
SEGS = [(p.num, i, l) for p in WORK.recited() for i, l in enumerate(p.lines, 1)]


def build(mode, **cfg):
    config = TP.load_config(None)
    for k, v in cfg.items():
        config[k] = {**config[k], **v} if isinstance(v, dict) else v
    return TP.build_plan(WORK, mode, config, "out/u")


def units(plan):
    return [u for p in plan["pasurams"] for u in p["units"]]


def test_single_is_one_reciter_unit_per_segment_in_order():
    plan, shard = build("single")
    us = units(plan)
    assert len(us) == 91 and {u["role"] for u in us} == {"reciter"}
    assert [(p["num"], u["seg"]) for p in plan["pasurams"] for u in p["units"]] == [(n, i) for n, i, _ in SEGS]
    assert [s["padas"] for s in shard] == [[l] for _, _, l in SEGS]
    assert {u["ref"] for u in us} == {"santhai"}


@pytest.mark.parametrize("k", [2, 3])
def test_santhai_teacher_then_students(k):
    plan, _ = build("santhai", santhai_repeats=k)
    us = units(plan)
    assert len(us) == (1 + k) * 91
    for j in range(0, len(us), 1 + k):
        block = us[j:j + 1 + k]
        assert [(u["role"], u["repeat"]) for u in block] == [("acharya", 0)] + [("shishya", r) for r in range(1, k + 1)]
        assert len({u["seg"] for u in block}) == 1


def test_goshti_single_voice_no_groups():
    plan, _ = build("goshti")
    us = units(plan)
    assert len(us) == 91 and {u["role"] for u in us} == {"goshti"} and {u["group"] for u in us} == {None}
    assert {u["ref"] for u in us} == {"goshti"}


def test_goshti_alternate_per_segment_restarts_each_pasuram():
    plan, _ = build("goshti", alternate=True, refs={"A": "ga", "B": "gb"})
    for p in plan["pasurams"]:
        assert [u["group"] for u in p["units"]] == ["AB"[(u["seg"] - 1) % 2] for u in p["units"]]
        assert p["units"][0]["group"] == "A"
        assert all(u["ref"] == {"A": "ga", "B": "gb"}[u["group"]] for u in p["units"])
        assert all(u["id"].endswith("_g" + u["group"]) for u in p["units"])


def test_goshti_alternate_per_pasuram_defaults_refs_to_goshti():
    plan, _ = build("goshti", alternate=True, alternate_unit="pasuram")
    for k, p in enumerate(plan["pasurams"], 1):
        assert {u["group"] for u in p["units"]} == {"A" if k % 2 else "B"}
    assert {u["ref"] for u in units(plan)} == {"goshti"}


GAPS = {"segment": 0.11, "pasuram_end": 0.22, "teacher_to_student": 0.33, "student_to_student": 0.44,
        "group_switch": 0.55}


@pytest.mark.parametrize("mode, cfg", [("single", {}), ("santhai", {}), ("santhai", {"santhai_repeats": 3}),
                                       ("goshti", {}), ("goshti", {"alternate": True}),
                                       ("goshti", {"alternate": True, "alternate_unit": "pasuram"})])
def test_every_gap_equals_its_configured_kind(mode, cfg):
    plan, _ = build(mode, gaps=GAPS, **cfg)
    k = plan["config"]["santhai_repeats"]
    for p in plan["pasurams"]:
        us = p["units"]
        for j, u in enumerate(us):
            assert u["gap_after_s"] == GAPS[u["gap_kind"]]
            if j == len(us) - 1:
                want = "pasuram_end"
            elif mode == "santhai":
                want = {"acharya": "teacher_to_student"}.get(u["role"]) or \
                    ("student_to_student" if u["repeat"] < k else "segment")
            elif cfg.get("alternate"):
                want = "group_switch" if us[j + 1]["group"] != u["group"] else "segment"
            else:
                want = "segment"
            assert u["gap_kind"] == want, u["id"]


def test_plan_is_deterministic_with_unique_ids_and_formula_seeds():
    a = [TP.dumps(x) for x in build("santhai")]
    b = [TP.dumps(x) for x in build("santhai")]
    assert a == b
    plan, shard = build("santhai", base_seed=7)
    ids = [u["id"] for u in units(plan)]
    assert len(set(ids)) == len(ids) and ids == [s["id"] for s in shard]
    assert ids[:3] == ["tiruppallandu_1.1.1_s01_acharya", "tiruppallandu_1.1.1_s01_shishya_r1",
                       "tiruppallandu_1.1.1_s01_shishya_r2"]
    assert all(u["seed"] == 7 + zlib.crc32(u["id"].encode()) % 100000 for u in units(plan))
    assert all(u["nsyll"] == TS.n_syllables(seg) for p in plan["pasurams"] for u in p["units"]
               for seg in [p["segments"][u["seg"] - 1]])


def test_shard_entries_are_render_compatible_and_additive():
    plan, shard = build("goshti", alternate=True)
    for s in shard:
        assert {"id", "meter", "padas", "seed", "no_sandhi", "out"} <= set(s)
        assert s["lang"] == "ta" and s["no_sandhi"] is True and isinstance(s["seed"], int)
        assert s["out"] == os.path.join("out/u", s["id"] + ".wav")
        assert set(s) <= {"id", "meter", "padas", "seed", "no_sandhi", "out", "lang", "mode", "role", "group",
                          "repeat", "pasuram", "seg", "pass"}


def test_repeat_marked_adds_second_pass_for_marked_pasurams_only():
    plan, _ = build("single", repeat_marked=True)
    p2 = {u["id"].split("_")[1] for u in units(plan) if u["id"].endswith("_p2")}
    assert p2 == {"1.1.1", "1.1.2", "1.1.11", "1.1.12"}
    assert len(units(plan)) == 91 + 4 + 8 + 8 + 8
    assert all(u["pass"] == 2 for u in units(plan) if u["id"].endswith("_p2"))


def run_cli(*args):
    return subprocess.run([sys.executable, os.path.join(REPO, "src", "tamil_plan.py"), *args],
                          capture_output=True, text=True, cwd=REPO)


def test_cli_help_lists_modes():
    r = run_cli("--help")
    assert r.returncode == 0
    for m in ("single", "santhai", "goshti", "alternate"):
        assert m in r.stdout


def test_cli_rejects_bad_mode():
    r = run_cli("--fixture", FIXTURE, "--mode", "bogus")
    assert r.returncode == 2 and "invalid choice" in r.stderr


@pytest.mark.parametrize("cfg, msg", [
    ('{"bogus": 1}', "unknown config key 'bogus'"),
    ('{"gaps": {"segment": -0.1}}', "gaps.segment"),
    ('{"gaps": {"nope": 0.1}}', "unknown gap kind 'nope'"),
    ('{"santhai_repeats": 0}', "santhai_repeats"),
    ('{"alternate_unit": "word"}', "alternate_unit"),
    ('{not json', "invalid JSON"),
])
def test_cli_rejects_bad_config(tmp_path, cfg, msg):
    p = tmp_path / "c.json"; p.write_text(cfg)
    r = run_cli("--fixture", FIXTURE, "--mode", "santhai", "--config", str(p),
                "--shard_out", str(tmp_path / "s.json"), "--plan_out", str(tmp_path / "p.json"))
    assert r.returncode == 1 and r.stderr.startswith("tamil_plan: error:") and msg in r.stderr
    assert not (tmp_path / "s.json").exists()


def test_committed_example_matches_fresh_cli_output(tmp_path):
    r = run_cli("--fixture", "examples/tamil/tiruppallandu.txt", "--mode", "santhai", "--unit_dir",
                "out/tamil_units", "--shard_out", str(tmp_path / "s.json"), "--plan_out", str(tmp_path / "p.json"))
    assert r.returncode == 0, r.stderr
    for got, ref in (("s.json", "tiruppallandu.santhai.shard.json"), ("p.json", "tiruppallandu.santhai.plan.json")):
        assert (tmp_path / got).read_bytes() == open(os.path.join(REPO, "examples", "tamil", ref), "rb").read()
    assert len(json.loads((tmp_path / "s.json").read_text())) == 273
