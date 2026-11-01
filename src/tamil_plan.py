"""Tamil recitation plan builder: fixture -> plan JSON + render.py shard (one entry per recited unit).

Modes: single  (one reciter unit per ⋆-segment)
       santhai (ācārya recites a segment, śiṣyas repeat it santhai_repeats times: T,S,S,...)
       goshti  (unison, single voice; config "alternate": groups A/B take turns per segment or pāsuram)
Gaps between units come from config["gaps"] by boundary type. Output is byte-deterministic.

  python src/tamil_plan.py --fixture examples/tamil/tiruppallandu.txt --mode santhai \\
      --unit_dir out/tamil_units --shard_out out/x.shard.json --plan_out out/x.plan.json"""
import os, sys, json, zlib, copy, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tamil_text as TT, tamil_syll as TS  # noqa: E402

FORMAT = "vagdhenu-tamil-plan/1"
SR = 24000
MODES = ("single", "santhai", "goshti")
DEFAULT_CONFIG = {
    "gaps": {"segment": 0.55, "pasuram_end": 0.95, "teacher_to_student": 0.8,
             "student_to_student": 0.6, "group_switch": 0.4},
    "santhai_repeats": 2,
    "alternate": False,
    "alternate_unit": "segment",
    "refs": {"reciter": "santhai", "acharya": "santhai", "shishya": "santhai",
             "goshti": "goshti", "A": "goshti", "B": "goshti"},
    "base_seed": 60,
    "repeat_marked": False,
    "sections": ["pasuram"],
}


class PlanConfigError(ValueError):
    pass


def load_config(path=None):
    """Defaults, overlaid with the JSON file at `path` (gaps/refs merged per key); validated."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg
    try:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
    except json.JSONDecodeError as e:
        raise PlanConfigError(f"{path}: invalid JSON ({e})")
    if not isinstance(user, dict):
        raise PlanConfigError(f"{path}: config must be a JSON object")
    for k, v in user.items():
        if k not in cfg:
            raise PlanConfigError(f"{path}: unknown config key '{k}'")
        if k in ("gaps", "refs"):
            if not isinstance(v, dict):
                raise PlanConfigError(f"{path}: '{k}' must be an object")
            if k == "gaps":
                for kk in v:
                    if kk not in cfg["gaps"]:
                        raise PlanConfigError(f"{path}: unknown gap kind '{kk}' (expected {', '.join(cfg['gaps'])})")
            cfg[k].update(v)
        else:
            cfg[k] = v
    _validate(cfg, path)
    return cfg


def _validate(cfg, where):
    def bad(msg):
        raise PlanConfigError(f"{where}: {msg}")
    for k, v in cfg["gaps"].items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            bad(f"gaps.{k} must be a number >= 0 (got {v!r})")
    r = cfg["santhai_repeats"]
    if isinstance(r, bool) or not isinstance(r, int) or r < 1:
        bad(f"santhai_repeats must be an integer >= 1 (got {r!r})")
    for k in ("alternate", "repeat_marked"):
        if not isinstance(cfg[k], bool):
            bad(f"{k} must be true/false")
    if cfg["alternate_unit"] not in ("segment", "pasuram"):
        bad(f"alternate_unit must be 'segment' or 'pasuram' (got {cfg['alternate_unit']!r})")
    for k, v in cfg["refs"].items():
        if not isinstance(v, str) or not v:
            bad(f"refs.{k} must be a non-empty bank key")
    if isinstance(cfg["base_seed"], bool) or not isinstance(cfg["base_seed"], int):
        bad("base_seed must be an integer")
    if not isinstance(cfg["sections"], list) or not cfg["sections"] \
            or any(s not in TT.NUMBERED for s in cfg["sections"]):
        bad(f"sections must be a non-empty list of {list(TT.NUMBERED)}")


def _roles(mode, k):
    if mode == "single":
        return [("reciter", 0)]
    if mode == "santhai":
        return [("acharya", 0)] + [("shishya", r) for r in range(1, k + 1)]
    return [("goshti", 0)]


def _gap_kind(mode, cfg, us, j):
    u = us[j]
    if j == len(us) - 1:
        return "pasuram_end"
    if mode == "santhai":
        if u["role"] == "acharya":
            return "teacher_to_student"
        return "student_to_student" if u["repeat"] < cfg["santhai_repeats"] else "segment"
    if mode == "goshti" and cfg["alternate"] and us[j + 1]["group"] != u["group"]:
        return "group_switch"
    return "segment"


def build_plan(work, mode, config, unit_dir):
    if mode not in MODES:
        raise PlanConfigError(f"unknown mode '{mode}' (expected {', '.join(MODES)})")
    _validate(config, "config")
    wid = work.meta["work_id"]
    alt = mode == "goshti" and config["alternate"]
    pasurams, shard = [], []
    for k, sec in enumerate(work.recited(tuple(config["sections"])), 1):
        tag = sec.num if sec.kind == "pasuram" else f"{sec.kind}{sec.num}"
        segs = [TT.display_text(l) for l in sec.lines]
        passes = (1, 2) if (config["repeat_marked"] and sec.marked) else (1,)
        us = []
        for ps in passes:
            for si, seg in enumerate(segs, 1):
                TT.model_text(seg, where=f"{work.path}:{sec.line_nos[si - 1]}")   # fail at plan time, not on the GPU
                for role, rep in _roles(mode, config["santhai_repeats"]):
                    group = None
                    if alt:
                        turn = si if config["alternate_unit"] == "segment" else k
                        group = "A" if turn % 2 else "B"
                    uid = f"{wid}_{tag}_s{si:02d}_{role}" + (f"_r{rep}" if role == "shishya" else "") \
                        + (f"_g{group}" if group else "") + ("_p2" if ps == 2 else "")
                    us.append({"id": uid, "seg": si, "role": role, "group": group, "repeat": rep, "pass": ps,
                               "ref": config["refs"][group if group else role],
                               "seed": config["base_seed"] + zlib.crc32(uid.encode()) % 100000,
                               "nsyll": TS.n_syllables(seg)})
        for j, u in enumerate(us):
            kind = _gap_kind(mode, config, us, j)
            u["gap_after_s"] = config["gaps"][kind]; u["gap_kind"] = kind
            shard.append({"id": u["id"], "meter": u["ref"], "padas": [segs[u["seg"] - 1]], "seed": u["seed"],
                          "no_sandhi": True, "out": os.path.join(unit_dir, u["id"] + ".wav"), "lang": "ta",
                          "mode": mode, "role": u["role"], "group": u["group"], "repeat": u["repeat"],
                          "pasuram": sec.num, "seg": u["seg"], "pass": u["pass"]})
        pasurams.append({"kind": sec.kind, "num": sec.num, "marked": sec.marked, "file": f"{wid}_{tag}.wav",
                         "segments": segs, "units": us})
    plan = {"format": FORMAT, "work_id": wid, "title": work.meta["title"], "fixture": os.path.basename(work.path),
            "mode": mode, "sr": SR, "config": config, "pasurams": pasurams}
    return plan, shard


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1) + "\n"


def _write(path, text):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="tamil_plan", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="modes:\n  single   one reciter unit per ⋆-segment\n"
               "  santhai  ācārya once, then śiṣyas repeat (config santhai_repeats, default 2)\n"
               "  goshti   unison, single voice; config alternate=true labels groups A/B "
               "(alternate_unit: segment|pasuram)\n"
               "config JSON keys (all optional): " + ", ".join(DEFAULT_CONFIG))
    ap.add_argument("--fixture", required=True, help="Tamil fixture .txt (see README 'Tamil (experimental)')")
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--config", default=None, help="JSON overriding DEFAULT_CONFIG keys")
    ap.add_argument("--unit_dir", default="out/tamil_units", help="where render.py should write unit wavs")
    ap.add_argument("--shard_out", default=None, help="default out/<work_id>.<mode>.shard.json")
    ap.add_argument("--plan_out", default=None, help="default out/<work_id>.<mode>.plan.json")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.config)
        work = TT.parse_file(a.fixture)
        plan, shard = build_plan(work, a.mode, cfg, a.unit_dir)
    except (PlanConfigError, TT.FixtureError, TT.TamilTextError, OSError) as e:
        sys.exit(f"tamil_plan: error: {e}")
    base = os.path.join("out", f"{plan['work_id']}.{a.mode}")
    shard_out = a.shard_out or base + ".shard.json"; plan_out = a.plan_out or base + ".plan.json"
    _write(shard_out, dumps(shard)); _write(plan_out, dumps(plan))
    os.makedirs(a.unit_dir, exist_ok=True)
    print(f"[tamil_plan] {plan['work_id']} {a.mode}: {len(plan['pasurams'])} pāsurams, {len(shard)} units "
          f"-> {shard_out}, {plan_out}")


if __name__ == "__main__":
    main()
