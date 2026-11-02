"""Deterministic snapshot of every Sanskrit-path output the Tamil work must not change (BC-1..BC-5).
compute() is pure CPU: render.py / render_core.py run under the conftest stubs."""
import os, sys, json, hashlib, inspect
from importlib.metadata import version

TESTS = os.path.dirname(os.path.abspath(__file__))
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)
import conftest  # noqa: E402  (installs the GPU-import blocker, puts src/ on sys.path)

REPO, SRC = conftest.REPO, conftest.SRC
INPUTS = os.path.join(TESTS, "golden", "sanskrit_inputs.json")
GOLDEN = os.path.join(TESTS, "golden", "sanskrit_golden.json")
BANK = os.path.join(SRC, "reference_bank", "bank.json")
ARGPARSE_KEY = "render_argparse_defaults"


def env():
    return {"indic_transliteration": version("indic_transliteration")}


def inputs():
    d = json.load(open(INPUTS, encoding="utf-8"))
    return [t for k in ("devanagari", "bank_full_verses", "kannada", "tamil_script") for t in d[k]]


def _try(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except Exception as e:  # an exception is a recorded behaviour too
        return f"ERR {type(e).__name__}: {e}"


def _norm_path(v):
    return v.replace(REPO, "<REPO>") if isinstance(v, str) else v


def _render_records(mod, st, clips):
    out = []
    for clip in clips:
        st.infer_calls.clear(); st.stitch_calls.clear()
        res = _try(mod.render_clip, clip)
        if isinstance(res, dict):
            res = {**res, "out": os.path.basename(res["out"])}
        out.append({"id": clip["id"], "result": res, "infer": list(st.infer_calls), "stitch": list(st.stitch_calls)})
    return out


def compute(tmp_dir):
    os.environ.pop("CHAMP_ROOT", None)
    import prep_text as PT, limits
    R, st = conftest.load_render(tmp_dir)
    RC = conftest.load_render_core()
    texts = inputs()

    per_input = []
    for t in texts:
        mt = _try(PT.model_text, t)
        mts = _try(PT.model_text_sandhi, t, echo_final=False)
        mts_echo = _try(PT.model_text_sandhi, t)
        rec = {
            "input": t,
            "model_text": mt, "model_text_sandhi": mts_echo, "model_text_sandhi_noecho": mts,
            "model_text_norm": _try(PT.model_text_norm, t), "mfa_text": _try(PT.mfa_text, t),
            "align_slp1": _try(PT.align_slp1, t), "detect_script": _try(PT.detect_script, t),
            "to_deva": _try(PT.to_deva, t),
            "detect_meter_key": _try(RC.detect_meter_key, t), "split_padas": _try(RC.split_padas, t),
            "n_aksharas_render": R.n_aksharas(t), "n_aksharas_core": RC.n_aksharas(t),
            "n_aksharas_limits": limits._n_aksharas(t), "validate_one_shloka": limits.validate_one_shloka(t),
            "chain": {},
        }
        for name, base in (("model_text", mt), ("model_text_sandhi", mts)):
            if base.startswith("ERR "):
                continue
            c = {}
            for mname, M in (("render", R), ("core", RC)):
                for fn in ("_satva", "_anusvara_m", "_danda_fix", "_hna_metathesis", "_vocalic_l"):
                    c[f"{mname}.{fn}"] = getattr(M, fn)(base)
                c[f"{mname}.n_aksharas"] = M.n_aksharas(base)
                c[f"{mname}.rep_depths"] = list(M._rep_depths(M._aksharas(base)))
                c[f"{mname}.ends_halant"] = M._ends_halant(base)
            c["core._visarga_ksha"] = RC._visarga_ksha(base)
            c["render.pipeline"] = R._vocalic_l(R._hna_metathesis(R._danda_fix(R._anusvara_m(
                R._satva(base) if name == "model_text_sandhi" else base))))
            c["core.pipeline"] = RC._vocalic_l(RC._hna_metathesis(RC._danda_fix(RC._visarga_ksha(RC._anusvara_m(
                RC._satva(base) if name == "model_text_sandhi" else base)))))
            rec["chain"][name] = c
        per_input.append(rec)

    sample = json.load(open(os.path.join(REPO, "examples", "sample_shard.json"), encoding="utf-8"))
    shard = [{"id": f"in{i:03d}_{'ns' if ns else 'sa'}", "meter": "anushtubh", "padas": [t], "seed": 60,
              "no_sandhi": ns, "out": "unused.wav"} for i, t in enumerate(texts) for ns in (True, False)]

    bank = json.load(open(BANK, encoding="utf-8"))
    keys = []
    for k, v in bank.items():
        if k.startswith("_") or not isinstance(v, dict) or "wav" not in v:
            continue
        keys += [k, v["wav"].replace(".wav", "")]
    refs = {}
    for k in keys + ["no_such_meter"]:
        r = _try(R.get_ref, k)
        refs[k] = r if isinstance(r, str) else [os.path.basename(r[0]), r[1], r[2], r[3]]

    return {
        "per_input": per_input,
        "render_sample_shard": _render_records(R, st, sample),
        "render_inputs_shard": _render_records(R, st, shard),
        "get_ref": refs,
        "lut_keys": sorted(R._lut),
        "bank_sha256": hashlib.sha256(open(BANK, "rb").read()).hexdigest(),
        ARGPARSE_KEY: {act.dest: _norm_path(act.default) for act in R.ap._actions},
        "renderer_signatures": {"__init__": str(inspect.signature(RC.Renderer.__init__)),
                                "render_one": str(inspect.signature(RC.Renderer.render_one)),
                                "meters": str(inspect.signature(RC.Renderer.meters))},
    }
