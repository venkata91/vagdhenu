"""Assemble rendered Tamil unit wavs into one wav per pāsuram + a manifest (CPU only).

pāsuram wav = concat over its units of [unit audio, silence(gap_after_s)] (the trailing pāsuram-end gap is
kept so pāsurams can be played back to back). Every unit is validated (exists, 24 kHz, mono) before anything
is written; each pāsuram is then built and written to *.partial in turn (one in memory at a time), and all
outputs are moved into place with os.replace only once every one has been written.

  python src/tamil_assemble.py --plan out/x.plan.json --units_dir out/tamil_units --outdir out/tamil"""
import os, sys, json, argparse
import numpy as np, soundfile as sf

FORMAT = "vagdhenu-tamil-manifest/1"
SR = 24000


class AssemblyError(RuntimeError):
    def __init__(self, unit_id, msg):
        super().__init__(f"unit {unit_id}: {msg}")
        self.unit_id = unit_id


def _check_units(plan, units_dir):
    paths = {}
    for p in plan["pasurams"]:
        for u in p["units"]:
            path = os.path.join(units_dir, u["id"] + ".wav")
            if not os.path.exists(path):
                raise AssemblyError(u["id"], f"missing {path}")
            info = sf.info(path)
            if info.samplerate != SR:
                raise AssemblyError(u["id"], f"sr {info.samplerate} != {SR} ({path})")
            if info.channels != 1:
                raise AssemblyError(u["id"], f"{info.channels} channels, expected mono ({path})")
            paths[u["id"]] = path
    return paths


def assemble(plan, units_dir, outdir):
    paths = _check_units(plan, units_dir)
    os.makedirs(outdir, exist_ok=True)
    mpath = os.path.join(outdir, f"{plan['work_id']}.{plan['mode']}.manifest.json")
    entries, staged = [], []   # staged: (partial, final); one pāsuram in memory at a time
    try:
        for p in plan["pasurams"]:
            parts, units, pos = [], [], 0
            for u in p["units"]:
                y, _ = sf.read(paths[u["id"]], dtype="float32")
                gap = np.zeros(round(u["gap_after_s"] * SR), np.float32)
                units.append({"id": u["id"], "seg": u["seg"], "role": u["role"], "group": u["group"],
                              "repeat": u["repeat"], "start_s": round(pos / SR, 3),
                              "end_s": round((pos + len(y)) / SR, 3), "start": pos, "end": pos + len(y)})
                parts += [y, gap]; pos += len(y) + len(gap)
            final = os.path.join(outdir, p["file"]); staged.append((final + ".partial", final))
            sf.write(staged[-1][0], np.concatenate(parts), SR, format="WAV", subtype="PCM_16")
            entries.append({"num": p["num"], "marked": p["marked"], "mode": plan["mode"], "file": p["file"],
                            "display_text": p["segments"], "duration_s": round(pos / SR, 3), "samples": pos,
                            "units": units})
        manifest = {"format": FORMAT, "work_id": plan["work_id"], "title": plan["title"], "mode": plan["mode"],
                    "sr": SR, "pasurams": entries}
        staged.append((mpath + ".partial", mpath))
        with open(staged[-1][0], "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    except BaseException:
        for tmp, _ in staged:
            if os.path.exists(tmp): os.remove(tmp)
        raise
    for tmp, final in staged:
        os.replace(tmp, final)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tamil_assemble", description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True, help="plan JSON from tamil_plan.py")
    ap.add_argument("--units_dir", required=True, help="dir holding <unit id>.wav from render.py")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args(argv)
    try:
        with open(a.plan, encoding="utf-8") as f:
            plan = json.load(f)
        man = assemble(plan, a.units_dir, a.outdir)
    except (AssemblyError, OSError, json.JSONDecodeError) as e:
        sys.exit(f"tamil_assemble: error: {e}")
    print(f"[tamil_assemble] {len(man['pasurams'])} pāsuram wavs + manifest -> {a.outdir}")


if __name__ == "__main__":
    main()
