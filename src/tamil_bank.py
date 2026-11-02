"""Tamil reference bank (src/reference_bank/bank_ta.json): resolve a Tamil clip's reference, measure sps.

Kept apart from bank.json so Tamil entries never reach Sanskrit meter lookup, its vasantatilaka fallback or the
demo dropdown. Every problem raises TamilBankError naming the entry and wav path; there is no fallback.

  python src/tamil_bank.py sps --wav ta_santhai.wav --ref_text 'பல்லாண்டு பல்லாண்டு'"""
import os, sys, json, argparse
from collections import namedtuple
import soundfile as sf
import tamil_text as TT, tamil_syll as TS

SPS_MARGIN = 1.10      # bank convention: measured sec/unit x 1.10
MAX_REF_S = 15.0       # half-reference limit

TaRef = namedtuple("TaRef", "key wav ref_text model_ref_text sps unit", defaults=("syllable",))
PACE_FIELD = {"syllable": "sec_per_syll", "matrai": "sec_per_matrai"}


class TamilBankError(RuntimeError):
    pass


def load_bank(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _has_tamil_letter(s):
    return any("\u0b85" <= c <= "\u0bb9" for c in s)


def resolve_ref(bank, bank_dir, clip):
    """Reference for a Tamil clip: bank entry clip["meter"], with clip ref_wav / ref_text / sps overriding it
    (a ref_wav override must bring its own ref_text).
    clip["dur_unit"] ('syllable' default, or 'matrai') picks the entry's pace field (sec_per_syll /
    sec_per_matrai); a clip sps is in that unit. A clip sps of 0 means speed-based duration (as for Sanskrit);
    otherwise the entry's pace field must be > 0."""
    key = clip["meter"]
    unit = clip.get("dur_unit", "syllable")
    if unit not in PACE_FIELD:
        raise TamilBankError(f"tamil ref '{key}': dur_unit must be one of {tuple(PACE_FIELD)} (got {unit!r})")
    e = bank.get(key)
    if key.startswith("_") or not isinstance(e, dict):
        raise TamilBankError(f"tamil ref '{key}': no such entry in the Tamil bank")
    if e.get("lang") != "ta":
        raise TamilBankError(f"tamil ref '{key}': entry is not lang 'ta'")
    if "ref_wav" in clip and "ref_text" not in clip:   # the entry's transcript can't describe other audio
        raise TamilBankError(f"tamil ref '{key}': ref_wav override needs ref_text ({clip['ref_wav']})")
    wav = clip["ref_wav"] if "ref_wav" in clip else os.path.join(bank_dir, e.get("wav", ""))
    if not os.path.isfile(wav):
        raise TamilBankError(f"tamil ref '{key}': wav not found: {wav}")
    ref_text = clip.get("ref_text", e.get("ref_text") or "")
    if not ref_text.strip():
        raise TamilBankError(f"tamil ref '{key}': ref_text is empty ({wav})")
    try:
        model_ref = TT.model_text(ref_text, where=f"ref_text of '{key}'")
    except TT.TamilTextError as err:
        raise TamilBankError(f"tamil ref '{key}': ref_text is not Tamil ({wav}): {err}") from None
    if not _has_tamil_letter(model_ref):
        raise TamilBankError(f"tamil ref '{key}': ref_text is not Tamil ({wav}): no Tamil letter")
    if "sps" in clip:
        sps = float(clip["sps"])
    else:
        field = PACE_FIELD[unit]
        sps = e.get(field)
        if isinstance(sps, bool) or not isinstance(sps, (int, float)) or sps <= 0:
            raise TamilBankError(f"tamil ref '{key}': {field} not measured ({wav}); "
                                 f"run `python src/tamil_bank.py sps` or pass a clip sps")
        sps = float(sps)
    return TaRef(key, wav, ref_text, model_ref, sps, unit)


def provisional_sps(wav, ref_text):
    info = sf.info(wav)
    dur = info.frames / info.samplerate
    nsyll = TS.n_syllables(ref_text)
    if nsyll == 0:
        raise TamilBankError(f"ref_text has no Tamil syllables: {ref_text!r}")
    warning = (f"clip is {dur:.2f} s, over the {MAX_REF_S:.0f} s half-reference limit; trim it"
               if dur > MAX_REF_S else None)
    nmatrai = TS.n_matrai(ref_text)
    return {"dur_s": round(dur, 3), "nsyll": nsyll, "sps": round(dur / nsyll * SPS_MARGIN, 3),
            "nmatrai": nmatrai, "spm": round(dur / nmatrai * SPS_MARGIN, 3), "warning": warning}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tamil_bank", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sps", help="provisional sec_per_syll (sps) and sec_per_matrai (spm) = duration / count x 1.10")
    s.add_argument("--wav", required=True); s.add_argument("--ref_text", required=True)
    a = ap.parse_args(argv)
    try:
        r = provisional_sps(a.wav, a.ref_text)
    except (TamilBankError, RuntimeError, OSError) as e:
        sys.exit(f"tamil_bank: error: {e}")
    print(json.dumps(r, ensure_ascii=False))
    if r["warning"]:
        print(f"WARNING: {r['warning']}", file=sys.stderr)


if __name__ == "__main__":
    main()
