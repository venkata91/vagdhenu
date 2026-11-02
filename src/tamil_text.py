"""Tamil text frontend: fixture-format parser (+ model-text normalisation, below).

Fixture format (UTF-8, NFC): '#' comments, '@key: value' header before the first section, then sections
'[invocation]' / '[taniyan N]' / '[pasuram N]' / '[adivaravu]' / '[colophon]'. In pasuram/taniyan sections
each non-blank line is one recitation segment (the edition's ⋆-segment); the repeat mark (header
'@repeat_mark:', default '‡' = recited twice) may only open the first segment and is removed from the
stored line (Section.marked keeps it); a trailing '॥ N ॥' must equal the section number.
See README "Tamil (experimental)"."""
import os, re, unicodedata
from dataclasses import dataclass, field

KINDS = ("invocation", "taniyan", "pasuram", "adivaravu", "colophon")
NUMBERED = ("taniyan", "pasuram")
REQUIRED_META = ("title", "work_id", "source")
DEFAULT_REPEAT_MARK = "‡"
_NUM = re.compile(r"^\d+(\.\d+)*$")
_HEADER = re.compile(r"^\[(\w+)(?:\s+(\S+))?\]$")
_META = re.compile(r"^@(\w+):\s*(.*)$")
_END_NUM = re.compile(r"॥\s*([\d.]+)\s*॥\s*$")


class FixtureError(ValueError):
    pass


@dataclass
class Section:
    kind: str
    num: str = None
    marked: bool = False
    lines: list = field(default_factory=list)
    line_nos: list = field(default_factory=list)
    header_line: int = 0


@dataclass
class Work:
    meta: dict
    path: str
    sections: list

    def recited(self, kinds=("pasuram",)):
        return [s for s in self.sections if s.kind in kinds]


def display_text(s):
    return unicodedata.normalize("NFC", s).strip()


def parse_file(path):
    with open(path, encoding="utf-8") as f:
        return parse_text(f.read(), path)


def parse_text(text, path="<text>"):
    def err(line, msg):
        return FixtureError(f"{path}:{line}: {msg}")

    meta, sections, cur = {}, [], None
    last_num = {}   # kind -> (prefix, last component) of the previous numbered section

    def close(sec):
        if sec is not None and sec.kind in NUMBERED and not sec.lines:
            raise err(sec.header_line, f"empty pāsuram [{sec.kind} {sec.num}]")

    lines = unicodedata.normalize("NFC", text).split("\n")
    for ln, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _META.match(line)
        if m:
            if cur is not None:
                raise err(ln, f"header '@{m.group(1)}' after the first section")
            meta[m.group(1)] = m.group(2).split(" #")[0].strip()
            continue
        m = _HEADER.match(line)
        if m:
            kind, num = m.group(1), m.group(2)
            if kind not in KINDS:
                raise err(ln, f"unknown section '[{kind}]' (expected one of {', '.join(KINDS)})")
            if cur is None:
                for k in REQUIRED_META:
                    if not meta.get(k):
                        raise err(ln, f"missing required header '{k}'")
                if not re.fullmatch(r"[a-z0-9_]+", meta["work_id"]):
                    raise err(ln, f"work_id '{meta['work_id']}' must match [a-z0-9_]+")
                if "repeat_mark" in meta and not meta["repeat_mark"]:
                    raise err(ln, "'@repeat_mark:' must not be empty")
                mark = meta.get("repeat_mark", DEFAULT_REPEAT_MARK)
            close(cur)
            if kind in NUMBERED:
                if num is None:
                    raise err(ln, f"[{kind}] requires a number, e.g. [{kind} 1.1.1]")
                if not _NUM.match(num):
                    raise err(ln, f"bad number '{num}' (expected digits separated by '.')")
                if any(s.kind == kind and s.num == num for s in sections):
                    raise err(ln, f"duplicate [{kind} {num}]")
                prefix, _, last = num.rpartition(".")
                prev = last_num.get(kind)
                if prev and prev[0] == prefix and int(last) != prev[1] + 1:
                    want = f"{prefix}.{prev[1] + 1}" if prefix else str(prev[1] + 1)
                    raise err(ln, f"numbering gap: [{kind} {num}] follows {prev[2]}, missing {want}")
                last_num[kind] = (prefix, int(last), num)
            elif num is not None:
                raise err(ln, f"[{kind}] takes no number")
            cur = Section(kind=kind, num=num, header_line=ln)
            sections.append(cur)
            continue
        if cur is None:
            raise err(ln, "text before the first section")
        if cur.kind in NUMBERED:
            opens = line.startswith(mark)
            if mark in line[len(mark) if opens else 0:] or (opens and cur.lines):
                raise err(ln, f"repeat mark '{mark}' is only allowed to open a section's first segment")
            if opens:
                cur.marked = True
                line = line[len(mark):].strip()
                if not line:
                    raise err(ln, f"empty segment after repeat mark '{mark}'")
            m = _END_NUM.search(line)
            if m and m.group(1) != cur.num:
                raise err(ln, f"trailing '॥ {m.group(1)} ॥' does not match [{cur.kind} {cur.num}]")
        cur.lines.append(line)
        cur.line_nos.append(ln)
    close(cur)
    if cur is None:
        raise err(len(lines), "no sections")
    return Work(meta=meta, path=path, sections=sections)


# ── model text (what reaches IndicF5): native Tamil script, no transliteration, no lexicon ──────────
_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "reference_bank", "vocab.txt"), encoding="utf-8") as _f:
    VOCAB = frozenset(t for t in _f.read().split("\n") if len(t) == 1)
# Tamil model text may only use Tamil-block vocab letters: Devanagari/Latin are in the IndicF5 vocab but
# inside a Tamil segment they mean mixed-script input, which is rejected rather than read by the model.
ALLOWED = frozenset(c for c in VOCAB if "\u0B80" <= c <= "\u0BFF")

MARKUP = set("⋆‡॥।|!?,.;\"'()–—-")
_DIGITS = re.compile(r"[0-9\u0BE6-\u0BEF\u0966-\u096F]")
# OOV hook: ஶ is not in the IndicF5 vocab -> nearest sibilant ஸ; Grantha-convention sub/superscript
# numerals (க₃ = ga) are dropped (native Tamil reading); Latin ':' (visarga stand-in, e.g. நம:) dropped.
_OOV_MAP = str.maketrans({"ஶ": "ஸ", ":": None,
                          **{c: None for c in "₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹"}})


class TamilTextError(ValueError):
    pass


def model_text(s, where=""):
    t = unicodedata.normalize("NFC", s)
    t = _END_NUM.sub(" ", t)
    t = "".join(" " if c in MARKUP else c for c in t)
    t = _DIGITS.sub(" ", t)
    t = t.translate(_OOV_MAP)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        raise TamilTextError(f"{where}: empty after normalisation")
    bad = sorted({c for c in t if c != " " and c not in ALLOWED})
    if bad:
        raise TamilTextError(f"{where}: unsupported characters " + ", ".join(f"U+{ord(c):04X} '{c}'" for c in bad))
    return t
