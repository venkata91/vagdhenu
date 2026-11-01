"""Tamil text frontend: fixture-format parser (+ model-text normalisation, below).

Fixture format (UTF-8, NFC): '#' comments, '@key: value' header before the first section, then sections
'[invocation]' / '[taniyan N]' / '[pasuram N]' / '[adivaravu]' / '[colophon]'. In pasuram/taniyan sections
each non-blank line is one recitation segment (the edition's ⋆-segment); '‡' may only open the first
segment; a trailing '॥ N ॥' must equal the section number. See README "Tamil (experimental)"."""
import re, unicodedata
from dataclasses import dataclass, field

KINDS = ("invocation", "taniyan", "pasuram", "adivaravu", "colophon")
NUMBERED = ("taniyan", "pasuram")
REQUIRED_META = ("title", "work_id", "source")
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
            if "‡" in line[1:] or (line.startswith("‡") and cur.lines):
                raise err(ln, "'‡' is only allowed as the first character of a section's first segment")
            if line.startswith("‡"):
                cur.marked = True
            m = _END_NUM.search(line)
            if m and m.group(1) != cur.num:
                raise err(ln, f"trailing '॥ {m.group(1)} ॥' does not match [{cur.kind} {cur.num}]")
        cur.lines.append(line)
        cur.line_nos.append(ln)
    close(cur)
    if cur is None:
        raise err(len(lines), "no sections")
    return Work(meta=meta, path=path, sections=sections)
