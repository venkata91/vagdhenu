"""Tamil eḻuttu / syllable count / mātrai count / asai (nēr-nirai) analysis. Pure stdlib.

n_syllables = vowel nuclei (uyir + uyirmey); mey (consonant + ்) and āytam (ஃ) are not counted. This is
the Tamil counterpart of render.py's n_aksharas and drives fix_duration = ref_len + n * sec_per_syll.
n_matrai = traditional length in mātrai: kuṟil 1, neṭil 2, oṟṟu / āytam ½ (duration_unit "matrai").
asai: nēr = one kuṟil/neṭil (+ following oṟṟu); nirai = kuṟil + kuṟil/neṭil (+ following oṟṟu).
Not modelled: kuṟṟiyalukaram, aikārakkuṟukkam, aḷapeṭai."""

PULLI = "\u0BCD"
AYTAM = "ஃ"
KURIL_VOWELS = set("அஇஉஎஒ")
KURIL_SIGNS = set("ிுெொ")
NEDIL_SIGNS = set("ாீூேைோௌ")


def _is_vowel(c):
    return "\u0B85" <= c <= "\u0B94"


def _is_cons(c):
    return "\u0B95" <= c <= "\u0BB9"


def eluttu(word):
    """Letters of a Tamil string as (letter, 'kuril'|'nedil'|'ottru'); non-letters are skipped."""
    out = []; i = 0; n = len(word)
    while i < n:
        c = word[i]; nxt = word[i + 1] if i + 1 < n else ""
        if _is_vowel(c):
            out.append((c, "kuril" if c in KURIL_VOWELS else "nedil")); i += 1
        elif _is_cons(c):
            if nxt == PULLI:
                out.append((c + nxt, "ottru")); i += 2
            elif nxt in KURIL_SIGNS:
                out.append((c + nxt, "kuril")); i += 2
            elif nxt in NEDIL_SIGNS:
                out.append((c + nxt, "nedil")); i += 2
            else:
                out.append((c, "kuril")); i += 1
        elif c == AYTAM:
            out.append((c, "ottru")); i += 1
        else:
            i += 1
    return out


def n_syllables(s):
    return sum(1 for _, k in eluttu(s) if k != "ottru")


MATRAI = {"kuril": 1.0, "nedil": 2.0, "ottru": 0.5}
DURATION_UNITS = ("syllable", "matrai")


def n_matrai(s):
    return sum(MATRAI[k] for _, k in eluttu(s))


def count_units(s, unit):
    """Duration units of `s`: 'syllable' (vowel nuclei) or 'matrai'."""
    if unit not in DURATION_UNITS:
        raise ValueError(f"duration unit must be one of {DURATION_UNITS} (got {unit!r})")
    return n_matrai(s) if unit == "matrai" else n_syllables(s)


def asai(word):
    """Greedy left-to-right nēr/nirai split of one word -> [(chunk, 'ner'|'nirai')]."""
    L = eluttu(word); out = []; i = 0; n = len(L)

    def take_ottru(j):
        chunk = ""
        while j < n and L[j][1] == "ottru":
            chunk += L[j][0]; j += 1
        return chunk, j

    while i < n:
        letter, kind = L[i]
        if kind == "ottru":   # stray leading oṟṟu: attach to the previous asai
            if out: out[-1] = (out[-1][0] + letter, out[-1][1])
            i += 1; continue
        if kind == "kuril" and i + 1 < n and L[i + 1][1] != "ottru":
            tail, j = take_ottru(i + 2)
            out.append((letter + L[i + 1][0] + tail, "nirai")); i = j
        else:
            tail, j = take_ottru(i + 1)
            out.append((letter + tail, "ner")); i = j
    return out
