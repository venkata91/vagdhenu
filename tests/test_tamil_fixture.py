"""FX-1/2/3/5/6: Tiruppallāṇḍu fixture contents and the fixture-format parser."""
import os, unicodedata
import pytest
import tamil_text as TT

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "tamil",
                       "tiruppallandu.txt")


@pytest.fixture(scope="module")
def work():
    return TT.parse_file(FIXTURE)


def test_fixture_is_nfc_utf8():
    raw = open(FIXTURE, encoding="utf-8").read()
    assert unicodedata.normalize("NFC", raw) == raw


def test_header_records_provenance(work):
    for k in ("title", "author", "source", "credit", "retrieved", "license", "note"):
        assert work.meta.get(k), k
    assert work.meta["work_id"] == "tiruppallandu"
    assert "prapatti.com" in work.meta["source"] and "Sunder Kid" in work.meta["credit"]
    assert "public domain" in work.meta["license"] and "word spaces restored" in work.meta["license"]
    assert "Taniyans are absent" in work.meta["note"]


def test_twelve_pasurams_with_marks_and_segment_counts(work):
    ps = work.recited()
    assert [p.num for p in ps] == [f"1.1.{i}" for i in range(1, 13)]
    assert [p.num for p in ps if p.marked] == ["1.1.1", "1.1.2", "1.1.11", "1.1.12"]
    assert [len(p.lines) for p in ps] == [4, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8]
    assert sum(len(p.lines) for p in ps) == 91
    for p in ps:
        assert p.lines[-1].endswith(f"॥ {p.num} ॥")
        assert all(l.endswith("⋆") for l in p.lines[:-1])
        assert sum("‡" in l for l in p.lines) == (1 if p.marked else 0)


def test_non_recited_sections_are_labelled_and_excluded(work):
    kinds = [s.kind for s in work.sections]
    assert kinds[0] == "invocation" and kinds[-2:] == ["adivaravu", "colophon"]
    assert all(s.kind == "pasuram" for s in work.recited())
    inv = next(s for s in work.sections if s.kind == "invocation")
    assert any("ஶ" in l and "₃" in l for l in inv.lines)
    col = next(s for s in work.sections if s.kind == "colophon")
    assert col.lines == ["திருப்பல்லாண்டு முற்றிற்று", "பெரியாழ்வார் திருவடிகளே சரணம்"]


def test_taniyan_sections_are_opt_in():
    w = TT.parse_text(HEAD + "[taniyan 1]\nஒரு வரி\n\n[pasuram 1]\nஇரு வரி ॥ 1 ॥\n", "x.txt")
    assert [s.kind for s in w.recited()] == ["pasuram"]
    assert [s.kind for s in w.recited(("taniyan", "pasuram"))] == ["taniyan", "pasuram"]


HEAD = "@title: t\n@work_id: w\n@source: s\n"


@pytest.mark.parametrize("body, line, msg", [
    ("[bogus]\nஅ\n", 4, "unknown section"),
    ("[pasuram 1]\nஅ\n[pasuram 1]\nஆ\n", 6, "duplicate"),
    ("[pasuram]\nஅ\n", 4, "requires a number"),
    ("[pasuram 1.1.1]\nஅ\n[pasuram 1.1.2]\nஆ\n[pasuram 1.1.4]\nஇ\n", 8, "missing 1.1.3"),
    ("[pasuram 1]\n\n[pasuram 2]\nஅ\n", 4, "empty pāsuram"),
    ("[pasuram 1]\nஅ ⋆\n‡ஆ\n", 6, "‡"),
    ("[pasuram 1]\nஅ ⋆\nஆ ॥ 2 ॥\n", 6, "॥ 2 ॥"),
    ("அ\n[pasuram 1]\nஆ\n", 4, "before the first section"),
])
def test_malformed_files_raise_with_file_and_line(body, line, msg):
    with pytest.raises(TT.FixtureError) as e:
        TT.parse_text(HEAD + body, "bad.txt")
    assert f"bad.txt:{line}:" in str(e.value) and msg in str(e.value)


def test_missing_required_header_key():
    with pytest.raises(TT.FixtureError, match=r"bad.txt:\d+: missing required header 'source'"):
        TT.parse_text("@title: t\n@work_id: w\n[pasuram 1]\nஅ\n", "bad.txt")
