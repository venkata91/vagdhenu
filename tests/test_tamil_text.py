"""TF-1..TF-6: native-Tamil model text (no transliteration), markup stripping, OOV hook, vocab gate."""
import os, re
import pytest
import tamil_text as TT

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "tamil",
                       "tiruppallandu.txt")
SEGMENTS = [(f"{p.num}#{i}", l) for p in TT.parse_file(FIXTURE).recited() for i, l in enumerate(p.lines, 1)]
TAMIL_LETTER = re.compile(r"[\u0B80-\u0BFF]")


def test_fixture_has_91_segments():
    assert len(SEGMENTS) == 91


@pytest.mark.parametrize("where, seg", SEGMENTS)
def test_segment_model_text_is_in_vocab_and_keeps_tamil_letters(where, seg):
    mt = TT.model_text(seg, where=where)
    assert set(mt) <= TT.VOCAB | {" "} and set(mt) <= TT.ALLOWED | {" "}
    assert TAMIL_LETTER.findall(mt) == TAMIL_LETTER.findall(seg.replace("ஶ", "ஸ"))
    for mark in "⋆‡॥!–—0123456789":
        assert mark not in mt
    assert "  " not in mt and mt == mt.strip()


def test_display_text_keeps_markup():
    seg = "‡பல்லாண்டு பல்லாண்டு பல்லாயிரத்தாண்டு ⋆ "
    assert TT.display_text(seg) == seg.strip()
    last = "உன் சேவடி செவ்வி திருக்காப்பு ॥ 1.1.1 ॥"
    assert TT.display_text(last) == last
    assert TT.model_text(last) == "உன் சேவடி செவ்வி திருக்காப்பு"
    assert TT.model_text("தொண்டக் குலத்தில் உள்ளீர் ! வந்தடி – தொழுது — ⋆") == "தொண்டக் குலத்தில் உள்ளீர் வந்தடி தொழுது"


def test_oov_hook_grantha_conventions():
    assert TT.model_text("ஶ்ரீமதே நிக₃மாந்த மஹாதே₃ஶிகாய நம:") == "ஸ்ரீமதே நிகமாந்த மஹாதேஸிகாய நம"
    assert TT.model_text("க²க³க⁴ க₂ க¹") == "ககக க க"


@pytest.mark.parametrize("bad, chars", [
    ("அ\U00011336", ["U+11336"]),
    ("நம नम", ["U+0928 'न'", "U+092E 'म'"]),
    ("abc பல்", ["U+0061 'a'", "U+0062 'b'", "U+0063 'c'"]),
])
def test_out_of_vocab_raises_with_chars_and_where(bad, chars):
    with pytest.raises(TT.TamilTextError) as e:
        TT.model_text(bad, where="seg-7")
    msg = str(e.value)
    assert msg.startswith("seg-7:")
    for c in chars:
        assert c in msg


@pytest.mark.parametrize("empty", ["", "   ", "⋆ ‡ ॥", "॥ 1.1.1 ॥", "௧௨", "१२ 34"])
def test_empty_after_normalisation_raises(empty):
    with pytest.raises(TT.TamilTextError, match="empty after normalisation"):
        TT.model_text(empty, where="x")


@pytest.mark.parametrize("word", ["நமோ நாராயணாய", "பரமாத்மனை", "இருடீகேசன்", "ழ ள ற ன ண ந"])
def test_sanskrit_words_in_tamil_script_pass_through(word):
    assert TT.model_text(word) == word
