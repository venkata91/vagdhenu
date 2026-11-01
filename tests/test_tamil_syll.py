"""SY-1..SY-3: vowel-nucleus syllable count and asai (nēr/nirai) segmentation."""
import os
import pytest
import tamil_text as TT, tamil_syll as TS

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "tamil",
                       "tiruppallandu.txt")
SEGMENTS = [l for p in TT.parse_file(FIXTURE).recited() for l in p.lines]


@pytest.mark.parametrize("seg, n", [
    ("‡பல்லாண்டு பல்லாண்டு பல்லாயிரத்தாண்டு ⋆", 12),
    ("பலகோடி நூறாயிரம் ⋆", 8),
    ("மல்லாண்ட திண்தோள் மணிவண்ணா ! ⋆", 9),
    ("உன் சேவடி செவ்வி திருக்காப்பு ॥ 1.1.1 ॥", 10),
    ("‡அடியோமோடும் நின்னோடும் ⋆", 8),
    ("பிரிவின்றி ஆயிரம் பல்லாண்டு ⋆", 10),
    ("இராக்கதர் வாழ் ⋆", 5),
    ("கலத்ததுண்டு ⋆", 5),
    ("பரமேட்டியை ⋆", 5),
    ("சூடுமித்தொண்டர்களோம் ⋆", 7),
    ("திருவோணத் திருவிழவில் அந்தியம் போதில் அரி உருவாகி ⋆", 20),
    ("எந்தை தந்தை தந்தை தந்தை தம் மூத்தப்பன் ⋆", 12),
])
def test_hand_counted_segments(seg, n):
    assert seg in SEGMENTS
    assert TS.n_syllables(seg) == n


def test_mey_and_aytam_are_not_counted():
    assert TS.n_syllables("அஃது") == 2
    assert TS.n_syllables("க் ங் ச்") == 0
    assert [k for _, k in TS.eluttu("அஃது")] == ["kuril", "ottru", "kuril"]
    assert TS.eluttu("கௌவை") == [("கௌ", "nedil"), ("வை", "nedil")]


@pytest.mark.parametrize("seg", SEGMENTS)
def test_every_segment_counts_the_same_on_display_and_model_text(seg):
    n = TS.n_syllables(seg)
    assert n > 0 and n == TS.n_syllables(TT.model_text(seg))


@pytest.mark.parametrize("word, asai", [
    ("பல்லாண்டு", [("பல்", "ner"), ("லாண்", "ner"), ("டு", "ner")]),
    ("பலகோடி", [("பல", "nirai"), ("கோ", "ner"), ("டி", "ner")]),
    ("மணிவண்ணா", [("மணி", "nirai"), ("வண்", "ner"), ("ணா", "ner")]),
    ("அடியோமோடும்", [("அடி", "nirai"), ("யோ", "ner"), ("மோ", "ner"), ("டும்", "ner")]),
    ("திருக்காப்பு", [("திருக்", "nirai"), ("காப்", "ner"), ("பு", "ner")]),
    ("உன்", [("உன்", "ner")]),
    ("வடிவாய்", [("வடி", "nirai"), ("வாய்", "ner")]),
    ("சேவடி", [("சே", "ner"), ("வடி", "nirai")]),
])
def test_asai_hand_annotated(word, asai):
    assert TS.asai(word) == asai
