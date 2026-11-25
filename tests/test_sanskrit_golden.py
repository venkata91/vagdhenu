"""BC-1..BC-5: Sanskrit outputs stay identical to the goldens captured at the base commit."""
import json
import pytest
import sanskrit_snapshot as S


@pytest.fixture(scope="module")
def snap_and_golden(tmp_path_factory):
    golden = json.load(open(S.GOLDEN, encoding="utf-8"))
    if golden["_env"] != S.env():
        pytest.fail(f"golden captured with {golden['_env']}, running with {S.env()}; recapture on {golden['_base']}")
    snap = json.loads(json.dumps(S.compute(tmp_path_factory.mktemp("golden"))))
    return snap, golden


def test_snapshot_sections_match_golden(snap_and_golden):
    snap, golden = snap_and_golden
    keys = sorted(k for k in golden if not k.startswith("_") and k != S.ARGPARSE_KEY)
    assert keys == sorted(k for k in snap if k != S.ARGPARSE_KEY)
    for k in keys:
        if isinstance(golden[k], list):
            for i, (g, s) in enumerate(zip(golden[k], snap[k])):
                assert s == g, f"{k}[{i}] differs"
            assert len(snap[k]) == len(golden[k]), k
        else:
            assert snap[k] == golden[k], f"{k} differs"


def test_preexisting_render_flags_keep_defaults(snap_and_golden):
    snap, golden = snap_and_golden
    for dest, default in golden[S.ARGPARSE_KEY].items():
        assert dest in snap[S.ARGPARSE_KEY], dest
        assert snap[S.ARGPARSE_KEY][dest] == default, dest
