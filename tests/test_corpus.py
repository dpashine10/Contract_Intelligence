import unicodedata
from pathlib import Path

import pytest

from app.corpus import corpus_manifest, load_corpus, match_name, resolve_path

REAL_CORPUS = Path("data/raw/corpus/cuad")
LECLANCHE = "LECLANCHÉ S.A. - JOINT DEVELOPMENT AND MARKETING AGREEMENT.txt"

NFC_NAME = unicodedata.normalize("NFC", "CAFÉ AGREEMENT.txt")
NFD_NAME = unicodedata.normalize("NFD", "CAFÉ AGREEMENT.txt")


def test_nfc_and_nfd_differ():
    assert NFC_NAME != NFD_NAME
    assert unicodedata.normalize("NFC", NFD_NAME) == NFC_NAME


def test_exact_match_wins():
    r = match_name("cuad/a.txt", ["a.txt", "b.txt"])
    assert (r.status, r.resolved_path) == ("exact", "a.txt")


def test_path_without_prefix_is_accepted():
    assert match_name("a.txt", ["a.txt"]).status == "exact"


def test_nfd_benchmark_name_matches_unique_nfc_file():
    r = match_name("cuad/" + NFD_NAME, [NFC_NAME, "other.txt"])
    assert (r.status, r.resolved_path) == ("nfc", NFC_NAME)


def test_two_normalised_matches_are_ambiguous():
    # Canonically equivalent names (combining marks in different orders):
    # both normalise to the same NFC string, neither equals the query exactly.
    e = "o\u0323\u0302.txt"  # o + dot below + circumflex
    f = "o\u0302\u0323.txt"  # o + circumflex + dot below
    query = unicodedata.normalize("NFC", e)
    assert len({e, f, query}) == 3
    assert unicodedata.normalize("NFC", f) == query
    r = match_name("cuad/" + query, [e, f])
    assert r.status == "ambiguous"
    assert set(r.candidates) == {e, f}
    assert r.resolved_path is None


def test_missing_name():
    r = match_name("cuad/x.txt", ["y.txt"])
    assert (r.status, r.resolved_path) == ("missing", None)


def test_load_corpus_reports_missing_file(tmp_path):
    corpus = load_corpus(["cuad/nope.txt"], tmp_path)
    assert corpus.documents == []
    assert corpus.resolutions[0].status == "missing"
    assert corpus.errors == [{"type": "missing_file", "file_path": "cuad/nope.txt", "candidates": []}]


def test_resolve_path_on_real_directory(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert resolve_path("cuad/a.txt", tmp_path).status == "exact"
    assert resolve_path("cuad/a.txt", tmp_path / "absent").status == "missing"


def test_text_is_read_without_modification(tmp_path):
    (tmp_path / "a.txt").write_bytes("line1\nline2 — café\n".encode("utf-8"))
    corpus = load_corpus(["cuad/a.txt"], tmp_path)
    assert corpus.documents[0].text == "line1\nline2 — café\n"
    manifest = corpus_manifest(corpus)
    assert manifest["num_documents"] == 1 and len(manifest["combined_sha256"]) == 64


def test_load_corpus_deduplicates_and_sorts(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    corpus = load_corpus(["cuad/b.txt", "cuad/a.txt", "cuad/b.txt"], tmp_path)
    assert [d.file_path for d in corpus.documents] == ["cuad/a.txt", "cuad/b.txt"]


@pytest.mark.skipif(not (REAL_CORPUS / LECLANCHE).exists(), reason="corpus not downloaded")
def test_real_leclanche_nfd_benchmark_path_resolves():
    benchmark_path = "cuad/" + unicodedata.normalize("NFD", LECLANCHE)
    r = resolve_path(benchmark_path, REAL_CORPUS)
    assert r.status in {"exact", "nfc"}
    assert unicodedata.normalize("NFC", r.resolved_path) == LECLANCHE
    corpus = load_corpus([benchmark_path], REAL_CORPUS)
    assert corpus.errors == [] and len(corpus.documents) == 1
