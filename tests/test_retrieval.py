import pytest

from app.corpus import Document
from app.retrieval import BM25Index, chunk_corpus, make_chunks, tokenize


def doc(file_path: str, text: str) -> Document:
    return Document(file_path=file_path, text=text, sha256="x")


def test_tokenize_is_deterministic_and_keeps_original_offsets():
    text = "The LICENSEE shall  not Sub-license; Straße 4.2"
    tokens = tokenize(text)
    assert tokenize(text) == tokens
    assert [t.text for t in tokens] == ["the", "licensee", "shall", "not", "sub", "license", "strasse", "4", "2"]
    # offsets slice the ORIGINAL text, even where casefold changed length (ß -> ss)
    assert [text[t.start : t.end] for t in tokens] == ["The", "LICENSEE", "shall", "not", "Sub", "license", "Straße", "4", "2"]


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize(" ,.;\n") == []


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_chunks_cover_text_with_overlap_and_map_to_source():
    text = words(10)
    chunks = make_chunks(doc("cuad/a.txt", text), chunk_size=4, chunk_overlap=1)
    # stride 3: token windows 0-3, 3-6, 6-9 (last one reaches the end)
    assert [c.tokens for c in chunks] == [
        ("w0", "w1", "w2", "w3"),
        ("w3", "w4", "w5", "w6"),
        ("w6", "w7", "w8", "w9"),
    ]
    for c in chunks:
        # chunk span is a contiguous, half-open slice of the original text
        assert text[c.start : c.end].split() == list(c.tokens)
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    assert [c.chunk_id for c in chunks] == [0, 1, 2]


def test_last_chunk_may_be_short_and_no_empty_chunks():
    chunks = make_chunks(doc("a", words(5)), chunk_size=4, chunk_overlap=1)
    assert [len(c.tokens) for c in chunks] == [4, 2]
    assert make_chunks(doc("a", ""), chunk_size=4, chunk_overlap=1) == []
    assert len(make_chunks(doc("a", words(3)), chunk_size=4, chunk_overlap=1)) == 1


def test_chunk_parameter_validation():
    with pytest.raises(ValueError):
        make_chunks(doc("a", "x"), chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        make_chunks(doc("a", "x"), chunk_size=4, chunk_overlap=4)


def test_chunk_corpus_ids_are_global_and_ordered():
    chunks = chunk_corpus([doc("cuad/b", words(5)), doc("cuad/a", words(5))], chunk_size=4, chunk_overlap=0)
    assert [c.file_path for c in chunks] == ["cuad/a", "cuad/a", "cuad/b", "cuad/b"]
    assert [c.chunk_id for c in chunks] == [0, 1, 2, 3]


def build(docs, size=8, overlap=2):
    return BM25Index(chunk_corpus(docs, size, overlap), k1=1.5, b=0.75, epsilon=0.25)


def test_bm25_ranks_matching_chunk_first():
    index = build(
        [
            doc("cuad/a", "payment terms net thirty days invoice"),
            doc("cuad/b", "the licensee may not sublicense the content"),
            doc("cuad/c", "governing law is the state of new york"),
        ]
    )
    results = index.search("Can the licensee sublicense?", top_k=3)
    assert results[0].file_path == "cuad/b"
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score > results[1].score
    assert results[0].start == 0


def test_ties_broken_by_file_path_then_offset():
    # Identical documents -> identical scores; order must be filename, then start.
    index = build([doc("cuad/z", "alpha beta"), doc("cuad/a", "alpha beta"), doc("cuad/m", "alpha beta")])
    results = index.search("alpha", top_k=3)
    assert len({r.score for r in results}) == 1
    assert [r.file_path for r in results] == ["cuad/a", "cuad/m", "cuad/z"]
    assert index.search("alpha", top_k=3) == results  # repeatable


def test_empty_query_and_empty_corpus():
    index = build([doc("cuad/a", "alpha beta")])
    assert index.search("", top_k=5) == []
    assert index.search("  ?!  ", top_k=5) == []
    empty = BM25Index([], k1=1.5, b=0.75, epsilon=0.25)
    assert len(empty) == 0 and empty.search("alpha", top_k=5) == []
    with pytest.raises(ValueError):
        index.search("alpha", top_k=0)


def test_top_k_larger_than_index():
    index = build([doc("cuad/a", "alpha beta")])
    assert len(index.search("alpha", top_k=64)) == 1
