"""Tests for validate_data.py using synthetic examples."""

import tempfile
import json
from pathlib import Path

from app.validate_data import (
    validate_snippet,
    resolve_corpus_path,
    mini,
)


def test_resolve_corpus_path_strips_cuad_prefix():
    corpus_root = Path("/fake/corpus")
    assert resolve_corpus_path("cuad/file.txt", corpus_root) == corpus_root / "file.txt"
    assert resolve_corpus_path("file.txt", corpus_root) == corpus_root / "file.txt"


def test_validate_snippet_exact_match():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp)
        doc = corpus / "contract.txt"
        doc.write_text("Hello world, this is a test document.", encoding="utf-8")

        # Span [6, 11] -> "world"
        result = validate_snippet("contract.txt", [6, 11], "world", corpus)
        assert result["status"] == "match"
        assert result["details"]["length"] == 5


def test_validate_snippet_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp)
        doc = corpus / "contract.txt"
        doc.write_text("Hello world, this is a test document.", encoding="utf-8")

        # Span [6, 11] -> "world" but answer says "World" (case mismatch)
        result = validate_snippet("contract.txt", [6, 11], "World", corpus)
        assert result["status"] == "mismatch"
        assert result["details"]["expected_length"] == 5
        assert result["details"]["actual_length"] == 5
        assert "expected_preview" in result["details"]
        assert "actual_preview" in result["details"]


def test_validate_snippet_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp)
        # File doesn't exist
        result = validate_snippet("nonexistent.txt", [0, 5], "Hello", corpus)
        assert result["status"] == "missing_file"
        assert "resolved_path" in result["details"]


def test_validate_snippet_malformed_span_not_list():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp)
        doc = corpus / "contract.txt"
        doc.write_text("Hello world", encoding="utf-8")

        # span is not a list
        result = validate_snippet("contract.txt", "not a list", "Hello", corpus)
        assert result["status"] == "malformed"
        assert "span must be a list" in result["details"]["error"]


def test_validate_snippet_malformed_span_wrong_length():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp)
        doc = corpus / "contract.txt"
        doc.write_text("Hello world", encoding="utf-8")

        # span has wrong length
        result = validate_snippet("contract.txt", [0], "Hello", corpus)
        assert result["status"] == "malformed"
        assert "span must be a list of two integers" in result["details"]["error"]


def test_validate_snippet_malformed_span_non_int():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp)
        doc = corpus / "contract.txt"
        doc.write_text("Hello world", encoding="utf-8")

        # span contains non-integers
        result = validate_snippet("contract.txt", [0, "five"], "Hello", corpus)
        assert result["status"] == "malformed"
        assert "span bounds must be integers" in result["details"]["error"]


def test_validate_snippet_malformed_span_negative():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp)
        doc = corpus / "contract.txt"
        doc.write_text("Hello world", encoding="utf-8")

        result = validate_snippet("contract.txt", [-1, 5], "Hello", corpus)
        assert result["status"] == "malformed"
        assert "invalid span bounds" in result["details"]["error"]


def test_validate_snippet_malformed_span_start_gt_end():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp)
        doc = corpus / "contract.txt"
        doc.write_text("Hello world", encoding="utf-8")

        result = validate_snippet("contract.txt", [10, 5], "Hello", corpus)
        assert result["status"] == "malformed"
        assert "invalid span bounds" in result["details"]["error"]


def test_validate_snippet_malformed_span_exceeds_file():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp)
        doc = corpus / "contract.txt"
        doc.write_text("Hello", encoding="utf-8")  # length 5

        result = validate_snippet("contract.txt", [0, 10], "Hello", corpus)
        assert result["status"] == "malformed"
        assert "exceeds file length" in result["details"]["error"]


def test_mini_deterministic():
    """Test that mini selection is deterministic."""
    tests = [
        {"query": f"q{i}", "snippets": [{"file_path": f"file_{i}.txt"}]}
        for i in range(250)
    ]
    first = mini(tests, n=194)
    second = mini(tests, n=194)
    assert len(first) == 194
    assert len(second) == 194
    assert [t["snippets"][0]["file_path"] for t in first] == [
        t["snippets"][0]["file_path"] for t in second
    ]


def test_mini_returns_all_when_below_limit():
    tests = [{"query": "q1", "snippets": [{"file_path": "a.txt"}]}]
    result = mini(tests, n=194)
    assert result == tests


def test_mini_selects_exact_count():
    tests = [
        {"query": f"q{i}", "snippets": [{"file_path": f"file_{i}.txt"}]}
        for i in range(250)
    ]
    result = mini(tests, n=194)
    assert len(result) == 194


if __name__ == "__main__":
    # Run tests manually
    test_resolve_corpus_path_strips_cuad_prefix()
    print("test_resolve_corpus_path_strips_cuad_prefix: PASS")

    test_validate_snippet_exact_match()
    print("test_validate_snippet_exact_match: PASS")

    test_validate_snippet_mismatch()
    print("test_validate_snippet_mismatch: PASS")

    test_validate_snippet_missing_file()
    print("test_validate_snippet_missing_file: PASS")

    test_validate_snippet_malformed_span_not_list()
    print("test_validate_snippet_malformed_span_not_list: PASS")

    test_validate_snippet_malformed_span_wrong_length()
    print("test_validate_snippet_malformed_span_wrong_length: PASS")

    test_validate_snippet_malformed_span_non_int()
    print("test_validate_snippet_malformed_span_non_int: PASS")

    test_validate_snippet_malformed_span_negative()
    print("test_validate_snippet_malformed_span_negative: PASS")

    test_validate_snippet_malformed_span_start_gt_end()
    print("test_validate_snippet_malformed_span_start_gt_end: PASS")

    test_validate_snippet_malformed_span_exceeds_file()
    print("test_validate_snippet_malformed_span_exceeds_file: PASS")

    test_mini_deterministic()
    print("test_mini_deterministic: PASS")

    test_mini_returns_all_when_below_limit()
    print("test_mini_returns_all_when_below_limit: PASS")

    test_mini_selects_exact_count()
    print("test_mini_selects_exact_count: PASS")

    print("\nAll tests passed!")