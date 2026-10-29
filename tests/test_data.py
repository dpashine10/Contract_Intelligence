
import hashlib

from app.data import add_splits, assign_split, mini, split_counts


def make_test(file_path: str, query: str) -> dict:
    return {
        "query": query,
        "snippets": [{"file_path": file_path}],
    }


def test_mini_returns_all_tests_when_below_limit():
    tests = [make_test("a.json", "question A")]

    assert mini(tests, n=194) == tests


def test_mini_selects_requested_number():
    tests = [
        make_test(f"contract_{i}.json", f"question {i}")
        for i in range(250)
    ]

    assert len(mini(tests, n=194)) == 194


def test_mini_is_deterministic():
    tests = [
        make_test(f"contract_{i}.json", f"question {i}")
        for i in range(250)
    ]

    first = mini(tests, n=194)
    second = mini(tests, n=194)

    assert [
        t["snippets"][0]["file_path"] for t in first
    ] == [
        t["snippets"][0]["file_path"] for t in second
    ]


def test_assign_split_matches_sha256_rule():
    query = "  Does the license allow sublicensing?  "
    normalized = query.strip()

    h = int(
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        16,
    )
    expected = "test" if h % 10 < 3 else "train"

    assert assign_split(query) == expected


def test_add_splits_preserves_queries_and_labels():
    tests = [
        make_test("a.json", "Question A"),
        make_test("b.json", "Question B"),
    ]

    result = add_splits(tests)

    assert len(result) == len(tests)
    assert all(t["split"] in {"train", "test"} for t in result)
    assert all("split" not in t for t in tests)


def test_split_counts():
    tests = [
        {"split": "train"},
        {"split": "train"},
        {"split": "test"},
    ]

    assert split_counts(tests) == {"train": 2, "test": 1}
