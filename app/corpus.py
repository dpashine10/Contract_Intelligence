"""Load the contracts referenced by the benchmark, preserving original text.

Benchmark snippets refer to contracts as ``cuad/<file name>``. The contract
text is read exactly as the benchmark authors did (Python text mode, UTF-8),
so annotation offsets stay valid: ``text[start:end]`` is the gold answer.

Filename resolution is explicit because one benchmark path is stored in
Unicode NFD form (``E`` + combining accent) while the file on disk uses NFC
(precomposed ``É``). macOS hides this difference; Linux does not.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CORPUS_ROOT = Path("data/raw/corpus/cuad")
BENCHMARK_PREFIX = "cuad/"


@dataclass(frozen=True)
class Resolution:
    """How one benchmark file_path was matched to a file on disk."""

    file_path: str  # as written in the benchmark
    status: str  # "exact" | "nfc" | "missing" | "ambiguous"
    resolved_path: str | None  # path on disk, relative to the corpus root
    candidates: tuple[str, ...] = ()  # filled for ambiguous matches


@dataclass(frozen=True)
class Document:
    """One contract. ``file_path`` is the benchmark's identifier for it."""

    file_path: str
    text: str
    sha256: str


@dataclass
class Corpus:
    documents: list[Document]
    resolutions: list[Resolution]
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def unresolved(self) -> list[Resolution]:
        return [r for r in self.resolutions if r.status in {"missing", "ambiguous"}]


def strip_prefix(file_path: str) -> str:
    """Turn a benchmark path (``cuad/x.txt``) into a corpus-relative name."""
    if file_path.startswith(BENCHMARK_PREFIX):
        return file_path[len(BENCHMARK_PREFIX):]
    return file_path


def nfc(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def resolve_path(file_path: str, corpus_root: Path = DEFAULT_CORPUS_ROOT) -> Resolution:
    """Resolve a benchmark file_path to a file under ``corpus_root``.

    1. Try the exact relative path.
    2. Otherwise compare NFC-normalised names against every file in the
       corpus folder; accept only a unique match.
    3. Otherwise report "missing" (no match) or "ambiguous" (several).
    """
    # Exact match must be byte-for-byte: on macOS, Path.exists() would also
    # succeed for a differently-normalised name, which would hide the issue.
    on_disk = sorted(p.name for p in corpus_root.iterdir()) if corpus_root.is_dir() else []
    return match_name(file_path, on_disk)


def match_name(file_path: str, on_disk: Iterable[str]) -> Resolution:
    """Pure matching rule used by ``resolve_path`` (exact, then unique NFC)."""
    relative = strip_prefix(file_path)
    on_disk = sorted(on_disk)
    if relative in on_disk:
        return Resolution(file_path, "exact", relative)

    target = nfc(relative)
    matches = tuple(name for name in on_disk if nfc(name) == target)
    if len(matches) == 1:
        return Resolution(file_path, "nfc", matches[0])
    if len(matches) > 1:
        return Resolution(file_path, "ambiguous", None, matches)
    return Resolution(file_path, "missing", None)


def read_text(path: Path) -> str:
    """Read a contract the way the benchmark offsets assume."""
    with open(path, encoding="utf-8") as handle:  # text mode, universal newlines
        return handle.read()


def load_corpus(
    file_paths: Iterable[str],
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
) -> Corpus:
    """Load each distinct benchmark file_path once, in sorted order.

    Missing or ambiguous files are recorded, never silently dropped; callers
    decide whether that is fatal (the evaluation runner treats it as fatal).
    """
    documents: list[Document] = []
    resolutions: list[Resolution] = []
    errors: list[dict[str, Any]] = []

    for file_path in sorted(set(file_paths)):
        resolution = resolve_path(file_path, corpus_root)
        resolutions.append(resolution)
        if resolution.resolved_path is None:
            errors.append(
                {
                    "type": f"{resolution.status}_file",
                    "file_path": file_path,
                    "candidates": list(resolution.candidates),
                }
            )
            continue
        try:
            text = read_text(corpus_root / resolution.resolved_path)
        except (OSError, UnicodeDecodeError) as exc:
            errors.append({"type": "read_error", "file_path": file_path, "error": str(exc)})
            continue
        documents.append(
            Document(
                file_path=file_path,
                text=text,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )

    return Corpus(documents=documents, resolutions=resolutions, errors=errors)


def corpus_manifest(corpus: Corpus) -> dict[str, Any]:
    """Per-file hashes plus one combined hash identifying the indexed corpus."""
    files = [
        {"file_path": d.file_path, "sha256": d.sha256, "num_chars": len(d.text)}
        for d in corpus.documents
    ]
    combined = hashlib.sha256(
        "\n".join(f"{f['file_path']}\t{f['sha256']}" for f in files).encode("utf-8")
    ).hexdigest()
    return {"num_documents": len(files), "combined_sha256": combined, "files": files}
