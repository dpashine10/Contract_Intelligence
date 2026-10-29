"""Baseline lexical retrieval: regex tokens -> fixed token windows -> BM25.

Every token and chunk carries character offsets into the ORIGINAL contract
text. Normalisation (casefold) only changes the token string BM25 sees; it
never changes offsets, so retrieved spans can be scored against gold spans.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

from app.corpus import Document

# A token is a run of Unicode letters/digits/underscore. Punctuation and
# whitespace are separators. No stemming, stop-words or synonyms (baseline).
TOKEN_PATTERN = re.compile(r"\w+")


@dataclass(frozen=True)
class Token:
    text: str  # normalised form used for matching
    start: int  # offset of the first character in the original text
    end: int  # offset one past the last character (half-open)


@dataclass(frozen=True)
class Chunk:
    chunk_id: int  # position in the index (stable for a fixed corpus+config)
    file_path: str
    start: int  # half-open [start, end) in the original contract text
    end: int
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    rank: int  # 1-based
    score: float  # corpus-level BM25 score
    chunk_id: int
    file_path: str
    start: int
    end: int
    rerank_score: float | None = None  # document-local BM25 score, if re-ranked


# Small, fixed English function-word list (used only when
# retrieval.query_stopwords is enabled; removed from QUERIES only).
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into is
    it its itself just me more most my no nor not of off on once only or other our ours
    out over own same she should so some such than that the their theirs them then there
    these they this those through to too under until up very was we were what when where
    which while who whom why will with would you your yours consider""".split()
)


def normalise(token: str) -> str:
    return token.casefold()


def tokenize(text: str) -> list[Token]:
    """Deterministic word tokens with original-text offsets."""
    return [Token(normalise(m.group()), m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def query_tokens(query: str, drop_stopwords: bool = False) -> list[str]:
    terms = [t.text for t in tokenize(query)]
    if drop_stopwords:
        terms = [t for t in terms if t not in STOPWORDS]
    return terms


def make_chunks(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
    first_chunk_id: int = 0,
) -> list[Chunk]:
    """Split a document into windows of ``chunk_size`` tokens.

    Consecutive windows share ``chunk_overlap`` tokens (stride = size - overlap).
    A chunk's span runs from its first token's start to its last token's end,
    so it is a contiguous slice of the original text. The final window is
    allowed to be shorter; no window is emitted past the last token.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    tokens = tokenize(document.text)
    stride = chunk_size - chunk_overlap
    chunks: list[Chunk] = []
    position = 0
    while position < len(tokens):
        window = tokens[position : position + chunk_size]
        chunks.append(
            Chunk(
                chunk_id=first_chunk_id + len(chunks),
                file_path=document.file_path,
                start=window[0].start,
                end=window[-1].end,
                tokens=tuple(t.text for t in window),
            )
        )
        if position + chunk_size >= len(tokens):
            break
        position += stride
    return chunks


def chunk_corpus(
    documents: Sequence[Document], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk documents in NFC-normalised file_path order for a stable index."""
    chunks: list[Chunk] = []
    for document in sorted(documents, key=lambda d: sort_key(d.file_path)):
        chunks.extend(make_chunks(document, chunk_size, chunk_overlap, first_chunk_id=len(chunks)))
    return chunks


def sort_key(file_path: str) -> str:
    return unicodedata.normalize("NFC", file_path)


@dataclass(frozen=True)
class ScoreFeatures:
    """Query-level evidence used by the abstention rule."""

    top1: float | None  # best corpus-level BM25 score
    max_possible: float  # sum over query terms of idf * (k1 + 1): BM25 upper bound
    mean: float | None  # mean score over all chunks
    std: float | None  # population std of scores over all chunks
    num_query_terms: int
    num_query_terms_in_vocab: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class BM25Index:
    """Okapi BM25 over chunks (rank_bm25 implementation)."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        k1: float,
        b: float,
        epsilon: float,
        query_stopwords: bool = False,
        rerank: str = "none",
        rerank_docs: int = 1,
    ):
        if rerank not in {"none", "doc_local_bm25"}:
            raise ValueError(f"unknown rerank {rerank!r}")
        self.chunks = list(chunks)
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.query_stopwords = query_stopwords
        self.rerank, self.rerank_docs = rerank, rerank_docs
        # rank_bm25 divides by the corpus size, so an empty corpus is handled
        # here instead of inside the library.
        self._bm25 = (
            BM25Okapi([list(c.tokens) for c in self.chunks], k1=k1, b=b, epsilon=epsilon)
            if self.chunks
            else None
        )
        self._by_doc: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self._by_doc.setdefault(c.file_path, []).append(i)
        self._local: dict[str, BM25Okapi] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def _order(self, indices, scores) -> list[int]:
        return sorted(
            indices,
            key=lambda i: (-float(scores[i]), sort_key(self.chunks[i].file_path), self.chunks[i].start),
        )

    def _local_scores(self, file_path: str, terms: list[str]) -> dict[int, float]:
        """BM25 over the chunks of ONE document (document-local IDF)."""
        idx = self._by_doc[file_path]
        if file_path not in self._local:
            self._local[file_path] = BM25Okapi(
                [list(self.chunks[i].tokens) for i in idx], k1=self.k1, b=self.b, epsilon=self.epsilon
            )
        local = self._local[file_path].get_scores(terms)
        return {i: float(local[j]) for j, i in enumerate(idx)}

    def search_detailed(self, query: str, top_k: int) -> tuple[list[SearchResult], ScoreFeatures]:
        """Ranked results plus score features for abstention.

        Ranking: corpus-level BM25, ties broken by NFC file_path then start.
        With rerank=doc_local_bm25, documents are ordered by their best chunk;
        chunks of the top ``rerank_docs`` documents are re-scored with a BM25
        built only over that document and placed first (document by document);
        remaining slots follow the corpus-level order.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        terms = query_tokens(query, self.query_stopwords)
        if not terms or self._bm25 is None:
            return [], ScoreFeatures(None, 0.0, None, None, len(terms), 0)

        scores = self._bm25.get_scores(terms)
        idf = self._bm25.idf
        in_vocab = [t for t in terms if t in idf]
        max_possible = sum(idf[t] * (self.k1 + 1) for t in in_vocab)
        n = len(scores)
        mean = float(sum(scores) / n)
        std = math.sqrt(max(0.0, float(sum((x - mean) ** 2 for x in scores)) / n))
        order = self._order(range(len(self.chunks)), scores)
        features = ScoreFeatures(float(scores[order[0]]), float(max_possible), mean, std, len(terms), len(in_vocab))

        local_scores: dict[int, float] = {}
        if self.rerank == "doc_local_bm25":
            docs: list[str] = []
            for i in order:
                fp = self.chunks[i].file_path
                if fp not in docs:
                    docs.append(fp)
                if len(docs) == self.rerank_docs:
                    break
            head: list[int] = []
            for fp in docs:
                ls = self._local_scores(fp, terms)
                local_scores.update(ls)
                head += self._order(list(ls), ls)
            placed = set(head)
            order = head + [i for i in order if i not in placed]

        results = []
        for rank, i in enumerate(order[:top_k], start=1):
            chunk = self.chunks[i]
            results.append(
                SearchResult(
                    rank=rank,
                    score=float(scores[i]),
                    chunk_id=chunk.chunk_id,
                    file_path=chunk.file_path,
                    start=chunk.start,
                    end=chunk.end,
                    rerank_score=local_scores.get(i),
                )
            )
        return results, features

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to ``top_k`` chunks, best first (see ``search_detailed``).

        An empty query (no tokens) or an empty index returns an empty list.
        """
        return self.search_detailed(query, top_k)[0]
