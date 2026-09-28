"""Embedding providers.

The slot is pluggable, and the honest default is not a semantic one. Three
implementations, in descending order of quality:

* `OpenAICompatEmbeddings` -- any `/embeddings` endpoint: OpenAI, DeepSeek,
  Qwen, Ollama, LM Studio, vLLM. Semantic, and the only one worth calling
  "vector search".
* `HashingEmbeddings` -- character n-gram hashing, offline, no model. This
  captures *form*, not meaning: it clusters near-duplicates and morphological
  variants, and it will not connect "the tests are slow" to "pytest takes
  forty seconds". Useful for the dedupe path, misleading if presented as
  semantic retrieval.
* `NullEmbeddings` -- no vectors at all. Hybrid search degrades to FTS5.

Why the lexical fallback earns its place: the contradiction judge below is an
LLM call, and it only needs *plausibly related* candidates to look at, not
perfect ranking. Hashing finds near-duplicates well, which is exactly the
case that matters (the same fact restated, or a fact that contradicts a
stored one).
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Iterable

import httpx

from unified_agent.errors import ModelError

_LATIN_WORD = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")


class EmbeddingProvider(ABC):
    name: str = "none"
    #: Does this capture meaning, or only form? Surfaced to the user, because
    #: the difference decides whether a search result is trustworthy.
    semantic: bool = False
    dim: int = 0

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per input, in order. Vectors are L2-normalised."""

    def embed_one(self, text: str) -> list[float]:
        raise NotImplementedError

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0] if vectors else []

    def describe(self) -> str:
        kind = "semantic" if self.semantic else "lexical (form only, not meaning)"
        return f"{self.name} dim={self.dim} ({kind})"


def _normalise(vector: Iterable[float]) -> list[float]:
    values = list(vector)
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return values
    return [v / norm for v in values]


class NullEmbeddings(EmbeddingProvider):
    """No vectors. Search falls back to FTS5 alone."""

    name = "none"
    semantic = False
    dim = 0

    @property
    def available(self) -> bool:
        return False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return []

    def describe(self) -> str:
        return "none (vector search disabled; FTS5 only)"


class HashingEmbeddings(EmbeddingProvider):
    """Character n-gram hashing. Deterministic, offline, lexical.

    Uses `hashlib` rather than the builtin `hash()`: `hash()` on a string is
    salted per process (PYTHONHASHSEED), so a reindex in a new process would
    produce entirely different vectors and every stored vector would silently
    become noise. That failure looks like "search got worse" and is nearly
    impossible to trace back.
    """

    name = "hashing"
    semantic = False

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        counts = [0.0] * self.dim
        for token, weight in self._tokens(text):
            index = self._bucket(token)
            counts[index] += weight
        # Sub-linear term weighting, as in TF-IDF: a repeated n-gram should
        # not dominate the vector.
        counts = [1.0 + math.log(c) if c > 0 else 0.0 for c in counts]
        return _normalise(counts)

    def _tokens(self, text: str) -> Iterable[tuple[str, float]]:
        lowered = text.lower()
        for match in _LATIN_WORD.finditer(lowered):
            yield match.group(0), 1.0
        for match in _CJK.finditer(lowered):
            run = match.group(0)
            # CJK has no spaces, so character bigrams are the words.
            for i in range(len(run) - 1):
                yield run[i : i + 2], 1.0
            if len(run) == 1:
                yield run, 1.0
        for n in (3, 4):
            for i in range(max(0, len(lowered) - n + 1)):
                yield lowered[i : i + n], 0.5

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.dim


class OpenAICompatEmbeddings(EmbeddingProvider):
    """Any `/embeddings` endpoint. The only genuinely semantic option here."""

    name = "openai_compat"
    semantic = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        dim: int = 0,
        timeout_s: float = 60.0,
        batch_size: int = 64,
    ) -> None:
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.dim = dim
        self.timeout_s = timeout_s
        self.batch_size = batch_size

    @property
    def available(self) -> bool:
        return True

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            out.extend(await self._batch(batch))
        return out

    async def _batch(self, batch: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {"model": self.model, "input": batch}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(
                    f"{self.base_url}/embeddings", headers=headers, json=payload
                )
        except httpx.HTTPError as exc:
            raise ModelError(f"embeddings transport error: {exc}", retryable=True) from exc
        if response.status_code >= 400:
            raise ModelError(
                f"embeddings HTTP {response.status_code}: {response.text[:300]}",
                retryable=response.status_code in {408, 429, 500, 502, 503, 504},
            )
        data = response.json()
        rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
        vectors = [_normalise(row.get("embedding") or []) for row in rows]
        if len(vectors) != len(batch):
            raise ModelError(
                f"embeddings returned {len(vectors)} vectors for {len(batch)} inputs"
            )
        if not self.dim and vectors:
            self.dim = len(vectors[0])
        return vectors

    def describe(self) -> str:
        return f"{self.name}:{self.model} dim={self.dim} (semantic)"


def build_embeddings(
    *,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    fallback_dim: int = 512,
) -> EmbeddingProvider:
    """Pick a provider. Never raises: a missing key degrades, it does not fail.

    Memory search must keep working on a machine with no embedding endpoint
    -- it is a feature of the runtime, not a bonus that requires a second
    service to be configured.
    """
    if not model:
        return HashingEmbeddings(dim=fallback_dim)
    if not api_key and "localhost" not in (base_url or "") and "127.0.0.1" not in (
        base_url or ""
    ):
        # Local endpoints do not need a key; hosted ones do.
        return HashingEmbeddings(dim=fallback_dim)
    return OpenAICompatEmbeddings(model=model, base_url=base_url, api_key=api_key)


def vector_key_for(provider: EmbeddingProvider) -> str:
    """The identity vectors are stored under.

    One definition, used by everything that writes or reads vectors. Two
    copies of this is how a write lands under `hashing:64` while the read
    looks under `hashing` -- and the symptom is "vector search returns
    nothing", with no error anywhere.

    The key must change when the model or the dimension changes: vectors from
    different models are not comparable, and mixing them produces a ranking
    that is quietly meaningless rather than obviously broken.
    """
    model = getattr(provider, "model", "") if provider.semantic else ""
    return f"{provider.name}:{model}:{provider.dim}"


def pack_vector(vector: list[float]) -> bytes:
    """float32 little-endian. sqlite-vec reads this format directly."""
    import struct

    return struct.pack(f"<{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    import struct

    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python fallback. Inputs are expected to be normalised."""
    if not a or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


__all__ = [
    "EmbeddingProvider",
    "HashingEmbeddings",
    "NullEmbeddings",
    "OpenAICompatEmbeddings",
    "build_embeddings",
    "cosine_similarity",
    "vector_key_for",
    "pack_vector",
    "unpack_vector",
]
