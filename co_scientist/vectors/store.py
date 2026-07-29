"""Per-session FAISS index wrapper.

Uses `IndexFlatIP` over L2-normalized vectors → exact cosine similarity.
Tradeoff: O(N) search, fine for N < ~10k hypotheses per session. If a session
ever grows past that, swap to `IndexHNSWFlat` (changes file format; bump
embeddings_meta migration).
"""

from __future__ import annotations

import asyncio
import json
import os

import faiss
import numpy as np

from ..config import Config


class FaissStore:
    def __init__(self, cfg: Config, session_id: str, dim: int) -> None:
        self.cfg = cfg
        self.session_id = session_id
        self.dim = dim
        self.index: faiss.IndexFlatIP | None = None
        self._dir = cfg.session_vector_dir(session_id)
        self._index_path = self._dir / "index.faiss"
        self._meta_path = self._dir / "index.meta.json"
        self._ordered_ids: list[str] = []   # hypothesis_id at each faiss offset
        self._offset_by_id: dict[str, int] = {}  # mirror of _ordered_ids for O(1) offset_of()
        # FAISS itself is not thread-safe. Even though our `to_thread` calls
        # serialize on the asyncio loop, they spawn into the default thread
        # pool — multiple coroutines calling `add` / `search` / `save`
        # concurrently can land on different threads and corrupt the index.
        self._lock = asyncio.Lock()

    # ------------------------- lifecycle -------------------------------- #

    async def load_or_create(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

        def _do() -> tuple[faiss.IndexFlatIP, list[str]]:
            if self._index_path.exists() and self._meta_path.exists():
                idx = faiss.read_index(str(self._index_path))
                meta = json.loads(self._meta_path.read_text())
                return idx, list(meta.get("ordered_ids", []))
            return faiss.IndexFlatIP(self.dim), []

        self.index, self._ordered_ids = await asyncio.to_thread(_do)
        self._offset_by_id = {hid: i for i, hid in enumerate(self._ordered_ids)}

    async def save(self) -> None:
        assert self.index is not None

        # Write to *.tmp then os.replace so a crash mid-write doesn't leave a
        # corrupted index file behind. Write meta first, then index, then
        # rename both: if we crash between renames the meta will reflect the
        # *old* index (which is what load_or_create expects).
        idx_tmp = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        meta_tmp = self._meta_path.with_suffix(self._meta_path.suffix + ".tmp")

        def _do() -> None:
            self._dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(idx_tmp))
            meta_tmp.write_text(
                json.dumps({"dim": self.dim, "ordered_ids": self._ordered_ids})
            )
            os.replace(idx_tmp, self._index_path)
            os.replace(meta_tmp, self._meta_path)


        async with self._lock:
            await asyncio.to_thread(_do)

    # ------------------------- ops -------------------------------------- #

    @property
    def n(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    async def add(self, hypothesis_id: str, vec: np.ndarray) -> int:
        """Append one vector. Returns its FAISS offset."""
        assert self.index is not None
        if vec.ndim == 1:
            vec = vec[None, :]

        def _do() -> int:
            off = self.index.ntotal
            self.index.add(vec.astype("float32"))
            return off

        async with self._lock:
            offset = await asyncio.to_thread(_do)
            self._ordered_ids.append(hypothesis_id)
            self._offset_by_id[hypothesis_id] = offset
        return offset

    async def search(
        self, query: np.ndarray, k: int = 5
    ) -> list[tuple[str, float]]:
        """Return [(hypothesis_id, cosine_sim)] best matches.

        Vectors are L2-normalized so inner product == cosine.
        """
        assert self.index is not None
        if self.n == 0:
            return []
        if query.ndim == 1:
            query = query[None, :]

        def _do(qk: int) -> tuple[np.ndarray, np.ndarray]:
            dists, idxs = self.index.search(query.astype("float32"), qk)
            return dists, idxs

        async with self._lock:
            k = min(k, self.n)
            dists, idxs = await asyncio.to_thread(_do, k)
            ordered = list(self._ordered_ids)
        out: list[tuple[str, float]] = []
        for sim, idx in zip(dists[0], idxs[0], strict=True):
            if idx < 0 or idx >= len(ordered):
                continue
            out.append((ordered[int(idx)], float(sim)))
        return out

    async def cosine_matrix(self) -> np.ndarray:
        """Full N×N cosine similarity matrix. Used for clustering."""
        assert self.index is not None
        if self.n == 0:
            return np.zeros((0, 0), dtype="float32")

        def _do() -> np.ndarray:
            vecs = self.index.reconstruct_n(0, self.n)
            return vecs @ vecs.T

        async with self._lock:
            return await asyncio.to_thread(_do)

    def offset_of(self, hypothesis_id: str) -> int | None:
        return self._offset_by_id.get(hypothesis_id)

    def hypothesis_at(self, offset: int) -> str | None:
        if 0 <= offset < len(self._ordered_ids):
            return self._ordered_ids[offset]
        return None
