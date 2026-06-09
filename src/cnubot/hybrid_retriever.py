"""하이브리드 검색: FAISS(밀집, KURE) + BM25(희소, char 3-gram) → RRF 융합.

§10 교훈: 한국어 BM25는 어절 split(66%) 아닌 **char trigram**(86%)이어야 함.
§B 교훈: 절대점수 체계 다른 두 엔진은 RRF(순위 역수)로 융합 (+2%p 검증).
generic 질의에서 고유명사(장학금명·과목코드)를 살려 canonical 청크를 끌어올림.
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from .module3_retriever import AcademicRetriever
from .schemas import Intent, RetrievalResult, RetrievedChunk


def _trigrams(text: str) -> list[str]:
    t = "".join(text.split())  # 공백 제거 후 char 3-gram
    return [t[i:i + 3] for i in range(len(t) - 2)] or ([t] if t else ["∅"])


class HybridRRFRetriever:
    def __init__(self, dense: AcademicRetriever, top_k: int = 3, rrf_k: int = 60,
                 pool: int = 10):
        self.dense = dense          # 증분 핫스왑된 FAISS + meta 포인터
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.pool = pool            # 각 엔진 후보 풀 크기
        self.bm25: BM25Okapi | None = None
        self.refresh_sparse()

    def refresh_sparse(self) -> None:
        """meta 본문을 trigram 토큰화하여 BM25 재구축 (핫스왑 후 동기화용)."""
        self.bm25 = BM25Okapi([_trigrams(c["content"]) for c in self.dense.meta])

    def retrieve(self, query: str) -> RetrievalResult:
        # 1. 밀집 (KURE@cuda:1)
        qv = np.ascontiguousarray(self.dense.embedder.encode([query]), dtype=np.float32)
        _, idxs = self.dense.index.search(qv, self.pool)
        dense_ids = [int(i) for i in idxs[0] if i >= 0]
        # 2. 희소 (BM25 trigram)
        sparse_scores = self.bm25.get_scores(_trigrams(query))
        sparse_ids = [int(i) for i in np.argsort(sparse_scores)[::-1][:self.pool]]
        # 3. RRF 융합
        rrf: dict[int, float] = {}
        for rank, i in enumerate(dense_ids):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, i in enumerate(sparse_ids):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:self.top_k]

        chunks = []
        for idx, score in top:
            if idx >= len(self.dense.meta):
                continue
            m = self.dense.meta[idx]
            chunks.append(RetrievedChunk(
                doc_id=m["doc_id"], content=m["content"], title=m.get("title"),
                source_url=m.get("source_url"), score=float(score)))  # score=RRF (cosine 아님)
        if not chunks:
            return RetrievalResult(intent=Intent.ACADEMIC, is_fallback=True,
                                   fallback_message="관련 학사 정보를 찾지 못했습니다.")
        return RetrievalResult(intent=Intent.ACADEMIC, chunks=chunks)
