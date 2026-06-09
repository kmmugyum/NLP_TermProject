"""모듈 1: 로컬 학사 JSON 적재 → KURE-v1 임베딩 → FAISS 인덱스 빌더.

진입점 가정: Colab 이 가공한 학사 JSON 이 data/ 에 이미 적재돼 있음 (외부 통신 deferred).
디바이스 규칙: 임베딩은 반드시 GPU1(cuda:1). cuda:0 은 vLLM 전용이라 침범 시 OOM →
            cuda:0 자동 폴백 금지, 안 보이면 명시적으로 에러.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from .schemas import BuildReport, CNUExtractedChunk, IndexBuildConfig


class Embedder(Protocol):
    """주입 가능한 임베더 계약. 단위 테스트는 FakeEmbedder 로 모델 없이 로직 검증."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """[N] 텍스트 → [N, dim] float32, L2-normalized (IP=cosine)."""
        ...


def _assert_device_visible(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"{device} 요청했으나 CUDA 사용 불가")
    idx = int(device.split(":", 1)[1]) if ":" in device else 0
    n = torch.cuda.device_count()
    if idx >= n:
        raise RuntimeError(
            f"{device} 요청했으나 보이는 GPU 는 {n}개뿐. "
            f"CUDA_VISIBLE_DEVICES=0,1 로 실행해 GPU1 을 노출하세요 "
            f"(cuda:0 폴백은 vLLM 영역 침범 → 금지)."
        )


class KUREEmbedder:
    """KURE-v1 (1024d) SentenceTransformer 래퍼. GPU1 고정."""

    def __init__(self, model_name: str, device: str):
        _assert_device_visible(device)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)


def merge_chunk_dicts(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """doc_id 기준 증분 병합. 같은 doc_id 는 incoming 이 override, 신규는 append.
    순서 보존. webhook 증분 갱신 시 기존 코퍼스 유실/중복 방지."""
    by_id = {c["doc_id"]: c for c in existing}
    order = [c["doc_id"] for c in existing]
    for c in incoming:
        if c["doc_id"] not in by_id:
            order.append(c["doc_id"])
        by_id[c["doc_id"]] = c
    return [by_id[i] for i in order]


def _embed_text(chunk: CNUExtractedChunk) -> str:
    # 제목을 본문 앞에 붙여 임베딩 (학과/제도명 신호 보존)
    return f"{chunk.title}\n{chunk.content}" if chunk.title else chunk.content


def _atomic_write_index(index: faiss.Index, path: Path) -> None:
    # tmp 에 완전히 쓴 뒤 os.replace 로 원자 교체 — 크래시 시 기존 인덱스 보존
    tmp = path.with_suffix(path.suffix + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build_vector_db(
    config: IndexBuildConfig, embedder: Embedder | None = None
) -> BuildReport:
    """학사 JSON 을 읽어 FAISS 인덱스 + 메타 사이드카를 디스크에 저장.

    embedder 미지정 시 KURE-v1 을 config.device 에 로드. ValidationError 는
    잡지 않고 전파한다 (계약 위반은 진입점에서 즉시 실패해야 함).
    """
    data_path = Path(config.data_path)
    if not data_path.exists():
        return BuildReport(
            ok=False, num_chunks=0, index_path=config.save_path,
            error=f"입력 파일 없음: {data_path}",
        )

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    parsed = [CNUExtractedChunk(**item) for item in raw]  # ValidationError 전파

    # 초단문(공백 제외) 청크 제외 — 검색 노이즈 방지
    chunks = [c for c in parsed if len(c.content.strip()) >= config.min_content_chars]
    skipped = len(parsed) - len(chunks)
    if not chunks:
        return BuildReport(
            ok=False, num_chunks=0, num_skipped=skipped, index_path=config.save_path,
            error="유효 청크 0개 (전부 비어있거나 너무 짧음)",
        )

    if embedder is None:
        embedder = KUREEmbedder(config.model_name, config.device)
    emb = np.ascontiguousarray(
        embedder.encode([_embed_text(c) for c in chunks]), dtype=np.float32
    )  # [N, dim]
    if emb.ndim != 2 or emb.shape[0] != len(chunks):
        return BuildReport(
            ok=False, num_chunks=len(chunks), num_skipped=skipped,
            index_path=config.save_path, error=f"임베딩 shape 이상: {emb.shape}",
        )
    # NaN/Inf 가드 — 깨진 벡터를 인덱스에 쓰지 않음 (임베더 교체 대비 안전망)
    if not np.isfinite(emb).all():
        return BuildReport(
            ok=False, num_chunks=len(chunks), num_skipped=skipped,
            index_path=config.save_path, error="임베딩에 NaN/Inf 포함 — 인덱스 미저장",
        )
    dim = int(emb.shape[1])

    index = faiss.IndexFlatIP(dim)  # normalized + IP = cosine
    index.add(emb)

    save_path = Path(config.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = save_path.with_suffix(save_path.suffix + ".meta.json")
    # 원자적 쓰기: 둘 다 완성 후 교체 (크래시 시 기존 산출물 보존)
    _atomic_write_index(index, save_path)
    _atomic_write_text(
        meta_path, json.dumps([c.model_dump() for c in chunks], ensure_ascii=False)
    )

    return BuildReport(
        ok=True, num_chunks=len(chunks), num_skipped=skipped,
        index_path=str(save_path), meta_path=str(meta_path), embed_dim=dim,
    )


if __name__ == "__main__":
    import sys

    report = build_vector_db(IndexBuildConfig())
    print(report.model_dump_json(indent=2))
    sys.exit(0 if report.ok else 1)
