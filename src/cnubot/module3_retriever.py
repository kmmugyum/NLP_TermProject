"""모듈 3: Retriever & Cache Verifier.

학사(정적): FAISS 인덱스 + meta(모듈1 산출) 로드 → KURE-v1@cuda:1 질의 임베딩 →
            top_k 검색 → 저점수 거부(score_threshold).
학식(동적): 인메모리 캐시(웹훅 직접 swap) + 디스크 reload, 24h 신선도/빈메뉴 이중 방어.

임베더는 모듈1과 동일 인스턴스(KURE-v1@cuda:1)를 주입받아 GPU1 메모리 공유.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import re

import faiss
import numpy as np

from .module1_indexer import Embedder, _atomic_write_index, _atomic_write_text
from .schemas import Intent, RetrievalResult, RetrievedChunk, WeeklyMealCache

# 학사 규정 질의 패턴(졸업학점·등록금·휴학 등). retrieve()가 이걸로 학사요람 청크 boost.
_REG_QUERY_RE = re.compile(
    r"졸업\s*요건|졸업\s*학점|소요\s*학점|이수\s*학점|이수\s*기준|"
    r"학사\s*규정|등록금|장학금\s*규정|휴학|복학|전과|자퇴|재입학|재수강|학점\s*포기|"
    r"교환학생|수료\s*요건|복수전공\s*요건|부전공\s*요건|"
    # 정의·구분 질의(학사요람의 정의 단락이 권위)
    r"전공\s*기초|전공\s*핵심|전공\s*심화|"
    r"이수\s*구분|교과목\s*구분|교과목\s*편성|영역\s*구분|"
    r"기초\s*교양|균형\s*교양|소양\s*교양|핵심\s*교양|공통기초\s*교양|"
    r"복수\s*전공|부전공|연계\s*전공|자기\s*설계\s*전공|산학\s*융복합\s*전공|"
    r"학사\s*학위|석사\s*학위|박사\s*학위|학위\s*종류|학위\s*과정|"
    r"졸업인정|수료인정|졸업.{0,8}수료|수료.{0,8}졸업|"
    r"성적\s*등급|성적\s*평점|평점\s*평균|평점\s*환산|등급-?평점|"
    r"학사\s*경고|성적\s*경고|유급\s*기준|유급\s*조건|"
    r"학점\s*등급|GPA|gpa|"
    r"교양\s*역량|6대\s*역량|7대\s*역량|핵심\s*역량|"
    r"자기관리\s*역량|의사소통\s*역량|대인관계\s*역량|"
    r"학점\s*인정|학점\s*환산|편입학.{0,10}학점|검정고시.{0,10}학점|"
    r"등록금\s*반환|등록금\s*환불|반환\s*기준|"
    r"융복합창의\s*전공|마이크로\s*디그리|"
    # 추가: 입학·수강·출결·학위·장학·학칙·징계 등
    r"입학\s*자격|신입학|편입학|입학\s*절차|입학\s*전형|"
    r"수업\s*연한|재학\s*연한|"
    r"수강\s*신청|수강\s*취소|수강\s*변경|수강\s*포기|"
    r"출석\s*인정|결석|출결|지각|"
    r"학위\s*수여|졸업\s*논문|학위증|학위\s*취소|졸업\s*취소|"
    r"장학금|국가\s*장학금|근로\s*장학|장학생|"
    r"학칙\s*개정|학칙\s*뭐|징계|이의\s*신청|학적부|학적\s*관리|"
    r"재수강|재이수|시간제\s*등록|시간제\s*학생|"
    r"학사학위\s*취득\s*유예|학사학위취득유예|졸업\s*유예|졸업유예|졸업연기"
)

# 정의 질의 → 학사요람 정의 청크 직접 매핑 테이블.
# (질의 패턴, 본문 마커 패턴). retrieve()가 임베딩 거리로 못 잡는 정의 단락을 강제 prepend.
# 매핑 entry: (질의 패턴, 본문 마커 튜플). 마커 여러 개면 각각 매칭되는 청크를 모두 prepend.
# 정의가 여러 조항에 흩어진 케이스(휴학 제35조/복학 제36조/제적 제37조 등)에 대응.
_DEFINITION_PREPEND_MAP: tuple = (
    # 전공 이수구분 정의(교과목의 구분/편성 맥락에서의 전공기초·전공핵심·전공심화 정의)
    (re.compile(r"전공\s*기초|전공\s*핵심|전공\s*심화|"
                r"교과목\s*구분|이수\s*구분|교과목\s*편성"),
     (re.compile(r"(?=.*?(?:교과목의\s*구분|교과목\s*편성|전문학술연구))"
                 r"(?=.*?전공기초[,\s]*전공핵심[,\s]*전공심화)", re.DOTALL),)),
    # 교양 영역 정의(기초/균형/소양 + 학점 배점)
    (re.compile(r"교양.{0,3}(기초|균형|소양|핵심|공통)|기초\s*교양|"
                r"균형\s*교양|소양\s*교양|핵심\s*교양|공통기초\s*교양"),
     (re.compile(r"기초교양에서.{0,20}학점.{0,30}균형교양|"
                 r"기초교양.{0,15}균형교양.{0,15}소양교양"),)),
    # 전공과정 종류(복수전공·부전공·연계전공·자기설계전공·산학융복합전공·마이크로디그리)
    (re.compile(r"복수\s*전공|부전공|연계\s*전공|자기\s*설계\s*전공|"
                r"산학\s*융복합\s*전공|융복합창의\s*전공|마이크로\s*디그리"),
     (re.compile(r"제56조\(융복합창의전공"),  # 융복합창의전공 정의 조항
      re.compile(r"복수전공과정.{0,15}부전공과정.{0,15}연계전공과정.{0,15}자기설계전공과정"))),
    # 학사학위취득유예(더 구체적인 entry — 학사학위·학위종류 매칭보다 우선)
    (re.compile(r"학사학위\s*취득\s*유예|학사학위취득유예|졸업\s*유예|졸업유예|졸업연기"),
     (re.compile(r"제70조의2\(학사학위취득유예\)"),)),
    # 학위 종류(학사·석사·전문석사·박사·전문박사·명예박사)
    (re.compile(r"학사\s*학위|석사\s*학위|박사\s*학위|학위\s*종류|학위\s*과정"),
     (re.compile(r"학사,\s*석사,\s*전문석사,\s*박사,\s*전문박사"),)),
    # 학적 변동(휴학·복학·제적): 3개 조항이 별개 청크 → 모두 prepend
    (re.compile(r"휴학|복학|제적"),
     (re.compile(r"제35조\(휴학\)"),
      re.compile(r"제36조\(복학\)"),
      re.compile(r"제37조\(제적\)"))),
    # 졸업 vs 수료(학부 졸업인정 / 대학원 수료인정 / 학점 기준): 3개 조항이 별개 청크
    (re.compile(r"졸업.{0,8}수료|수료.{0,8}졸업|졸업인정|수료인정"),
     (re.compile(r"제61조\(졸업인정\s*및\s*학위수여\)"),
      re.compile(r"제54조\(수료인정\)"),
      re.compile(r"제59조\(졸업\s*및\s*수료\s*소요학점\)"))),
    # 성적 등급·평점·경고·유급: 4개 조항(평가 방법·등급-평점 표·경고 1.75·유급)
    (re.compile(r"성적\s*등급|성적\s*평점|평점\s*평균|평점\s*환산|등급-?평점|"
                r"학사\s*경고|성적\s*경고|유급\s*기준|유급\s*조건|"
                r"학점\s*등급|GPA|gpa"),
     (re.compile(r"제40조\(성적평점평균과\s*등급\)"),
      re.compile(r"제31조\(성적평가\)"),
      re.compile(r"제66조\(성적경고\)"),
      re.compile(r"제67조\(유급\)"))),
    # 교양 6대 역량(자기관리·의사소통·대인관계·창의융합·인성·글로벌·플러스)
    (re.compile(r"교양\s*역량|6대\s*역량|7대\s*역량|핵심\s*역량|"
                r"자기관리\s*역량|의사소통\s*역량|대인관계\s*역량|"
                r"창의[\s·,]*융합\s*역량|인성\s*역량|글로벌\s*역량"),
     (re.compile(r"제3조\(교육\s*역량\).{0,200}자기관리역량", re.DOTALL),)),
    # 학점인정(편입학·교환학생·검정고시 등 외부 학점의 본교 인정)
    (re.compile(r"학점\s*인정|학점\s*환산|편입학.{0,10}학점|"
                r"검정고시.{0,10}학점|타\s*대학\s*학점|교환학생.{0,10}학점"),
     (re.compile(r"제61조\(학점의?\s*인정\)"),
      re.compile(r"제63조\(편입학자의\s*학점\)"),
      re.compile(r"제70조\(학점인정"))),
    # 등록금 반환 기준(제42조: 반환 사유·반환 금액 표)
    (re.compile(r"등록금\s*반환|등록금\s*환불|반환\s*기준|"
                r"등록금\s*돌려|학기\s*개시일\s*전"),
     (re.compile(r"제42조\(등록금의?\s*반환\)"),)),
    # 학적 변동(전과·자퇴·재입학): 3개 조항이 별개 청크
    (re.compile(r"전과(?!\s*과정)|자퇴|재입학"),
     (re.compile(r"제52조\(전과\)"),
      re.compile(r"제38조\(자퇴\)"),
      re.compile(r"제76조\(재입학\)"))),
    # 입학·입학자격·편입학(신입학·편입학·전형)
    (re.compile(r"입학\s*자격|신입학|편입학(?!.{0,10}학점)|입학.{0,5}(절차|시기|전형)"),
     (re.compile(r"제48조\(입학자격\)"),
      re.compile(r"제49조\(재.?편입학\)"),
      re.compile(r"제23조\(입학절차"))),
    # 수업연한·재학연한
    (re.compile(r"수업\s*연한|재학\s*연한"),
     (re.compile(r"제32조\(대학의\s*수업연한\s*및\s*재학연한\)"),)),
    # 수강신청·수강취소·수강변경
    (re.compile(r"수강\s*신청(?!\s*기간)|수강\s*취소|수강\s*변경|수강\s*포기"),
     (re.compile(r"제66조\(수강신청\s*및\s*학점인정"),
      re.compile(r"제34조\(수강취소\)"))),
    # 출석·결석·출결
    (re.compile(r"출석|결석|출결|지각"),
     (re.compile(r"제89조\(출석인정\)"),
      re.compile(r"제41조\(결석자"))),
    # 학위수여·졸업논문·학위증
    (re.compile(r"학위\s*수여(?!\s*취소)|졸업\s*논문|학위증\s*받|학위증\s*수여"),
     (re.compile(r"제69조\(학위수여\)"),
      re.compile(r"제68조\(졸업논문\)"))),
    # 장학금(지급 기준·종류)
    (re.compile(r"장학금|국가\s*장학금|근로\s*장학|장학생"),
     (re.compile(r"제45조\(장학금\s*지급\)"),
      re.compile(r"제64조\(성적우수자\)"))),
    # 학칙(개정·시행)
    (re.compile(r"학칙\s*개정|학칙\s*시행|학칙\s*뭐"),
     (re.compile(r"제97조\(학칙의?\s*개정\)"),)),
    # 징계·이의신청
    (re.compile(r"징계|이의\s*신청|학사\s*징계|학생\s*징계"),
     (re.compile(r"제95조\(징계\)"),)),
    # 학적부 관리
    (re.compile(r"학적부|학적\s*변동|학적\s*관리"),
     (re.compile(r"제45조\(학적부"),)),
    # 학위수여 취소
    (re.compile(r"학위\s*수여\s*취소|학위\s*취소|졸업\s*취소"),
     (re.compile(r"제87조\(학위수여\s*취소\)"),
      re.compile(r"제71조\(졸업의?\s*취소\)"))),
    # 재수강·재이수
    (re.compile(r"재수강|재이수"),
     (re.compile(r"제26조\(재이수\)"),)),
    # 시간제 등록생
    (re.compile(r"시간제\s*등록|시간제\s*학생|학점\s*등록"),
     (re.compile(r"제72조\(시간제\s*등록생\)"),)),
)
# 학사요람 본문에서 '제N조(졸업·소요학점·이수·수료·등록금·휴학 등)' 패턴.
_REG_ARTICLE_RE = re.compile(
    r"제\d{1,3}조\([^)]*(?:졸업|소요학점|이수|수료|등록금|휴학|복학|전과)"
)
_REG_CREDIT_RE = re.compile(r"\d+\s*학점")
# 대학원 과정 마커. 학부 질의에 이런 청크가 끼면 노이즈(논문연구·특론·세미나류).
_GRAD_MARK_RE = re.compile(
    r"석사|박사|석박사|대학원|수료학점|논문연구|특론|특강|"
    r"석사학위과정|박사학위과정|전문박사|전문석사"
)
# 질의가 '대학원/석사/박사'를 명시했는지(명시 시 페널티 끄기).
_GRAD_QUERY_RE = re.compile(r"대학원|석사|박사|전문박사|전문석사")


def is_regulation_query(query: str) -> bool:
    """학사 규정류 질의 여부. _plan에서도 라이브 fetch 우회 판단에 재사용."""
    return bool(_REG_QUERY_RE.search(query or ""))

MEAL_FALLBACK_MSG = "현재 식단 정보를 불러올 수 없습니다. 홈페이지를 확인해주세요."

_WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
_REL = {"모레": 2, "내일": 1, "오늘": 0, "어제": -1}  # '모레'를 '내일'보다 먼저 검사


def resolve_target_date(query: str, now: datetime) -> tuple[date, str]:
    """질의의 시간 한정어('내일','금요일' 등)를 기준일(now) 대비 실제 날짜로 변환.
    반환: (대상 date, 라벨 '2026-05-28 (목)'). 한정어 없으면 오늘."""
    base = now.date()
    for term, off in _REL.items():
        if term in query:
            d = base + timedelta(days=off)
            return d, f"{d.isoformat()} ({_WEEKDAYS[d.weekday()][0]})"
    for i, wd in enumerate(_WEEKDAYS):
        if wd in query:  # 이번 주 해당 요일
            d = base - timedelta(days=base.weekday()) + timedelta(days=i)
            return d, f"{d.isoformat()} ({wd[0]})"
    return base, f"{base.isoformat()} ({_WEEKDAYS[base.weekday()][0]})"


class AcademicRetriever:
    def __init__(
        self,
        index_path: str,
        meta_path: str,
        embedder: Embedder,
        top_k: int = 3,
        score_threshold: float | None = None,  # None = 거부 비활성(캘리브레이션 유보)
        canonical_boost: float = 0.0,  # plus.cnu/sub05 canonical 점수 가산(요람 경쟁서 보호)
    ):
        self.embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.canonical_boost = canonical_boost
        self.index = faiss.read_index(self._require(index_path))
        meta = json.loads(Path(self._require(meta_path)).read_text(encoding="utf-8"))
        # 모듈1 은 meta 를 LIST 로 저장 → faiss 위치 인덱스(0..N-1)가 곧 리스트 위치
        if not isinstance(meta, list):
            raise ValueError("meta 포맷은 list 여야 함 (모듈1 산출 계약)")
        self.meta: list[dict] = meta

    @staticmethod
    def _require(path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"인덱스 파일 없음: {path}")
        return path

    def add_chunks(self, chunks: list[dict],
                   persist_paths: tuple[str, str] | None = None) -> int:
        """증분 핫스왑: 임베딩 → index.add → meta append (전체 재빌드 X).
        persist_paths=(index, meta) 주면 디스크에 원자적 영속(durability)."""
        if not chunks:
            return 0
        texts = [f"{c['title']}\n{c['content']}" if c.get("title") else c["content"]
                 for c in chunks]
        emb = np.ascontiguousarray(self.embedder.encode(texts), dtype=np.float32)
        self.index.add(emb)            # 인메모리 즉시 반영
        self.meta.extend(chunks)
        if persist_paths:
            _atomic_write_index(self.index, Path(persist_paths[0]))
            _atomic_write_text(Path(persist_paths[1]),
                               json.dumps(self.meta, ensure_ascii=False))
        return len(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        # Embedder 계약: list[str] 입력 → [N, dim]. 단일 질의는 [query].
        use_k = top_k or self.top_k
        is_reg = is_regulation_query(query)
        _is_grad_query = bool(_GRAD_QUERY_RE.search(query or ""))
        q = np.ascontiguousarray(self.embedder.encode([query]), dtype=np.float32)
        pool = max(use_k, 30) if (self.canonical_boost or is_reg) else use_k
        scores, idxs = self.index.search(q, pool)

        scored: list[tuple[float, float, dict]] = []  # (adj, raw, meta)
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.meta):  # faiss 미충족 슬롯 = -1
                continue
            m = self.meta[idx]
            adj = float(score)
            if self.canonical_boost and "plus.cnu.ac.kr/html/kr/sub05" in (m.get("source_url") or ""):
                adj += self.canonical_boost  # canonical 가산 후 재정렬
            # 규정 질의: 학사요람 권위 청크 boost. '제N조(졸업/이수/수료/등록금…)' + 학점 숫자가
            # 본문에 있으면 진짜 답이 들어있는 청크 → 추가 boost.
            if is_reg and "학사요람" in (m.get("title") or ""):
                adj += 0.04
                content = m.get("content") or ""
                if _REG_ARTICLE_RE.search(content) and _REG_CREDIT_RE.search(content):
                    adj += 0.05
                # 학부 질의(대학원 미명시)에 대학원 마커가 3+ 들어간 청크는 노이즈로 페널티.
                if not _is_grad_query and len(_GRAD_MARK_RE.findall(content)) >= 3:
                    adj -= 0.06
            scored.append((adj, float(score), m))
        scored.sort(key=lambda x: -x[0])

        chunks: list[RetrievedChunk] = []
        for _adj, raw, m in scored[:use_k]:
            chunks.append(
                RetrievedChunk(
                    doc_id=m["doc_id"], content=m["content"],
                    title=m.get("title"), source_url=m.get("source_url"),
                    score=raw,  # 원 cosine 보존(boost는 정렬에만)
                )
            )

        # 정의 질의: 임베딩이 정의 청크와 거리가 멀어 top에 못 잡힘 → 학사요람 정의 단락을
        # 매핑 테이블(_DEFINITION_PREPEND_MAP)로 키워드 매칭해 강제 prepend. 매핑 entry의
        # 마커가 여러 개면 각각 매칭되는 청크를 모두 prepend(휴학·복학·제적 같은 분산 정의 대응).
        if is_reg:
            for q_pat, content_markers in _DEFINITION_PREPEND_MAP:
                if not q_pat.search(query):
                    continue
                to_prepend: list = []
                category_claimed = False  # 자연 retrieve로 이미 잡힌 경우도 카테고리 claim
                for marker in content_markers:
                    # 자연 retrieve로 이미 chunks에 있는 청크인지 확인 (duplicate)
                    if any(marker.search(ch.content or "") for ch in chunks):
                        category_claimed = True
                        continue
                    # chunks에 없으면 meta 선형 스캔
                    for m in self.meta:
                        if "학사요람" not in (m.get("title") or ""):
                            continue
                        c = m.get("content") or ""
                        if not marker.search(c):
                            continue
                        if any(ch["doc_id"] == m["doc_id"] for ch in to_prepend):
                            break
                        to_prepend.append({
                            "doc_id": m["doc_id"], "content": c,
                            "title": m.get("title"), "source_url": m.get("source_url"),
                        })
                        category_claimed = True
                        break
                # 첫 마커가 chunks[0]이 되도록 reverse 순으로 insert
                for ch in reversed(to_prepend):
                    chunks.insert(0, RetrievedChunk(
                        doc_id=ch["doc_id"], content=ch["content"],
                        title=ch["title"], source_url=ch["source_url"],
                        score=1.0,
                    ))
                chunks = chunks[:use_k]
                if category_claimed:
                    break  # 매칭된 카테고리는 추가 entry 발동 차단

        if not chunks:
            return RetrievalResult(
                intent=Intent.ACADEMIC, is_fallback=True,
                fallback_message="관련 학사 정보를 찾지 못했습니다.",
            )
        if self.score_threshold is not None and chunks[0].score < self.score_threshold:
            return RetrievalResult(
                intent=Intent.ACADEMIC, is_fallback=True,
                fallback_message="질의와 일치하는 학사 규정을 찾지 못했습니다.",
            )
        return RetrievalResult(intent=Intent.ACADEMIC, chunks=chunks)


class CafeteriaRetriever:
    """주간 식단 캐시(WeeklyMealCache) 기반. 질의의 날짜 한정어로 대상 요일 슬롯 추출."""

    def __init__(self, cache_path: str = "data/cnu_meal_mock.json",
                 stale_after_days: int = 1):
        import threading
        self.cache_path = cache_path
        self.stale_after_days = stale_after_days
        self.fallback_message = MEAL_FALLBACK_MSG
        self._cached: WeeklyMealCache | None = None
        self._refresh_lock = threading.Lock()
        self._last_refresh_attempt: datetime | None = None
        self.reload_cache_from_disk()

    def update_cache_directly(self, fresh: WeeklyMealCache) -> None:
        """웹훅/동기화가 수신·검증한 WeeklyMealCache 를 메모리에 즉시 swap (객체 재할당, GIL 안전)."""
        self._cached = fresh

    def reload_cache_from_disk(self) -> None:
        """서버 startup 용 로컬 로더. 없거나 깨졌으면 None (fallback 유도).
        로드 후 stale이면 즉시 라이브 크롤 시도."""
        try:
            if not os.path.exists(self.cache_path):
                self._cached = None
            else:
                raw = json.loads(Path(self.cache_path).read_text(encoding="utf-8"))
                self._cached = WeeklyMealCache(**raw)
        except Exception:
            self._cached = None
        # 시작 시 캐시가 없거나 stale이면 즉시 라이브 크롤
        c = self._cached
        now = datetime.now()
        needs_refresh = (
            c is None or not c.days
            or (now - c.timestamp) > timedelta(days=self.stale_after_days)
        )
        if needs_refresh:
            try:
                from .meal_crawler import MEAL_URL, crawl_week
                print("[meal] 시작 시 식단 라이브 크롤 중...")
                wc = crawl_week(MEAL_URL)
                if wc.days:
                    self.update_cache_directly(wc)
                    try:
                        Path(self.cache_path).write_text(
                            wc.model_dump_json(), encoding="utf-8")
                    except Exception:
                        pass
                    print(f"[meal] 라이브 크롤 완료: {len(wc.days)}일치 수집")
            except Exception as e:
                print(f"[meal] 시작 시 라이브 크롤 실패: {e}")

    def _auto_refresh_if_needed(self, now: datetime, target_iso: str) -> None:
        """캐시가 비었거나 stale이거나 target 날짜 미보유 시 자동 크롤·핫스왑.
        1차: 전체 주간 크롤 시도. 2차: 해당 날짜만 개별 fetch.

        쿨다운 정책(폭주 방지 vs 복구):
          - 성공(target_iso 확보)하면 3분 쿨다운 유지 → 과도한 재크롤 차단.
          - 실패하면 쿨다운을 리셋(_last_refresh_attempt=None) → 다음 질의에서 즉시 재시도.
            (Colab 등 일시적 접속 실패 후 곧바로 복구 가능해야 하므로 실패를 가두지 않음)"""
        c = self._cached
        needs = (
            c is None or not c.days
            or target_iso not in c.days
            or (now - c.timestamp) > timedelta(days=self.stale_after_days)
        )
        if not needs:
            return
        with self._refresh_lock:
            if self._last_refresh_attempt and \
                    (now - self._last_refresh_attempt) < timedelta(minutes=3):
                return
            self._last_refresh_attempt = now

        got_target = False
        try:
            from .meal_crawler import MEAL_URL, crawl_week, crawl_meal, fetch_meal_html, parse_meal_html
            # 1차: 전체 주간 크롤
            wc = crawl_week(MEAL_URL)
            if wc.days:
                self.update_cache_directly(wc)
                try:
                    Path(self.cache_path).write_text(
                        wc.model_dump_json(), encoding="utf-8")
                except Exception:
                    pass
            # 2차: target 날짜가 아직 없으면 해당 날짜만 개별 fetch
            c = self._cached
            if c and target_iso not in (c.days or {}):
                try:
                    from datetime import date as _date
                    target_date = _date.fromisoformat(target_iso)
                    mc = parse_meal_html(fetch_meal_html(MEAL_URL, target=target_date))
                    if mc.menus:
                        if c.days is None:
                            c.days = {}
                        c.days[target_iso] = mc.menus
                        try:
                            Path(self.cache_path).write_text(
                                c.model_dump_json(), encoding="utf-8")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[meal] {target_iso} 개별 크롤 실패 → {type(e).__name__}: {e}")
            c = self._cached
            got_target = bool(c and c.days and target_iso in c.days)
        except Exception as e:
            print(f"[meal] 자동 갱신 실패 → {type(e).__name__}: {e}")

        # 실패 시 쿨다운 리셋 → 다음 질의에서 즉시 재시도 가능
        if not got_target:
            with self._refresh_lock:
                self._last_refresh_attempt = None
            print(f"[meal] {target_iso} 갱신 실패 — 쿨다운 해제, 다음 질의에서 재시도")

    def retrieve(self, query: str, now: datetime) -> RetrievalResult:
        target, label = resolve_target_date(query, now)
        self._auto_refresh_if_needed(now, target.isoformat())
        c = self._cached
        if c is None or not c.days:
            return self._fallback(self.fallback_message)
        if now - c.timestamp > timedelta(days=self.stale_after_days):  # 주간 신선도 만료
            return self._fallback(self.fallback_message)
        menus = c.days.get(target.isoformat())
        if not menus:  # 주말 휴무 / 미수집 날짜
            return self._fallback(
                f"{label} 식단 정보가 없습니다(주말 휴무이거나 수집되지 않음). "
                "충남대 홈페이지를 확인해주세요.")
        return RetrievalResult(intent=Intent.CAFETERIA, menus=menus, meal_date_label=label)

    def _fallback(self, msg: str) -> RetrievalResult:
        return RetrievalResult(
            intent=Intent.CAFETERIA, is_fallback=True, fallback_message=msg)
