"""모듈 경계 공통 스키마 (Pydantic v2).

모든 모듈은 이 타입으로 입출력 정합성을 강제한다. 모듈 1은 이 중
CNUExtractedChunk / IndexBuildConfig / BuildReport 를 사용.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Intent(str, Enum):
    ACADEMIC = "academic"        # 학사 (정적, FAISS 검색)
    CAFETERIA = "cafeteria"      # 학식 (동적, 캐시 + 신선도 검증)
    TEMPORAL_NOTICE = "temporal_notice"  # 시의성 공지/최신 게시판 — 정적 코퍼스에 부재 → 홈페이지 안내
    OUT_OF_SCOPE = "out_of_scope"  # 거부 응답 (LLM 우회)


class CNUExtractedChunk(BaseModel):
    """Colab → 서버로 넘어오는 학사 청크 1건의 계약.

    extra='forbid': 계약에 없는 key 가 들어오면 ValidationError. 오타/스키마
    드리프트를 진입점에서 차단하기 위함.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    content: str
    title: str | None = None
    source_url: str | None = None
    file_type: Literal["html", "pdf", "hwp"] = "html"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VLLMIntentResponse(BaseModel):
    """LLM JSON-mode 출력 계약. LLM 출력은 노이즈 경계라 extra='ignore'
    (forbid 면 키 하나 더 붙어도 parse 실패 → 조용히 fallback 으로 샘)."""

    model_config = ConfigDict(extra="ignore")

    llm_intent: Literal["academic", "cafeteria", "temporal_notice", "out_of_scope"]
    refined_query: str = Field(description="오타/축약 교정된 표준 한국어 평서문")
    reason: str = ""


class IntentResult(BaseModel):
    """모듈 2 최종 출력 — 오케스트레이터(모듈 4)로 전달."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    refined_query: str | None = None
    matched_keyword: str | None = None
    method: Literal[
        "fast_cache_hit",
        "fast_regex_cafeteria",
        "fast_canonical_cafeteria",  # 정규화된 '제N학생회관' 결정론 학식 오버라이드
        "fast_temporal_notice",      # '최신 공지' 결정론 공지 오버라이드
        "llm_json_precise_router",
        "router_fallback",
    ]


class IndexBuildConfig(BaseModel):
    data_path: str = "data/cnu_academic_mock.json"
    save_path: str = "storage/faiss_index.bin"
    model_name: str = "nlpai-lab/KURE-v1"
    device: str = "cuda:1"   # GPU1 전용. cuda:0 은 vLLM → 침범 금지.
    embed_dim: int = 1024    # KURE-v1
    min_content_chars: int = 10  # 공백 제외 이 미만 content 는 검색 노이즈 → 제외


class BuildReport(BaseModel):
    ok: bool
    num_chunks: int          # 실제 인덱싱된 청크 수
    num_skipped: int = 0     # 초단문으로 제외된 청크 수
    index_path: str
    meta_path: str | None = None
    embed_dim: int | None = None
    error: str | None = None


# === 모듈 3 (Retriever & Cache Verifier) ===

class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    content: str
    title: str | None = None
    source_url: str | None = None
    score: float             # cosine (IndexFlatIP, 정규화) ∈ [-1, 1]


class DailyMenu(BaseModel):
    model_config = ConfigDict(extra="forbid")

    place: str               # 식당명 (제1~4학생회관, 생활과학대학). 실데이터 enum 가변 → str
    meal_type: Literal["조식", "중식", "석식"]
    target: Literal["직원", "학생"]   # 대상 — 직원/학생 가격·메뉴가 분리됨 (실원본 차원)
    menu_list: list[str]     # [0]=정식(가격), 이후 정제된 음식 토큰 (영문주석/&·/ 정제)


class MealCache(BaseModel):
    """단일일자 학식 캐시 계약 (구버전 — 웹훅/일부 테스트 호환용)."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime      # 적재 시점 (24h 신선도 측정)
    target_date: date        # 식단 기준 날짜
    menus: list[DailyMenu] = Field(default_factory=list)


class WeeklyMealCache(BaseModel):
    """주간 학식 캐시 — 날짜(ISO)별 DailyMenu 리스트. RC-2 B안: '내일/금요일' 등 지원."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime              # 적재 시점 (주간 단위 신선도)
    week_start: date                 # 해당 주 월요일
    days: dict[str, list[DailyMenu]] = Field(default_factory=dict)  # "2026-05-28" → menus


class NoticeItem(BaseModel):
    """게시판 공지 1건 (온디맨드 라이브 파싱). article_no 내림차순 = 최신순."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str | None = None
    dept: str | None = None
    posted: str | None = None        # 게시일(파싱되면)
    article_no: int | None = None    # 최신순 정렬 키


class RetrievalResult(BaseModel):
    """모듈 3 → 오케스트레이터(모듈 4) 전달 계약."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    chunks: list[RetrievedChunk] = Field(default_factory=list)  # 학사 경로
    menus: list[DailyMenu] = Field(default_factory=list)        # 학식 경로
    meal_date_label: str | None = None  # 학식: 해석된 대상 날짜(예: "2026-05-28 (목)")
    is_fallback: bool = False
    fallback_message: str | None = None


# === 모듈 4 (Generator & FastAPI) ===

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    session_id: str | None = None


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    source_url: str | None = None


class CNUBotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    references: list[Reference] = Field(default_factory=list)  # 학식/OOS는 빈 배열
    intent: Intent
    is_fallback: bool = False  # 캐시만료·거부·OOS → LLM 우회
    refined_query: str | None = None  # 라우터 정제문 (디버깅 가시성)


class IngestPayload(BaseModel):
    """Colab 웹훅 수신. academic=청크 dict 배열, meal=[MealCache dict] (단일, data[0])."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["academic", "meal"]
    data: list[dict[str, Any]]
