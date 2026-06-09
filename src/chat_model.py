"""CNU Campus ChatBot — 통합 LLM/RAG 엔트리포인트 (Colab T4 단일 GPU).

핵심:
- Qwen2.5-7B-Instruct 4bit (bitsandbytes) + KURE-v1 임베더를 모두 cuda:0 에 로드
- Orchestrator(라우터·검색·생성) 인스턴스를 모듈 전역으로 한 번만 빌드 → 게이트웨이/워커 분리 없음
- 5-way 분류 (졸업요건/공지/학사일정/식단/셔틀) 별도 함수
- 학식 자동 캐시 트리거(`module3_retriever.CafeteriaRetriever`)는 그대로 동작
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_PKG_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_ROOT.parent

# ---------------------------------------------------------------------------
# Orchestrator (지연 로드)
# ---------------------------------------------------------------------------
_orch: Any = None
_orch_lock = threading.Lock()


def _build_orch():
    """KURE + Qwen 모두 cuda:0 에 로드한 Orchestrator. 모듈 첫 호출 시에만."""
    from cnubot.module1_indexer import KUREEmbedder
    from cnubot.module2_hf_handler import HFIntentLLM
    from cnubot.module3_retriever import AcademicRetriever, CafeteriaRetriever
    from cnubot.module4_api import (
        ACADEMIC_INDEX_PATH,
        ACADEMIC_META_PATH,
        FOODCOURT_PATH,
        INDEX_PATH,
        MEAL_CACHE_PATH,
        META_PATH,
        CNUGenerator,
        CNUHybridIntentRouter,
        Orchestrator,
        render_foodcourt,
    )
    from cnubot.module4_generator import HFAnswerLLM
    from cnubot.notice import NoticeService

    device = "cuda:0"
    embedder = KUREEmbedder("nlpai-lab/KURE-v1", device)
    if Path(ACADEMIC_INDEX_PATH).exists():
        academic = AcademicRetriever(
            ACADEMIC_INDEX_PATH, ACADEMIC_META_PATH,
            embedder=embedder, top_k=5, canonical_boost=0.04,
        )
    else:
        academic = AcademicRetriever(INDEX_PATH, META_PATH, embedder=embedder, top_k=3)
    cafeteria = CafeteriaRetriever(cache_path=MEAL_CACHE_PATH)
    llm = HFAnswerLLM("Qwen/Qwen2.5-7B-Instruct", device)
    _ = llm.generate("안녕")  # warm-up
    router = CNUHybridIntentRouter(llm=HFIntentLLM(backend=llm))
    foodcourt = render_foodcourt(FOODCOURT_PATH) if Path(FOODCOURT_PATH).exists() else None
    notice = NoticeService() if (_PKG_ROOT / "cnubot" / "data" / "dept_registry.json").exists() else None
    return Orchestrator(
        router, academic, cafeteria, CNUGenerator(llm),
        foodcourt_text=foodcourt, notice=notice,
    )


def get_orchestrator():
    global _orch
    if _orch is None:
        with _orch_lock:
            if _orch is None:
                _orch = _build_orch()
    return _orch


# ---------------------------------------------------------------------------
# 5-way 분류 (LLM prompt-based)
# ---------------------------------------------------------------------------
LABEL_KR = ["졸업요건", "학교 공지사항", "학사일정", "식단 안내", "통학/셔틀 버스"]
_CLS_SYS = (
    "당신은 충남대학교 학생용 챗봇의 질문 분류기입니다. "
    "사용자의 질문을 정확히 아래 5개 카테고리 중 하나로 분류해 0~4 숫자 한 글자만 출력하세요. "
    "다른 텍스트는 절대 출력하지 마세요.\n"
    "0: 졸업요건 — 졸업학점, 전공/교양 졸업 요건, 학위 수여 조건 등\n"
    "1: 학교 공지사항 — 학교·학과 공지, 게시판 알림, 최근 공지 등\n"
    "2: 학사일정 — 수강 신청/정정 기간, 학기 시작/종료, 시험 기간 등\n"
    "3: 식단 안내 — 교내 학생식당 메뉴·시간표·가격\n"
    "4: 통학/셔틀 버스 — 셔틀버스 시간표·정류장·노선·운행 여부"
)


def classify(question: str) -> int:
    """질문 → 0..4 라벨. 모호한 출력은 정규식 폴백."""
    orch = get_orchestrator()
    llm = orch.generator.llm
    prompt = (
        f"<|im_start|>system\n{_CLS_SYS}<|im_end|>\n"
        f"<|im_start|>user\n질문: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    out = llm.generate(prompt, max_new_tokens=4)
    m = re.search(r"[0-4]", out)
    return int(m.group()) if m else _heuristic_label(question)


def _heuristic_label(q: str) -> int:
    """LLM 출력이 망가졌을 때의 폴백 키워드 휴리스틱."""
    q = q.lower()
    if any(k in q for k in ["셔틀", "통학", "버스 노선", "정류장"]):
        return 4
    if any(k in q for k in ["학식", "식단", "메뉴", "점심", "저녁", "조식", "중식", "석식"]):
        return 3
    if any(k in q for k in ["수강 신청", "수강신청", "정정 기간", "학사일정", "학기 시작", "기말고사", "중간고사"]):
        return 2
    if any(k in q for k in ["공지", "알림", "최근"]):
        return 1
    return 0


def classify_batch(in_path: str, out_path: str) -> None:
    """test_cls.json (list of {question}) → outputs/cls_output.json (list of {question, label})."""
    items = json.loads(Path(in_path).read_text(encoding="utf-8"))
    results = [{"question": it["question"], "label": classify(it["question"])} for it in items]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 챗봇 응답 (RAG + 실시간 식단)
# ---------------------------------------------------------------------------
def respond(question: str) -> str:
    """질문 → 자연어 답변 (Orchestrator 전체 파이프라인)."""
    orch = get_orchestrator()
    resp = orch.handle(question)
    return resp.answer or ""


def chat_batch(in_path: str, out_path: str) -> None:
    """test_chat.json (list of {user}) → outputs/chat_output.json (list of {user, model})."""
    items = json.loads(Path(in_path).read_text(encoding="utf-8"))
    results = [{"user": it["user"], "model": respond(it["user"])} for it in items]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
