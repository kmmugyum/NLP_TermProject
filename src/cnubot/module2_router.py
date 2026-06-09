"""모듈 2: 하이브리드 의도 라우터 (3-way: academic / cafeteria / out_of_scope).

레이어: [1] 인메모리 LRU 캐시(0ms) → [2] CPU 정규식 단순 학식(0ms)
        → [3] LLM JSON-mode 분류 + refined_query (GPU0 vLLM, 주입식)
        → [4] refined_query 명사사전 크로스체크 + 학사 우선 override
예외 시: Safe Parsing Guard → 원문 기반 룰 fallback.

LLM 클라이언트는 주입(IntentLLM). 미주입/예외 시 fallback 으로 graceful degrade
→ vLLM 미가동(현 상태)에서도 결정론 로직을 단위 테스트 가능.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from .schemas import Intent, IntentResult, VLLMIntentResponse

SYSTEM_PROMPT = (
    "You are an intent classification router for Chungnam National University (CNU) bot.\n"
    "먼저 'reason' 필드에서 단계적으로 추론한 뒤 결론을 내리시오. "
    "아래 JSON만, 키 순서 그대로 출력(reason 먼저):\n"
    '{"reason": "단계별 판단 — ①교내 학식 메뉴/가격/운영인가 ②외부음식·요리·영양·'
    '브랜드인가 ③학사 규정/제도/절차 내용인가 ④최신 공지/게시판 등 실시간 정보인가 '
    '⑤비학교 주제인가", '
    '"llm_intent": "academic"|"cafeteria"|"temporal_notice"|"out_of_scope", '
    '"refined_query": "<여기에 사용자 질문을 오타·축약만 고쳐 다시 적기>"}\n\n'
    "분류 정의:\n"
    "- academic: CNU 학사 행정·규정·제도·절차의 *내용*. 졸업·전과·휴학·재이수·"
    "학사경고·장학/등록 규정, 신청 방법·자격·기준 등 (요람·학칙에 상주). 또한 교내 "
    "편의시설 위치(편의점·GS25·은행·우체국·카페·서점·문구점이 어느 건물/층인지)도 academic.\n"
    "- cafeteria: CNU 교내 학생식당(제1~4학생회관·생활과학대학)의 메뉴/가격/운영시간. "
    "'어제/오늘/내일/요일' 등 날짜가 붙어도 교내 식사 메뉴를 물으면 cafeteria.\n"
    "- temporal_notice: 그때그때 바뀌는 최신 게시물/공지 — '오늘/이번주/최근/새로 올라온' "
    "공지사항·학과공지·게시판 최신글, '가장 최근 공지', 사업단소식 등 학과/부서 공지 조회. "
    "(학교 관련이므로 out_of_scope 아님)\n"
    "- out_of_scope: 그 외 전부 — 외부 식당·브랜드(성심당 등)·요리법·레시피·"
    "영양/칼로리/다이어트 상담·금융·연예·날씨.\n\n"
    "중요 규칙:\n"
    "1. 학식과 학사규정/학자금이 섞이면 최종 목적은 학사 → academic "
    "(예: '1학에서 밥먹으며 전과 규정 확인' → academic).\n"
    "2. 교내 학식이 아닌 일반 음식·요리·영양 상담은 cafeteria 아님 → out_of_scope.\n"
    "3. 모호한 약어는 문맥으로: '2학'은 보통 '제2학생회관'(cafeteria)이지 '2학점'이 아님.\n"
    "4. 규정/제도/절차의 *내용*을 물으면 academic (예: '휴학 신청 방법', "
    "'등록금 납부 규정', '재이수 규정'). 반면 '오늘 공지', '최근 공지 목록', "
    "'이번주 새 게시글'처럼 *최신 게시물/실시간 일정*을 물으면 temporal_notice.\n"
    "5. '이번주 금요일 점심', '내일 학식', '어제 메뉴'처럼 *날짜+교내 식사*는 "
    "temporal_notice나 out_of_scope가 아니라 cafeteria이다.\n"
    "6. 교내 편의시설이 *어디/어느 건물*에 있는지 묻는 위치 질문 — 편의점(GS25·CU 등)·은행·"
    "우체국·카페·서점·문구점·복사실 등 — 은 academic이다(식당 메뉴가 아니므로 cafeteria 아님, "
    "학교 정보이므로 out_of_scope 아님)."
)


# ---------------------------------------------------------------------------
# 동의어 정규화 — 표면형(3학·3학생회관)을 캐논의컬(제3학생회관)로 입구에서 1회 치환.
# 정규식을 분류 로직 곳곳에 흩뿌리는 대신, 질의를 표준 용어로 먼저 통일한다.
# → 라우터·메뉴필터 등 하류가 모두 일관된 용어를 받음(유지보수 지점 1곳).
# ---------------------------------------------------------------------------
# (표면형 정규식, 캐논의컬) 순서대로 적용. 'N학'은 식사 맥락이 함께일 때만 학생회관으로 치환.
_MEAL_CONTEXT_RX = re.compile(r"학식|메뉴|식단|점심|저녁|아침|조식|중식|석식|식사|급식|먹")
_HALL_FULL_RX = re.compile(r"(?:제\s*)?([1-4])\s*학생회관")      # 제3학생회관 / 3학생회관
_HALL_ABBR_RX = re.compile(r"(?<![가-힣])([1-4])\s*학(?![점년기위과부사자생])")  # 줄임말 'N학'
# 정규화 후 '제N학생회관'은 100% 식당 의미(학점/학년과 혼동 불가) → 결정론적 학식 신호.
_HALL_CANONICAL_RX = re.compile(r"제[1-4]학생회관|생활과학대학")

# 공지 결정론 신호: '공지/공지사항/게시판/소식' 등 게시판 조회 명사.
# 이 명사가 있고 학사 규정 키워드가 없으면 temporal_notice 확정(LLM의 academic 오분류 우회).
# recency('최신/최근') 동반은 불필요 — '학과 공지', '공지 알려줘'처럼 시점어 없는
# 공지 조회도 게시판 조회이므로 temporal_notice 가 맞다(학사 키워드가 academic 가드).
# '공지'는 '인공지능'의 '공지'와 오매칭되므로 앞에 '인'이 오면 제외(negative lookbehind).
_NOTICE_NOUN_RX = re.compile(r"(?<!인)공지|게시판|게시글|게시물|새\s*글|소식|알림")


def normalize_synonyms(query: str) -> str:
    """질의의 식당 표면형을 '제N학생회관'으로 통일.
    - '3학생회관'/'제 3 학생회관' → '제3학생회관'
    - '3학'(줄임말)은 식사 맥락이 있을 때만 → '제3학생회관' (아니면 '3학점/3학년' 보호)
    """
    q = _HALL_FULL_RX.sub(lambda m: f"제{m.group(1)}학생회관", query)
    if _MEAL_CONTEXT_RX.search(q):  # 식사 맥락이 있을 때만 줄임말 확장
        q = _HALL_ABBR_RX.sub(lambda m: f"제{m.group(1)}학생회관", q)
    return re.sub(r"\s+", " ", q).strip()  # 치환 후 중복/경계 공백 정리


class IntentLLM(Protocol):
    """주입 가능한 분류 LLM 계약. classify(query) -> VLLMIntentResponse 용 dict."""

    def classify(self, query: str) -> dict: ...


class VLLMIntentLLM:
    """실제 vLLM(OpenAI 호환) 클라이언트 래퍼. GPU0 의 Qwen2.5-7B 사용.

    주의: vLLM 서버가 떠 있어야 동작. 미가동 시 router 가 fallback 으로 우회.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: str = "EMPTY",
        timeout: float = 10.0,
    ):
        from openai import OpenAI  # 지연 import — CPU-only 테스트는 openai 불필요

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def classify(self, query: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)


class CNUHybridIntentRouter:
    def __init__(self, llm: IntentLLM | None = None, max_cache_size: int = 1000):
        self.llm = llm
        self.cache: dict[str, IntentResult] = {}
        self.max_cache_size = max_cache_size

        # 레벨2: 단순 학식 질의만 (공백 제거 후 매칭). 좁은 게 의도 — 나머지는 LLM/크로스체크.
        self.fast_cafeteria_rx = re.compile(
            r"^(오늘|내일)?(학식|메뉴|식단|점심|저녁|조식|중식|석식)(뭐야|메뉴|식단|알려줘)?$"
        )


        # 도메인 명사 사전 — 이제 _fallback(LLM 다운) 전용. 정상 경로는 CoT LLM 신뢰.
        self.ACADEMIC_KEYWORDS = {
            "졸업", "이수학점", "학점", "학사", "전과", "복수전공", "부전공", "휴학",
            "복학", "수강신청", "장학금", "등록금", "규정", "학자금", "학자금대출",
            "한국장학재단",
        }
        self.CAFETERIA_KEYWORDS = {
            "학식", "메뉴", "식단", "점심", "아침", "저녁", "1학", "2학", "3학",
            "학생회관", "기숙사식당", "생활관식당",
        }
        self.OUT_OF_SCOPE_KEYWORDS = {
            "주식", "비트코인", "코인", "부동산", "재테크", "토토", "로또", "연예인",
            "아이돌", "날씨", "신용대출", "담보대출", "영끌", "갭투자",
        }

    def _cross_check(self, llm_intent: str, refined_query: str) -> tuple[Intent, str | None]:
        """refined_query 를 사전과 substring 대조. 학사 우선 override, 순수 금융만 거부."""
        acad = next((k for k in self.ACADEMIC_KEYWORDS if k in refined_query), None)
        cafe = next((k for k in self.CAFETERIA_KEYWORDS if k in refined_query), None)
        oos = next((k for k in self.OUT_OF_SCOPE_KEYWORDS if k in refined_query), None)

        if oos and not (acad or cafe):
            return Intent.OUT_OF_SCOPE, oos
        if acad:  # 혼합 문맥(학자금대출 등)은 학사로 승격
            return Intent.ACADEMIC, acad
        # cafe 키워드 override는 LLM이 명시적 out_of_scope 로 판정한 경우엔 적용 안 함
        # ('다이어트 식단'→'식단', '오늘 저녁 뭐 해먹지'→'저녁' 같은 음식-OOD 오승격 방지)
        if cafe and llm_intent != Intent.OUT_OF_SCOPE.value:
            return Intent.CAFETERIA, cafe
        return Intent(llm_intent), None  # 키워드 0매칭 / LLM OOS 존중 → LLM 판단 채택

    def get_intent(self, query: str) -> IntentResult:
        # 동의어 정규화: '3학/3학생회관' → '제3학생회관'. 입구에서 1회 → 하류 전체 일관.
        cleaned = normalize_synonyms(query.strip())
        key = cleaned.replace(" ", "")

        # 레벨1: 캐시
        if key in self.cache:
            return self.cache[key].model_copy(update={"method": "fast_cache_hit"})

        # 레벨2: 단순 학식 정규식
        if self.fast_cafeteria_rx.match(key):
            res = IntentResult(
                intent=Intent.CAFETERIA, refined_query=cleaned,
                matched_keyword="regex", method="fast_regex_cafeteria",
            )
            self._update_cache(key, res)
            return res

        # 레벨2.5: 결정론적 학식 오버라이드.
        # 정규화로 '제N학생회관'(또는 생활과학대학)이 확정됐고 학사 키워드가 없으면,
        # 7B LLM의 비일관 분류('석식'→학사 등)를 우회해 CAFETERIA 확정.
        # '제N학생회관'은 표준화된 식당명이라 학점/학년 등과 혼동 불가 → 오승격 위험 없음.
        if _HALL_CANONICAL_RX.search(cleaned):
            if not any(k in cleaned for k in self.ACADEMIC_KEYWORDS):
                res = IntentResult(
                    intent=Intent.CAFETERIA, refined_query=cleaned,
                    matched_keyword="canonical_hall", method="fast_canonical_cafeteria",
                )
                self._update_cache(key, res)
                return res

        # 레벨2.6: 결정론적 공지 오버라이드.
        # '공지/공지사항/게시판' 등 게시판 조회 명사가 있으면 temporal_notice 확정.
        # '졸업 규정', '휴학 신청 공지'처럼 학사 규정 키워드가 섞이면 제외(규정 *내용*은 academic).
        if _NOTICE_NOUN_RX.search(cleaned):
            if not any(k in cleaned for k in self.ACADEMIC_KEYWORDS):
                res = IntentResult(
                    intent=Intent.TEMPORAL_NOTICE, refined_query=cleaned,
                    matched_keyword="recent_notice", method="fast_temporal_notice",
                )
                self._update_cache(key, res)
                return res

        # 레벨3: CoT LLM 분류 — 추론된 intent를 신뢰 (키워드 cross_check override 제거).
        # 규칙 기반 override는 '2학'→'2학점'/'식단'→cafeteria 같은 오승격을 유발했음.
        # cross_check 키워드는 이제 LLM 다운 시 _fallback 에서만 사용(graceful degrade).
        if self.llm is not None:
            try:
                data = VLLMIntentResponse(**self.llm.classify(cleaned))
                # refined_query 가드: LLM이 스키마 placeholder/설명문을 echo하면 원문 사용
                # (echo 시 그 텍스트로 검색되어 엉뚱한 결과가 나오던 버그 방지)
                refined = (data.refined_query or "").strip()
                if (len(refined) < 2 or refined.startswith("<")
                        or "평서문" in refined or "교정" in refined or "다시 적" in refined):
                    refined = cleaned
                res = IntentResult(
                    intent=Intent(data.llm_intent), refined_query=refined,
                    matched_keyword=(data.reason[:40] or None),
                    method="llm_json_precise_router",
                )
                self._update_cache(key, res)
                return res
            except Exception as e:  # Safe Parsing Guard
                return self._fallback(cleaned, e)

        return self._fallback(cleaned, None)

    def _fallback(self, cleaned: str, err: Exception | None) -> IntentResult:
        # 원문에 동일 크로스체크 적용 (LLM 무 → llm_intent 기본 academic)
        intent, kw = self._cross_check("academic", cleaned)
        tag = type(err).__name__ if err else "no_llm"
        return IntentResult(
            intent=intent, refined_query=cleaned,
            matched_keyword=f"fallback:{tag}" if not kw else kw,
            method="router_fallback",
        )

    def _update_cache(self, key: str, value: IntentResult) -> None:
        if len(self.cache) >= self.max_cache_size:
            del self.cache[next(iter(self.cache))]  # LRU 근사: 가장 오래된 키 제거
        self.cache[key] = value
