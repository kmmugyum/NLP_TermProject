"""의도 분류 회귀 테스트 — intent 혼동 쌍 고정.

harness-100 chatbot-builder의 '의도 혼동 매트릭스' 권고에 따라,
실제로 발생했던 오분류 버그를 회귀 테스트로 고정한다:
  - cafeteria ↔ academic  ('석식' / 'N학생회관'이 학사로 새던 버그)
  - temporal_notice ↔ academic  ('최신 공지'가 학사로 새던 버그)
그리고 동의어 정규화의 오승격 방지('2학점'·'3학년'이 학식으로 새지 않음)를 검증한다.

실행:
  pytest src/tests/test_intent_routing.py        # pytest 있을 때
  python3 src/tests/test_intent_routing.py       # 표준 unittest (의존성 0)

설계: LLM(GPU) 없이 결정론 경로(레벨1~2.6 + fallback)만 검증한다.
      레벨2.5/2.6 오버라이드는 LLM 도달 전에 동작하므로, 이 버그들은
      GPU 없이도 재현·검증 가능하다(회귀 테스트가 로컬 CI에서 항상 돈다).
"""
from __future__ import annotations

import os
import sys
import unittest

# src/ 를 import 경로에 추가 (tests/ 의 부모)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CNU_NET_DISABLE", "1")  # 네트워크 차단(테스트 격리)

from cnubot.module2_router import CNUHybridIntentRouter, normalize_synonyms
from cnubot.schemas import Intent


class TestSynonymNormalization(unittest.TestCase):
    """동의어 정규화: 식당 표면형 → '제N학생회관' 캐논의컬."""

    def test_hall_full_form(self):
        self.assertEqual(normalize_synonyms("3학생회관 식단"), "제3학생회관 식단")
        self.assertEqual(normalize_synonyms("제 3 학생회관 식단"), "제3학생회관 식단")

    def test_hall_abbreviation_with_meal_context(self):
        # 'N학' 줄임말은 식사 맥락이 함께일 때만 확장
        self.assertEqual(normalize_synonyms("3학 점심"), "제3학생회관 점심")
        self.assertEqual(normalize_synonyms("2학 메뉴"), "제2학생회관 메뉴")
        self.assertEqual(normalize_synonyms("내일 3학 석식"), "내일 제3학생회관 석식")

    def test_academic_terms_preserved(self):
        # 'N학'이 학사 단어로 이어지면 정규화하지 않음(오승격 방지)
        for q in ["2학점 들어야 졸업", "3학년 졸업요건", "2학기 수강신청", "2학년 전과"]:
            self.assertEqual(normalize_synonyms(q), q, f"학사 용어가 변형됨: {q!r}")

    def test_whitespace_clean(self):
        # 치환 후 공백이 깨지지 않음
        self.assertEqual(normalize_synonyms("내일 3학생회관 학식 메뉴"),
                         "내일 제3학생회관 학식 메뉴")


class TestIntentRouting(unittest.TestCase):
    """결정론 경로(LLM 없음)에서 intent 오분류 회귀 고정."""

    def setUp(self):
        # llm=None → 레벨1~2.6 결정론 + fallback 만 동작(GPU 불필요)
        self.router = CNUHybridIntentRouter(llm=None)

    def assertIntent(self, query: str, expected: Intent):
        got = self.router.get_intent(query).intent
        self.assertEqual(got, expected,
                         f"{query!r} → {got.value} (기대: {expected.value})")

    # --- cafeteria 회귀: '석식'/'N학생회관'이 학사로 새던 버그 ---
    def test_cafeteria_canonical_hall(self):
        self.assertIntent("내일 3학생회관 석식", Intent.CAFETERIA)
        self.assertIntent("내일 제3학생회관 석식", Intent.CAFETERIA)
        self.assertIntent("오늘 2학 점심", Intent.CAFETERIA)
        self.assertIntent("내일 3학 점심", Intent.CAFETERIA)
        self.assertIntent("생활과학대학 메뉴", Intent.CAFETERIA)

    def test_cafeteria_simple_regex(self):
        # 레벨2 fast regex 경로
        self.assertIntent("오늘 학식", Intent.CAFETERIA)
        self.assertIntent("내일 메뉴", Intent.CAFETERIA)

    # --- temporal_notice 회귀: '최신 공지'가 학사로 새던 버그 ---
    def test_temporal_notice(self):
        self.assertIntent("컴퓨터인공지능학부 최신 공지", Intent.TEMPORAL_NOTICE)
        self.assertIntent("최근 공지", Intent.TEMPORAL_NOTICE)
        self.assertIntent("이번주 새 게시글", Intent.TEMPORAL_NOTICE)
        self.assertIntent("요즘 공지 뭐 올라옴", Intent.TEMPORAL_NOTICE)

    def test_temporal_notice_without_recency(self):
        # 시점어('최신/최근') 없이 공지 명사만 있어도 게시판 조회 → temporal_notice
        self.assertIntent("공지 알려줘", Intent.TEMPORAL_NOTICE)
        self.assertIntent("학과 공지", Intent.TEMPORAL_NOTICE)
        self.assertIntent("컴퓨터학부 공지사항", Intent.TEMPORAL_NOTICE)
        self.assertIntent("게시판 글 보여줘", Intent.TEMPORAL_NOTICE)

    # --- academic 유지: 규정 *내용* 질의는 학사로 남아야 함 ---
    def test_academic_preserved(self):
        # 식당 키워드/공지 키워드와 혼동되지 않고 academic 유지
        for q in ["졸업 규정", "전과 규정", "2학점 졸업요건", "3학년 수강신청"]:
            self.assertIntent(q, Intent.ACADEMIC)

    # --- 혼합 질의: 학사 키워드가 섞이면 오버라이드 안 함 ---
    def test_mixed_query_defers_to_academic(self):
        # '제3학생회관'이 있어도 '전과 규정'이 함께면 학사(오버라이드 제외)
        self.assertIntent("제3학생회관에서 전과 규정", Intent.ACADEMIC)

    # --- '인공지능'의 '공지' 오매칭 회귀: 학사 질의가 공지로 새면 안 됨 ---
    def test_ai_not_misrouted_to_notice(self):
        # '인공지능'에 '공지'가 들어있어 공지 오버라이드로 새던 버그.
        # llm=None 이면 academic fallback. 핵심은 temporal_notice 가 아니어야 함.
        for q in ["컴퓨터인공지능학부 미적분학은 몇 학년 과목이야",
                  "인공지능학과 졸업요건", "인공지능학과 1학년 교과목"]:
            self.assertNotEqual(self.router.get_intent(q).intent,
                                Intent.TEMPORAL_NOTICE,
                                f"{q!r} 가 공지로 오분류됨('인공지능'의 '공지' 오매칭)")


class TestOrchestratorRescueSignals(unittest.TestCase):
    """_plan 의 OOS/notice→academic 구제 + 셔틀 결정론 라우팅 회귀(정규식 레벨).

    실제 버그: 라이브 LLM 이 '미적분학은 몇 학년 과목이야?'를 out_of_scope 로,
    '셔틀버스 시간표'를 (셔틀 intent 부재로) out_of_scope 로 흘려 '범위 밖' 거부가 났다.
    이 구제는 module4_api 의 공유 정규식 상수로 동작하므로, 그 상수만 직접 검증한다
    (fastapi 미설치 환경에서는 skip — 정규식 자체는 GPU/네트워크 불필요)."""

    def setUp(self):
        try:
            from cnubot import module4_api as m
        except Exception as e:  # fastapi 등 미설치 → 로컬 CI skip
            self.skipTest(f"module4_api import 불가(의존성): {e}")
        self.m = m

    def test_academic_signal_rescues_verbose_curriculum(self):
        # 회화체/축약형 양쪽 모두 학사 신호로 잡혀야 함(OOS·notice 양 분기에서 구제).
        for q in ["미적분학은 몇 학년 과목이야?", "미적분학 몇학년과목?",
                  "컴퓨터인공지능학부 미적분학은 몇 학년 과목이야"]:
            self.assertTrue(self.m._ACADEMIC_SIGNAL_RE.search(q),
                            f"{q!r} 가 학사 신호로 안 잡힘 → OOS 거부 위험")
            self.assertFalse(self.m._NOTICE_SIGNAL_RE.search(q),
                             f"{q!r} 에 공지 신호 오매칭 → 학사 구제가 막힘")

    def test_academic_signal_does_not_overcapture_oos(self):
        # 진짜 범위 밖 질의는 학사 신호로 잡히면 안 됨(거부 유지).
        for q in ["성심당 빵 추천해줘", "오늘 날씨 어때?", "오늘 학식 메뉴"]:
            self.assertFalse(self.m._ACADEMIC_SIGNAL_RE.search(q),
                             f"{q!r} 가 학사 신호로 오매칭 → 잘못된 학사 라우팅")

    def test_shuttle_deterministic_match(self):
        # 셔틀류는 표면형(시간표/노선/첫차)이 달라도 광역 토큰으로 모두 잡혀야 함.
        for q in ["셔틀버스 시간표", "셔틀 노선", "통학버스 첫차", "스쿨버스 운행시간"]:
            self.assertTrue(self.m._SHUTTLE_RE.search(q),
                            f"{q!r} 가 셔틀로 안 잡힘 → OOS 거부 위험")
        for q in ["성심당 빵", "오늘 날씨", "미적분학 몇학년", "오늘 학식"]:
            self.assertFalse(self.m._SHUTTLE_RE.search(q),
                             f"{q!r} 가 셔틀로 오매칭")


class TestGitHubDataMode(unittest.TestCase):
    """GitHub 데이터 모드: Colab 에서 CNU 라이브 fetch 전면 차단."""

    def test_cnu_fetch_blocked_when_github_mode(self):
        # CNU_DATA_REPO 설정 시 *.cnu.ac.kr 요청은 즉시 ConnectError (78초 대기 X)
        import importlib
        os.environ["CNU_DATA_REPO"] = "https://raw.githubusercontent.com/x/y/main"
        try:
            import cnubot._net as net
            importlib.reload(net)
            self.assertTrue(net._GITHUB_MODE)
            import httpx
            import time as _t
            t0 = _t.time()
            with self.assertRaises(httpx.ConnectError):
                net.get("https://computer.cnu.ac.kr/computer/notice/bachelor.do")
            self.assertLess(_t.time() - t0, 1.0, "차단이 즉시여야 함(라이브 대기 없음)")
        finally:
            os.environ.pop("CNU_DATA_REPO", None)
            import cnubot._net as net
            importlib.reload(net)


if __name__ == "__main__":
    unittest.main(verbosity=2)
