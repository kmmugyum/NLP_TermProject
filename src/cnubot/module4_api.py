"""모듈 4-B: Orchestrator + FastAPI.

Orchestrator 는 순수 클래스(router/retriever/generator 주입) → 모델/서버 없이 단위 테스트.
FastAPI endpoint 는 app.state.orchestrator 로 thin-wrap. 실제 모델 로드는 lifespan 에서만.
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from pydantic import BaseModel

from .module2_router import CNUHybridIntentRouter
from .module3_retriever import AcademicRetriever, CafeteriaRetriever
import re as _re
import threading as _threading

# 진행상태 스트리밍용 thread-local 큐(handle_stream 스레드가 _plan을 worker thread에서 돌리고,
# helpers는 이 큐에 progress 메시지를 푸시 → 메인 제너레이터가 SSE status 이벤트로 전달).
_progress_local = _threading.local()

from .module4_generator import (
    FACILITY_MSG,
    NOTICE_MSG,
    REFUSAL_MSG,
    CNUGenerator,
    HFAnswerLLM,
)

# 도서관 시설 위치 질문(층별안내가 평면도 이미지뿐 → 텍스트 RAG 불가) → 도서관 시설 안내문.
# 다른 건물/편의시설은 텍스트 소스가 있을 수 있어 일괄 거부하지 않음(도서관에 한정).
_FACILITY_RE = _re.compile(
    r"화장실|엘리베이터|승강기|에스컬레이터|흡연|사물함|자판기|평면도|층별|몇\s?층|비상구")

# 모호/빈 질문 되물음(clarify) guard 패턴. 좁게 — 정상 질문은 절대 걸리지 않게.
CLARIFY_MSG = ("질문이 모호해요. 무엇에 대해(어떤 신청/일정/과목 등) 묻는지 "
               "조금만 더 구체적으로 알려주세요.")
# (a) 구두점/공백만으로 이뤄진 빈 입력(예: "?", "??", "...").
_CLARIFY_EMPTY_RE = _re.compile(r"^[?？.\s]+$")
# (b) 지시어-only 모호("그거 언제까지?", "이거 어디?") — 지시어로 시작하고 곧장 의문사.
#     한글 사이엔 \b가 경계로 안 잡혀 사용하지 않음(start-anchor로 충분히 좁음).
_CLARIFY_DEICTIC_RE = _re.compile(
    r"^(그거|그건|그게|이거|이게|저거|저건|그)\s*(언제|어디|얼마|뭐|어떻게|까지)")
# (b') 주어 없는 동작-only 모호("신청 언제부터?", "접수 어디?").
_CLARIFY_SUBJLESS_RE = _re.compile(
    r"^(신청|접수|등록)\s*(언제|어디|어떻게)")

# 대표→하위 도메인 라우팅: 코퍼스에 답 없으면 가장 관련된 CNU 사이트 URL을 안내.
# (키워드 튜플, 라벨, URL) — 위에서부터 우선 매칭.
_SITE_GUIDE = [
    (("도서관", "열람실", "대출", "평면도", "층별"), "중앙도서관", "https://library.cnu.ac.kr"),
    (("기숙사", "생활관", "사생", "입사", "생활관식당"), "생활관", "https://dorm.cnu.ac.kr"),
    (("입학", "입시", "수시", "정시", "모집요강", "전형"), "입학안내", "https://ipsi.cnu.ac.kr"),
    (("교환학생", "국제교류", "외국인", "어학연수", "파견"), "국제교류본부", "https://cnuint.cnu.ac.kr"),
    (("체육", "헬스", "수영", "운동장", "체력단련", "웰리스"), "체육시설", "https://gymn.cnu.ac.kr"),
    (("건물", "캠퍼스맵", "약도", "오시는길", "찾아오", "건물번호", "건물 위치"),
     "캠퍼스 안내도", "https://plus.cnu.ac.kr/html/kr/sub01/sub01_010804.html"),
    (("편의시설", "편의점", "은행", "우체국", "카페", "서점", "매점", "식당 위치"),
     "편의시설안내", "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050101.html"),
    (("증명", "제증명", "발급"), "증명발급(CNU포털)", "https://plus.cnu.ac.kr"),
    (("보건소", "보건진료소", "의무실", "건강검진", "건강관리"),
     "건강관리실", "https://health.cnu.ac.kr"),
    (("산학협력단", "산단", "산학"),
     "산학협력단", "https://iuc.cnu.ac.kr"),
    (("동아리", "학생회", "총학생회"),
     "총학생회/동아리", "http://cnustudent.cnu.ac.kr"),
    (("셔틀", "셔틀버스", "스쿨버스", "통학버스"),
     "셔틀버스 안내",
     "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050501.html"),
    (("ROTC", "rotc", "학군단", "학사장교"),
     "학군단", "https://116rotc.cnu.ac.kr"),
    (("평생교육원", "평교원", "평생교육"),
     "평생교육원", "https://lifelong.cnu.ac.kr"),
]
from .schemas import (
    ChatRequest,
    CNUBotResponse,
    IngestPayload,
    Intent,
    WeeklyMealCache,
)

_PKG = Path(__file__).resolve().parent
INDEX_PATH = str(_PKG / "storage" / "faiss_index.bin")            # 구 mock (webhook 재빌드용)
META_PATH = str(_PKG / "storage" / "faiss_index.bin.meta.json")
# 승격: 서빙은 실제 21,560 청크 학사 코퍼스 사용
ACADEMIC_INDEX_PATH = str(_PKG / "storage" / "academic_real.bin")
ACADEMIC_META_PATH = str(_PKG / "storage" / "academic_real.bin.meta.json")
MEAL_CACHE_PATH = str(_PKG / "data" / "cnu_meal_mock.json")
ACADEMIC_DATA_PATH = str(_PKG / "data" / "cnu_academic_mock.json")
FOODCOURT_PATH = str(_PKG / "data" / "cnu_1hak_foodcourt.json")


def ensure_academic_index() -> None:
    """Self-bootstrap: fresh git clone 에는 실제 벡터 .bin 이 없음(.gitignore).
    repo 루트의 `academic_v2_bin.zip` 을 풀어 `academic_real.bin`(+meta)을 생성한다.
    이미 있으면 no-op. 로컬(py3.10/torch2.5.1)·Colab(T4) 양쪽에서 동작.
    모든 진입점이 chat_model._build_orch → 이 함수를 거치므로 단일 부트스트랩 지점."""
    import zipfile
    import shutil

    idx = Path(ACADEMIC_INDEX_PATH)
    meta = Path(ACADEMIC_META_PATH)
    if idx.exists() and meta.exists():
        return

    storage = _PKG / "storage"
    storage.mkdir(parents=True, exist_ok=True)

    # 1) 인덱스 본체: zip 에서 academic_v2.bin 추출 → academic_real.bin
    if not idx.exists():
        seen: dict = {}
        # 1순위: repo 루트(_PKG.parent.parent)·cwd 인접 — 정상 배치면 여기서 즉시 발견
        for base in (_PKG.parent.parent, Path.cwd(), Path.cwd().parent):
            p = base / "academic_v2_bin.zip"
            if p.is_file():
                seen[str(p.resolve())] = p
        # 2순위(폴백): /content 하위 임의 위치. 1순위에서 찾았으면 건너뛴다.
        # ※ Path('/content').glob('**/...') 는 /content/drive(Google Drive, 네트워크 FUSE)까지
        #    재귀 스캔해 사실상 멈춤 → drive 경로는 제외하고, 이미 찾았으면 아예 실행 안 함.
        if not seen:
            try:
                for p in Path("/content").iterdir():
                    if not p.is_dir() or p.name == "drive":
                        continue
                    for q in p.glob("**/academic_v2_bin.zip"):
                        if q.is_file():
                            seen[str(q.resolve())] = q
                top = Path("/content") / "academic_v2_bin.zip"
                if top.is_file():
                    seen[str(top.resolve())] = top
            except (FileNotFoundError, PermissionError):
                pass
        cands = list(seen.values())
        if not cands:
            raise FileNotFoundError(
                "academic_v2_bin.zip 을 찾을 수 없습니다. repo 루트(또는 /content 하위)에 "
                "있어야 합니다 — fresh clone 자립 부트스트랩용."
            )
        with zipfile.ZipFile(cands[0]) as z:
            names = z.namelist()
            binname = next((n for n in names if n.endswith("academic_v2.bin")), names[0])
            with z.open(binname) as src, open(idx, "wb") as dst:
                shutil.copyfileobj(src, dst)

    # 2) meta: storage/academic_v2.bin.meta.json → academic_real.bin.meta.json
    if not meta.exists():
        v2meta = storage / "academic_v2.bin.meta.json"
        if not v2meta.is_file():
            raise FileNotFoundError(f"메타 파일 없음: {v2meta}")
        shutil.copy2(v2meta, meta)


# 단과대학 → 공식 사이트 매핑(별칭 포함). 새 단과대학 추가 시 여기에만 한 줄 추가하면 됨.
_COLLEGE_DOMAINS: tuple = (
    (("공과대학", "공대"), "https://eng.cnu.ac.kr"),
    (("인문대학", "인문대"), "https://human.cnu.ac.kr"),
    (("사회과학대학", "사회과학대", "사과대"), "https://socialscience.cnu.ac.kr"),
    (("자연과학대학", "자연대"), "https://cns.cnu.ac.kr"),
    (("경상대학", "경상대"), "https://cem.cnu.ac.kr"),
    (("농업생명과학대학", "농생대"), "https://cals.cnu.ac.kr"),
    (("약학대학", "약대"), "https://pharm.cnu.ac.kr"),
    (("의과대학", "의대"), "https://medicine.cnu.ac.kr"),
    (("생활과학대학", "생활대"), "https://homeco.cnu.ac.kr"),
    (("예술대학", "예대"), "https://art.cnu.ac.kr"),
    (("수의과대학", "수의대"), "https://vetmed.cnu.ac.kr"),
    (("사범대학", "사범대"), "https://edu.cnu.ac.kr"),
    (("간호대학", "간호대"), "https://nursing.cnu.ac.kr"),
    (("생명시스템과학대학", "생명시스템대"), "https://cbb.cnu.ac.kr"),
)
_COLLEGE_NOISE = {"국가안보융합학부", "자유전공학부", "대학/학부", "대학·학부", "입학과",
                  "충남대학교 입학과"}

# 학사 신호: 교과목/학년/학점/졸업/커리큘럼 등. 라우터가 OUT_OF_SCOPE/TEMPORAL_NOTICE로
# 오라우팅해도 이 신호가 있으면 학사 경로로 교정한다(회화체 '~은 몇 학년 과목이야?' 구제).
_ACADEMIC_SIGNAL_RE = _re.compile(
    r"몇\s*학년|학년\s*과목|교과목|교과과정|커리큘럼|이수학점|"
    r"졸업\s*요건|졸업\s*학점|선수과목|학점\s*인정")
# 실시간 공지 신호: 위 학사 신호와 함께 있으면 교정하지 않는다(진짜 공지 보존).
# '공지'는 '인공지능'의 '공지' 오매칭 방지(negative lookbehind).
_NOTICE_SIGNAL_RE = _re.compile(r"(?<!인)공지|게시|최근|새\s*글|소식|공고|알림")
# 셔틀/스쿨버스/통학버스: 표면형 나열 대신 광역 토큰으로 결정론 라우팅(시간표/노선/배차 등 모두).
_SHUTTLE_RE = _re.compile(r"셔틀|스쿨버스|통학버스")

# ── 라우팅 결함 4건(M1~M4) 가드용 좁은 패턴 ───────────────────────────────
# M3) 메타/도발: 봇의 진위·환각을 따지는 비판/도발. 학식/메뉴 단어가 섞여 cafeteria로
#     오라우팅돼 학식표를 덤프(도발에 표로 답하는 역효과)하는 것을 라우팅 전에 차단.
#     좁게 — 일반 질의('학식 메뉴 알려줘')엔 진위 도발 토큰이 없어 걸리지 않는다.
_META_PROVOKE_RE = _re.compile(
    r"지어내|가짜|거짓말|꾸며|허위|챗봇\s*(?:맞|지어|이지)|"
    r"진짜\s*(?:야|냐|임|인가|인지)|"
    r"믿을\s*수\s*(?:있|없)|신뢰할\s*수\s*(?:있|없)")
_META_PROVOKE_MSG = (
    "이 봇은 충남대학교 공식 자료(학사 요람·학칙·공식 홈페이지 공지, 학생식당 운영 정보 등)에 "
    "근거해 답변합니다. 임의로 정보를 만들어 내지 않으며, 자료에 없는 내용은 '확인되지 않는다'고 "
    "안내합니다. 궁금한 학사·시설·학식 정보를 구체적으로 물어봐 주세요.")

# M2) 도서관 운영시간: '도서관'+시간/개관/폐관 패턴은 위치('어디')가 아니라 운영시간 질의 →
#     _FACILITY_RE(화장실·층별)·_SITE_GUIDE(위치)에 안 잡혀 OOS로 오거부됐다. 공식 URL 안내.
_LIBRARY_HOURS_RE = _re.compile(
    r"몇\s?시|운영\s?시간|이용\s?시간|개관|폐관|문\s?닫|문\s?여|"
    r"주말.*(?:시간|시까지|운영)|평일.*(?:시간|시까지|운영)")

# M1) 멀티 intent: 연결어로 묶인 서로 다른 학사 신호가 2개+면 URL 빠른경로(단일 토픽 위치질의)를
#     건너뛰고 메인 academic RAG로 보내 한쪽만 답·드롭되는 일을 막는다.
# 연결어: 명시적 접속(그리고/이고/하고/또/,/및/이랑) + 구어 의문접속(뭐고/뭐냐/뭐지 …고).
# '고'는 단독으론 너무 흔해(보고/하고…) 의문사 직후('뭐고/뭐냐고')에서만 접속으로 인정.
_MULTI_CONNECTOR_RE = _re.compile(
    r"그리고|이고|하고|또|,|및|이랑|와\s|과\s|"
    r"뭐고|뭐냐|뭐지|뭔지|어떻고|어떤지|"
    r"(?:뭐|어디|언제|얼마|몇)\S*\s*고\s")
# 서로 다른 토픽 신호(겹치지 않는 카테고리). 2개 이상 매칭되면 멀티 intent로 판단.
_MULTI_SIGNAL_RES = (
    _re.compile(r"학식|메뉴|식단|점심|저녁|아침|조식|중식|석식"),          # 식사
    _re.compile(r"졸업|이수학점|학점|학년|교과목|커리큘럼|전공"),           # 학사 규정/과정
    _re.compile(r"휴학|복학|수강신청|재수강|복수전공|부전공|전과"),         # 학적/신청
    _re.compile(r"장학|등록금|학자금"),                                     # 등록/장학
    _re.compile(r"도서관|기숙사|생활관|셔틀|체육|운동"),                    # 시설
)

# M4a) 한영 혼합 오거부: 영문이 섞여도 핵심 한국어/충남대 맥락 학사어가 있으면 학사로.
#      명백 케이스만(졸업/학점/충남대 등) — 과탐 방지.
_KOR_ACADEMIC_HINT_RE = _re.compile(
    r"졸업|학점|학사|전공|휴학|수강신청|장학|등록금|충남대|CNU|cnu")
_EN_ACADEMIC_HINT_RE = _re.compile(
    r"graduat|credit|major|enroll|tuition|scholarship|semester|curriculum", _re.I)

# M4b) 감정표현 공감: '상담' 키워드 없이 와도 차가운 OOS 대신 짧은 공감 + 학생상담센터 안내.
_EMOTION_RE = _re.compile(r"우울|힘들|위로|외롭|지쳐|지친|스트레스|불안|괴로")
_COUNSEL_URL = "https://plus.cnu.ac.kr/html/hub/support/support_020203.html"
_EMOTION_MSG = (
    "많이 힘드셨겠어요. 혼자 감당하기 버거운 마음이 들 땐 전문가의 도움을 받는 것도 큰 힘이 됩니다. "
    "충남대학교 학생상담센터에서 재학생 누구나 무료로 심리·정서 상담을 받을 수 있어요:\n"
    f"{_COUNSEL_URL}")


def _multi_intent(query: str) -> bool:
    """연결어로 묶인 서로 다른 토픽 신호가 2개 이상이면 멀티 intent로 판단(M1)."""
    if not _MULTI_CONNECTOR_RE.search(query):
        return False
    hits = sum(1 for rx in _MULTI_SIGNAL_RES if rx.search(query))
    return hits >= 2

_GLUED_LATIN_RE = _re.compile(r"(?<=[가-힣])[A-Za-z]+(?=[가-힣])")
# 한자(CJK Unified, 호환·확장 일부) + 일본어 가나(히라가나·가타카나·반각 가나)
# + 키릴(U+0400~04FF)·아랍(U+0600~06FF) 강제 제거. LogitsProcessor가 한자/가나만 막아
# 누수되는 케이스(예: '디자чер'의 키릴)를 출력단에서 보정. 한글 U+AC00~D7A3은 보존,
# ASCII/IoT/SDN 등 라틴 약어도 영향 없음.
_DROP_CJK_RE = _re.compile(
    r"[\u0400-\u04FF\u0600-\u06FF"
    r"\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff"
    r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\uff65-\uff9f]+"
)

# 질의에 등장한 한국어 단어에 대해서만, 모델이 흘린 영어 대응어를 되돌리는 좁은 화이트리스트.
# (kor_in_query, english_regex(IGNORECASE), korean_replacement)
# 한국어 \b 이슈 회피: 라틴 양쪽에 라틴이 없을 때(=한글/공백/구두점/문장경계)에만 매칭.
_NL = r"(?<![A-Za-z])"   # 앞 비-라틴
_NR = r"(?![A-Za-z])"    # 뒤 비-라틴
_KOR_GLOSSARY: tuple = (
    ("연구실", _re.compile(_NL + r"research\s+lab(?:oratory|s)?" + _NR, _re.I), "연구실"),
    ("연구실", _re.compile(_NL + r"lab(?:oratory|s)?" + _NR, _re.I), "연구실"),
    ("연구실", _re.compile(_NL + r"research" + _NR, _re.I), "연구"),
    ("홈페이지", _re.compile(_NL + r"home\s*page" + _NR, _re.I), "홈페이지"),
    ("구성원", _re.compile(_NL + r"members?" + _NR, _re.I), "구성원"),
    ("이메일", _re.compile(_NL + r"e-?mails?" + _NR, _re.I), "이메일"),
    ("연락처", _re.compile(_NL + r"(?:contacts?|phone)" + _NR, _re.I), "연락처"),
)


def _strip_glued_latin(text: str, query: str = "") -> str:
    """한글-라틴-한글로 양쪽이 모두 한글에 글루드된 영어 토큰 제거 + 질의에 등장한 한국어
    단어에 한해 모델이 흘린 영어 대응어를 좁게 되돌림. URL·약어(IoT·SDN)는 보존."""
    if not text:
        return text
    # 글로서리 먼저: 모델이 흘린 'research lab' 등을 사용자 한국어 표현으로 복원.
    if query:
        for kor, eng_re, repl in _KOR_GLOSSARY:
            if kor in query:
                text = eng_re.sub(repl, text)
    # 한자·가나 강제 제거(LogitsProcessor 누수 보정).
    text = _DROP_CJK_RE.sub("", text)
    # 그 다음 마지막 그물: 양쪽-글루드 라틴 토큰만 제거.
    text = _GLUED_LATIN_RE.sub("", text)
    # 한글이 공백 없이 곧바로 붙어 있으면 단어 경계로 보고 한 칸 띄움(예: '그연구실' → '그 연구실').
    text = _re.sub(r"(?<=[가-힣])(?=연구실|홈페이지|구성원|이메일|연락처)", " ", text)
    return text


def _lcs_len(a: str, b: str) -> int:
    """두 문자열의 최장 공통 부분문자열 길이(학과명 매칭용)."""
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    best = 0
    for i in range(len(a)):
        ndp = [0] * (len(b) + 1)
        for j in range(len(b)):
            if a[i] == b[j]:
                ndp[j + 1] = dp[j] + 1
                if ndp[j + 1] > best:
                    best = ndp[j + 1]
        dp = ndp
    return best


def render_foodcourt(path: str) -> str:
    """제1학생회관 고정 푸드코트 JSON → 마크다운 텍스트(직접 서빙용)."""
    import json as _json

    d = _json.loads(Path(path).read_text(encoding="utf-8"))
    lines = [f"{d['place']} 푸드코트 메뉴 (고정, {d.get('weekend', '')})"]
    for cat in d["categories"]:
        items = ", ".join(f"{n} {p:,}원" for n, p in cat["items"])
        lines.append(f"\n■ {cat['name']} — {cat['hours']}\n{items}")
    return "\n".join(lines)


class Orchestrator:
    def __init__(
        self,
        router: CNUHybridIntentRouter,
        academic: AcademicRetriever,
        cafeteria: CafeteriaRetriever,
        generator: CNUGenerator,
        now_fn: Callable[[], datetime] = datetime.now,
        foodcourt_text: str | None = None,  # 제1학생회관 고정 푸드코트
        notice=None,  # NoticeService | None — 온디맨드 공지 RAG
    ):
        self.router = router
        self.academic = academic
        self.cafeteria = cafeteria
        self.generator = generator
        self.now_fn = now_fn
        self.foodcourt_text = foodcourt_text
        self.notice = notice

    def _progress(self, text: str) -> None:
        """현재 스레드에 큐가 묶여 있으면 진행상태 메시지를 푸시(없으면 noop)."""
        q = getattr(_progress_local, "q", None)
        if q is not None:
            try:
                q.put_nowait(text)
            except Exception:
                pass

    def _plan_notice(self, q: str, require_match: bool):
        """공지 답변 계획 → (prompt, references) 또는 None.
        require_match=True: 제목 매칭 있을 때만(OOS 오인된 행사명 질의 구제용)."""
        if self.notice is None:
            print("[notice-debug] NoticeService 미주입(self.notice=None) → 거절")
            return None
        try:
            label, items = self.notice.collect(q)
        except Exception as e:
            import traceback
            print(f"[notice-debug] collect() 예외 → {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
        print(f"[notice-debug] collect 결과: label={label!r}, items={len(items)}건")
        if not items:
            print("[notice-debug] items 비어있음 → 거절 (게시판 fetch 실패 또는 파싱 0건)")
            return None
        self._progress(f"공지 게시판 검색 중… ({label})")
        focuses = self.notice.top_title_matches(q, items, limit=3)
        if focuses:
            self._progress(f"관련 공지 {len(focuses)}건 본문 읽는 중…")
        if require_match and not focuses:
            return None
        from .module4_generator import build_notice_prompt
        from .schemas import Reference
        # 같은 행사에 묶인 공지가 여러 건이면 본문을 모두 합쳐 LLM에 제공(예: devday — 일정/장소/프로그램).
        parts = []
        for it in focuses:
            b = self.notice.fetch_body(it.url) if it.url else None
            if b:
                parts.append(f"=== {it.title} ===\n{b}")
        body = "\n\n".join(parts) if parts else None
        prompt = build_notice_prompt(q, items, label, body=body, now=_now())
        focus_set = {id(it) for it in focuses}
        refs = [Reference(title=it.title[:70], source_url=it.url)
                for it in focuses + [x for x in items if id(x) not in focus_set]][:5]
        return prompt, refs

    def _resolve_site(self, query: str) -> tuple[str, str] | None:
        """질의를 가장 관련된 CNU 하위 도메인/페이지로 라우팅 → (라벨, URL). 없으면 None."""
        for kws, label, url in _SITE_GUIDE:
            if any(k in query for k in kws):
                return label, url
        if self.notice is not None:
            # 다명칭 학과 별칭(레지스트리명과 달라 LCS가 어긋나는 경우). 예: 컴퓨터융합학부→computer
            for kws, host in ((("컴퓨터", "소프트웨어", "컴공", "인공지능", "에이아이"),
                               "computer.cnu.ac.kr"),):
                if any(k in query for k in kws):
                    d = next((x for x in self.notice.depts if x["host"] == host), None)
                    if d:
                        return d["name"], f"https://{host}"
        if self.notice is not None:  # 학과명: 질의와 학과명의 최장공통부분문자열(LCS) ≥3 최대 선택
            qn = query.replace(" ", "")
            best, best_len = None, 0
            for d in self.notice.depts:
                name = (d.get("name") or "").replace(" ", "")
                ov = _lcs_len(qn, name)
                if ov >= 3 and ov > best_len:
                    best, best_len = d, ov
            if best:
                return best["name"], f"https://{best['host']}"
        return None

    def _url_suffix(self, query: str, answer: str | None, is_fallback: bool) -> str | None:
        """코퍼스에 답 없으면(거부/미검색) 관련 CNU 페이지 URL 안내문 반환. 없으면 None."""
        not_found = is_fallback or (answer and _re.search(
            r"찾을 수 없|확인할 수 없|확인이 어렵|제공되지 않|포함되어 있지 않", answer))
        if not not_found:
            return None
        g = self._resolve_site(query)
        if g and g[1] not in (answer or ""):
            return f"\n\n정확한 정보는 '{g[0]}' 페이지에서 확인해 주세요: {g[1]}"
        return None

    def _plan(self, query: str) -> dict:
        """라우팅·검색 후 답변 계획 산출(handle/handle_stream 공유).
        키: intent, is_fallback, static(LLM 미사용 답변), prompt(LLM 입력), max_tokens, references, refined."""
        def P(intent, is_fallback=False, static=None, prompt=None, max_tokens=0,
              references=None, refined=None, static_prefix=None):
            return dict(intent=intent, is_fallback=is_fallback, static=static, prompt=prompt,
                        max_tokens=max_tokens, references=references or [], refined=refined,
                        static_prefix=static_prefix)

        # 모호/빈 질문 되물음 guard (라우팅 전): 환각·임의답 대신 구체화 요청.
        # 좁은 패턴만 — 정상 질문("오늘 학식 뭐야", "졸업 몇 학점")은 걸리지 않는다.
        _q = (query or "").strip()
        _qn = _q.replace(" ", "")
        if (len(_qn) <= 2 or _CLARIFY_EMPTY_RE.match(_q)
                or _CLARIFY_DEICTIC_RE.match(_q) or _CLARIFY_SUBJLESS_RE.match(_q)):
            return P(Intent.ACADEMIC, is_fallback=True, static=CLARIFY_MSG)

        # M3) 메타/도발 가드 (모든 라우팅·cafeteria override 이전):
        # '학식/메뉴' 단어가 섞여도 진위 도발이면 학식표 덤프 대신 출처를 차분히 설명.
        if _META_PROVOKE_RE.search(query):
            return P(Intent.OUT_OF_SCOPE, is_fallback=True, static=_META_PROVOKE_MSG)

        # 도서관 내부 시설(화장실·층별)은 평면도 이미지뿐 → 라우팅 전 전용 안내
        if "도서관" in query and _FACILITY_RE.search(query):
            return P(Intent.OUT_OF_SCOPE, is_fallback=True, static=FACILITY_MSG)
        # M2) 도서관 운영시간 질의는 위치/시설 가드에 안 잡혀 OOS로 오거부됐다 → 공식 URL 안내.
        if "도서관" in query and _LIBRARY_HOURS_RE.search(query):
            from .schemas import Reference
            lib_url = "https://library.cnu.ac.kr"
            return P(Intent.ACADEMIC, is_fallback=False, refined=query,
                     static=("충남대학교 중앙도서관의 개관·운영시간(평일·주말·시험기간 연장 등)은 "
                             f"열람실별로 다를 수 있어, 정확한 시간은 아래 공식 페이지에서 확인해 "
                             f"주세요:\n{lib_url}"),
                     references=[Reference(title="중앙도서관", source_url=lib_url)])
        # M4b) 감정표현 공감: '상담' 키워드 없이 와도 차가운 OOS 대신 짧은 공감 + 학생상담센터 안내.
        # 학사 신호가 함께면(예: '시험 스트레스 줄이려 휴학 가능?') 공감 가드를 건너뛰고 학사 처리.
        if _EMOTION_RE.search(query) and not _ACADEMIC_SIGNAL_RE.search(query):
            from .schemas import Reference
            return P(Intent.OUT_OF_SCOPE, is_fallback=False, refined=query,
                     static=_EMOTION_MSG,
                     references=[Reference(title="학생상담센터", source_url=_COUNSEL_URL)])

        # 주관적·외부정보 거절: 충남대 행정/학사 봇 범위 외
        # (평판·순위·주관 평가·교내 비교·근처 시설·통학거리·SNS 굿즈 등)
        _REJECT_PATTERNS = (
            # 주관적 의견·평가·비교
            ("평판", "순위", "랭킹",
             "맛있나요", "맛있어", "맛집",
             "좋나요", "좋아요", "어떤가요", "어떤지", "어떨까요",
             "쉽나요", "쉬워요", "어렵나요", "어려워요",
             "잘 주는", "잘주는", "꿀강의", "꿀교수", "꿀과목",
             "가장 인기", "가장 어려운", "가장 쉬운",
             "추천 전공", "추천 학과", "추천 교수",
             "학점 잘", "학점 짜게"),
            # 캠퍼스 외부·통학·근처
            ("대전역에서", "유성구청 거리", "부산에서", "서울에서",
             "KTX 가까운", "공항 거리", "통학 시간",
             "근처 카페", "근처 맛집", "근처 PC방", "근처 노래방",
             "근처 영화관", "근처 쇼핑몰", "근처 병원",
             "근처 약국", "근처 편의점", "근처 ATM",
             "자취방 시세", "원룸 시세", "고시원 가격"),
            # SNS·굿즈·기념품·홍보 외형
            ("SNS", "인스타 공식", "페이스북 공식",
             "유튜브 채널", "카카오톡 채널",
             "굿즈", "후드티", "학교 점퍼", "머그컵", "기념품",
             "공식 스토어", "마스코트", "응원가", "교가 mp3",
             "교가 가사", "교색 코드", "LCK", "LCK 시청"),
        )
        for group in _REJECT_PATTERNS:
            if any(k in query for k in group):
                return P(Intent.OUT_OF_SCOPE, is_fallback=True, static=(
                    "이 봇은 충남대학교 학사·행정·시설·공식 안내에 한정해 답변합니다. "
                    "주관적 평가, 캠퍼스 외부 정보, 비공식 콘텐츠는 다루지 않습니다. "
                    "관련해서 학교 공식 자료나 학사정보가 궁금하시면 다시 질문해 주세요."
                ))
        # M1) 멀티 intent(연결어 + 서로 다른 토픽 신호 2개+)면 URL 빠른경로(단일 토픽 위치질의)를
        # 건너뛰고 메인 academic RAG로 보내 한쪽만 답·드롭되는 일을 막는다.
        _is_multi = _multi_intent(query)
        # 시설/기관 위치 질의: 정확히 '시설명 + 어디·위치·주소·찾는' → _SITE_GUIDE 공식 URL.
        if not _is_multi and _re.search(r"어디|위치|주소|찾[\s가-힣]{0,3}", query):
            for kws, label, url in _SITE_GUIDE:
                if any(k in query for k in kws):
                    from .schemas import Reference
                    static = (f"충남대 {label}은(는) 다음 공식 페이지에서 확인하실 수 있습니다:\n"
                              f"{url}")
                    return P(Intent.ACADEMIC, static=static, refined=query,
                             references=[Reference(title=label, source_url=url)])
        # 인덱스 청크가 약하고 정보가 외부 사이트에 있는 카테고리(평생교육원·자유전공·ROTC 등)는
        # 키워드 매칭만으로 즉시 URL 안내(공지/엉뚱 학과 오라우팅 방지).
        _STANDALONE_SITES = (
            (("평생교육원", "평교원", "평생교육"), "평생교육원", "https://lifelong.cnu.ac.kr"),
            (("자유전공학부", "자유전공", "지식융합학부"),
             "자유전공학부 / 지식융합학부", "https://liberalarts.cnu.ac.kr"),
            (("ROTC", "rotc", "학군단", "학사장교"), "학군단", "https://116rotc.cnu.ac.kr"),
            # 외부 권위 사이트(라이브 fetch 본문 빈약 → URL 안내가 정답)
            (("국가장학금", "학자금 대출", "학자금대출",
              "한국장학재단", "장학재단", "취업후상환"),
             "한국장학재단", "https://www.kosaf.go.kr"),
            (("학점교류", "학점 교류", "학점상호인정", "타대학 수강", "타대학 학점"),
             "학점교류 안내(학사지원과)",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_0501.html"),
            (("공학교육인증", "ABEEK", "abeek", "공학인증"),
             "공학교육인증", "https://abeek.cnu.ac.kr"),
            (("교생실습", "교육실습", "교생", "교직과정"),
             "교직과정/교생실습 안내", "https://edu.cnu.ac.kr"),
            (("취업", "진로", "커리어", "취업지원",
              "모의 면접", "모의면접", "이력서 첨삭", "자기소개서 첨삭",
              "NCS", "ncs", "공기업 취업", "대기업 채용", "채용 설명회",
              "산업체 견학",
              "전공 적성 검사", "진로 검사", "MBTI", "진로 코칭",
              "진로 워크숍", "진로 스피치", "진로 도서", "진로 영상"),
             "취업지원본부", "https://career.cnu.ac.kr"),
            (("창업", "스타트업", "창업동아리", "창업지원"),
             "창업지원단", "https://withu.cnu.ac.kr"),
            (("공모전", "경진대회", "캠퍼스 라이프"),
             "충남대 학생활동 안내", "http://cnustudent.cnu.ac.kr"),
            (("LINC", "linc", "링크3.0", "LINC3.0", "LINC 3.0", "링크 사업단",
              "산학연협력 선도대학", "산학연 선도대학"),
             "LINC 3.0 사업단", "https://linc.cnu.ac.kr"),
            (("사회봉사 학점", "봉사 학점", "봉사학점", "사회봉사학점",
              "사회봉사 교과", "사회봉사교과", "봉사 시간 인정", "봉사활동 학점"),
             "사회봉사 (학생과)", "https://plus.cnu.ac.kr/html/hub/support/support_030205.html"),
            (("기숙사 비용", "기숙사 요금", "기숙사비", "기숙사 식비",
              "생활관 비용", "생활관 요금", "생활관비", "생활관 식비"),
             "학생생활관(비용)", "https://dorm.cnu.ac.kr"),
            (("모바일 학생증", "학생증 모바일", "Young Hana", "영하나", "모바일학생증"),
             "학생증발급안내 (모바일/실물)",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_050401.html"),
            (("튜터링", "튜터 프로그램", "학업UP", "상상 튜터링", "학습공동체",
              "또래학습", "learning community",
              "영어 글쓰기", "작문 클리닉", "글쓰기 센터", "학습 클리닉",
              "논문 작성법", "영어 회화",
              "비교과 트랙", "학생 성장 트랙",
              "학생 경력 관리", "학생 경력 인증",
              "학생 활동 인증", "학생 봉사 인증",
              "학생 발표 자료", "학생 포스터"),
             "학업UP 상상 튜터링·비교과 (교수학습지원센터)",
             "https://ile.cnu.ac.kr/ile/ctl/study03.do"),
            # 학사규정 영역 (청강·조기졸업·학번변경) — 학사요람에 일부 있으나 검색 약함
            (("청강", "청강 신청", "청강 가능"),
             "학사규정 안내 (학사지원과)",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05030201.html"),
            (("조기 졸업", "조기졸업"),
             "학사규정 안내 (조기졸업)",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05030201.html"),
            (("학번 변경", "학번변경", "학번 정정"),
             "학사지원과 안내",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05030201.html"),
            # 전문/특수 대학원
            (("로스쿨", "법학전문대학원", "법전원"),
             "법학전문대학원", "https://law.cnu.ac.kr"),
            (("의학전문대학원", "의전원", "의과대학 입학", "의예과 입학"),
             "의과대학", "https://medicine.cnu.ac.kr"),
            (("야간대학원", "직장인 대학원", "특수대학원", "대학원 진학",
              "대학원 일반", "대학원 일정", "대학원 학과", "대학원 폐지",
              "대학원 정원", "대학원 등록금", "대학원 사은금",
              "대학원 행정", "대학원 교수", "대학원 강좌",
              "대학원 시간표", "대학원 입학 일반"),
             "대학원 (일반/특수)", "https://grad.cnu.ac.kr"),
            # 한국어/외국어 교육
            (("한국어교육원", "한국어 교육원", "외국인 어학연수", "어학연수원",
              "TOPIK", "topik", "한국어 강좌"),
             "국제언어교육센터 (한국어교육)", "https://dream.cnu.ac.kr"),
            # 사이버캠퍼스/LMS
            (("e-class", "eclass", "사이버캠퍼스", "사이버 캠퍼스", "LMS", "lms",
              "이러닝", "온라인 강의실",
              "강의 동영상", "동영상 자막", "동영상 속도",
              "동영상 출석", "비대면 강의", "비대면 시험",
              "ZOOM", "zoom", "Webex", "webex", "Teams", "teams",
              "원격 강의", "원격수업", "녹화 강의"),
             "사이버캠퍼스(LMS)", "https://e-learn.cnu.ac.kr"),
            # 도서관 시설·DB·서비스
            (("도서관 그룹스터디", "그룹스터디룸", "스터디룸 예약",
              "도서관 멀티미디어", "멀티미디어실",
              "도서관 논문 신청", "논문 신청", "단행본 구입 신청", "단행본 신청",
              "RISS", "riss", "KISS DB", "DBpia", "dbpia",
              "Web of Science", "Scopus", "scopus",
              "IEEE Xplore", "ACM Digital", "Springer", "Elsevier",
              "ScienceDirect", "EZproxy", "ezproxy",
              "학외 접속", "학외 DB", "IP 인증", "도서관 학외",
              "노트북 대여", "우산 대여", "도서관 사물함",
              "도서관 1인 좌석", "도서관 카페", "도서관 학습공간",
              "분실 도서", "도서관 직원", "도서관 책 반납",
              "학과 추천 도서", "교과서 비치", "24시간 열람",
              "도서관 모바일 출입",
              "도서관 추천 도서", "도서관 신간", "도서관 외국어 도서",
              "도서관 음악실", "도서관 영상실", "도서관 전시실",
              "도서관 발표실", "도서관 토론실", "도서관 강의실",
              "도서관 강좌", "도서관 교육 프로그램",
              "정보 활용 교육", "논문 검색 교육",
              "인용 작성법", "도서관 어학실",
              "도서관 학과별 자료", "도서관 지정도서"),
             "중앙도서관", "https://library.cnu.ac.kr"),
            # 시설 (정심화·박물관)
            (("정심화국제문화회관", "정심화회관", "정심화", "백마홀"),
             "정심화국제문화회관", "https://jsh.cnu.ac.kr"),
            (("자연사박물관",),
             "자연사박물관", "https://nhm.cnu.ac.kr"),
            (("박물관", "무궁관"),
             "박물관", "https://museum.cnu.ac.kr"),
            # 학교 행사 (백마대동제·개교기념일·동상제·학생공연·기업탐방)
            (("백마축제", "백마대동제", "대동제", "개교기념일",
              "동상제", "학생 공연", "학생공연", "기업탐방", "기업 탐방"),
             "학생활동·행사 안내", "http://cnustudent.cnu.ac.kr"),
            # 분실물
            (("분실물", "잃어버린", "분실 신고"),
             "분실물광장 (학생민원)",
             "https://plus.cnu.ac.kr/_prog/lostandfound/?site_dvs_cd=kr&menu_dvs_cd=07080301"),
            # 학내 인쇄 (도서관 출력 서비스)
            (("학내 인쇄", "인쇄 서비스", "프린트 서비스", "출력 서비스"),
             "중앙도서관 (출력 서비스)", "https://library.cnu.ac.kr"),
            # 학생회관
            (("학생회관",),
             "편의·복지시설 안내",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050102.html"),
            # 동물병원 (수의대 부속)
            (("동물병원", "수의대 부속병원", "수의과대학 부속", "부속동물병원"),
             "수의과대학 부속동물병원", "https://cnuah.cnu.ac.kr"),
            # 인권센터 (성희롱·양성평등·인권상담·학교폭력·소수자 지원)
            (("인권센터", "양성평등", "성희롱", "성폭력 신고", "데이트 성폭력",
              "사이버 성폭력", "인권 상담", "학교폭력 신고",
              "LGBT", "lgbt", "성소수자",
              "다양성 지원", "소수자 지원"),
             "인권센터", "https://fovu.cnu.ac.kr"),
            # 종합상담센터·학생 정신건강 (우울·스트레스·위기·익명)
            (("종합상담센터", "학생상담센터",
              "우울증 상담", "우울 상담", "스트레스 관리", "스트레스 상담",
              "위기 상담", "익명 상담", "심리 상담"),
             "학생상담센터",
             "https://plus.cnu.ac.kr/html/hub/support/support_020203.html"),
            # 대학신문·방송국 (충대신문)
            (("대학신문", "충대신문", "교내 방송국", "교내방송국", "학보사"),
             "충대신문 (신문방송사)", "http://press.cnu.ac.kr"),
            # 농업생명과학대학 (실습장·식물원·농촌실습)
            (("농촌실습장", "농업실습", "식물원", "임업실습장", "농생대 실습",
              "농업과학"),
             "농업생명과학대학", "https://cals.cnu.ac.kr"),
            # 운동시설 (수영장·테니스·야구·농구·헬스·대운동장)
            (("수영장", "테니스장", "야구장", "농구장", "헬스장", "대운동장",
              "운동장 사용", "체육관 시설"),
             "체육시설", "https://gymn.cnu.ac.kr"),
            # 캠퍼스 지도·건물·행정처·강의실 위치
            (("캠퍼스 지도", "캠퍼스지도", "교내 지도", "본관 위치", "정문 위치",
              "후문 위치", "교내 건물 위치",
              "강의실 번호", "강의실 위치", "공학관 위치",
              "인문대 건물", "자연대 건물", "사회대 건물",
              "경상대 건물", "사범대 건물", "예술대 건물",
              "약대 건물", "간호대 건물", "수의대 건물",
              "의대 건물", "농생대 건물", "생활과학대 건물",
              "공학동", "공학관",
              "환경 공학동", "신소재공학동", "전자정보공학동",
              "기계공학동", "화학공학동", "토목환경공학동",
              "건축공학동", "산업공학동", "항공우주공학동", "컴퓨터공학동",
              "IT 정보화관", "학술정보관", "백마교양교육관",
              "인재개발원", "농업과학기술원", "의생명과학융합원",
              "융복합과학원 위치", "식품영양과학원",
              "시설처 위치", "인사처 위치", "학생처 위치",
              "기획처 위치", "재무처 위치", "교무처 위치",
              "입학처 위치", "연구처 위치", "산학협력단 위치",
              "평생교육원 위치", "보건진료소 위치",
              "우체국 위치", "약국 정확한 위치", "카페 정확한 위치"),
             "캠퍼스 안내 (지도)",
             "https://plus.cnu.ac.kr/html/kr/sub01/sub01_01080301.html"),
            # 캠퍼스 편의시설 (매점·자판기·ATM·약국·응급실·흡연구역)
            (("매점 운영시간", "매점 위치", "자판기 위치", "ATM 위치",
              "교내 ATM", "흡연구역", "응급실 위치", "인근 약국"),
             "편의시설 안내",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050101.html"),
            # 외국인 비자·교환학생 비자 (국제교류본부)
            (("D-2 비자", "D2 비자", "교환학생 비자", "외국인 비자",
              "외국인 D-2"),
             "국제교류본부 (비자)", "https://cnuint.cnu.ac.kr"),
            # 다양성 장학금 (다자녀·북한이탈·영주권자·다문화)
            (("다자녀 장학금", "북한이탈주민", "영주권자 학비",
              "다문화 장학금"),
             "학생과 장학금 안내",
             "https://plus.cnu.ac.kr/html/hub/support/support_030104.html"),
            # 학생증 혜택 (할인 제휴·도서 대출·IC·ATM·교통·페이)
            (("학생증 할인", "학생증 제휴", "학생증 도서 대출",
              "학생증으로", "학생증 사용처",
              "학생증 IC", "학생증 교통", "학생증 출입",
              "학생증 식당", "학생증 매점", "학생증 ATM",
              "학생증 충전", "학생증 잔액", "학생증 분실 정지"),
             "학생증발급안내",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_050401.html"),
            # 어학시험·한국사검정·TEPS·TOEIC·토익 인증 (학사지원과)
            (("어학시험 인증", "한국사능력검정시험", "한국사검정",
              "TEPS", "teps", "TOEIC", "toeic", "토익 점수", "어학 점수",
              "졸업 영어", "졸업 인증"),
             "학사지원과 (어학시험 인증)",
             "https://plus.cnu.ac.kr/html/hub/affairs/affairs_020304.html"),
            # 산학협력단 (연구비·기술이전·특허·학술연구·학부생연구·논문지원·연구소·연구원 채용)
            (("산학협력단", "기술이전", "특허출원", "특허 지원",
              "지식재산권", "TLO", "연구비 신청", "학내 연구과제",
              "학술연구진흥", "학부생 연구", "학생 학회 발표 지원",
              "국제학회 참가 지원", "학회 등록비 지원", "영문 논문 교정",
              "연구소 목록", "사회과학 연구소", "자연과학 연구소",
              "공학 연구소", "인문 연구소", "의학 연구소",
              "약학 연구소", "농업 연구소",
              "연구원 채용", "박사후연구원", "실험실 채용",
              "학생 연구실 인턴"),
             "산학협력단", "https://iuc.cnu.ac.kr"),
            # BK21 사업단 (융복합과학원)
            (("BK21", "bk21", "융복합과학원", "두뇌한국21",
              "교육연구단", "교육연구사업"),
             "BK21 융복합과학원", "https://cit-bk21.cnu.ac.kr"),
            # 학사 일정 (수강신청 정정·정정기간)
            (("수강신청 정정", "수강신청 변경 기간", "정정 기간",
              "수강 변경 기간"),
             "학사일정",
             "https://plus.cnu.ac.kr/_prog/academic_calendar/?site_dvs_cd=kr&menu_dvs_cd=05020101"),
            # 학생 활동·봉사·캠페인 (졸업작품전시·OT·헌혈·환경·자선)
            (("졸업 작품 전시", "졸업작품전시", "졸업전시",
              "신입생 오리엔테이션", "신입생 OT", "오리엔테이션",
              "학내 봉사활동", "학교 행사 봉사", "행사 운영 봉사",
              "자선 모금", "헌혈 캠페인", "환경 보호 활동",
              "친환경 캠페인", "학생 환경 동아리", "학생 봉사 단체"),
             "충남대 학생활동·행사 안내", "http://cnustudent.cnu.ac.kr"),
            # 학번 부여 (학사지원과)
            (("학번 부여", "학번부여", "학번 체계"),
             "학사지원과", "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05030201.html"),
            # 자기설계전공·융복합전공·전공 자율 설계
            (("자기설계전공", "자기 설계 전공", "전공 자율 설계",
              "융복합 전공", "융복합전공",
              "빅데이터 융합", "데이터사이언스 대학원", "ICT 융합",
              "의생명융합", "인공지능 융합", "디지털 인문학"),
             "자유전공학부 / 융복합 전공", "https://liberalarts.cnu.ac.kr"),
            # 글로벌 캠퍼스·국제 인재·해외 모집·자매학교
            (("글로벌 캠퍼스", "글로벌 인재", "해외 단기 연수",
              "글로벌 프로그램",
              "해외 인턴", "해외 봉사", "국제 교류 협정", "자매 학교",
              "외국인 교환학생", "외국인 박사", "외국인 대학원",
              "외국인 한국어 능력", "외국인 영어 능력",
              "외국인 입학원서", "외국인 학생 등록금",
              "외국인 학생 비율", "외국인 교수 비율", "영어 강의 비율"),
             "국제교류본부", "https://cnuint.cnu.ac.kr"),
            # AI정보화본부 (메일·와이파이·VPN·모바일앱·포털·학내망·IP)
            (("학교 메일", "학교메일", "메일 용량", "메일 비밀번호",
              "외부 메일 전달", "메일 외부 전송",
              "학내망", "학내 IP", "학내IP", "와이파이 비밀번호",
              "게스트 와이파이", "WiFi", "wifi",
              "모바일 앱 설치", "모바일 앱 로그인", "충남대 앱",
              "포털 비밀번호", "포털 비번", "통합정보시스템 로그인",
              "포털 로그인", "VPN", "vpn"),
             "AI정보화본부", "https://cic.cnu.ac.kr"),
            # 입학본부 (학과별 입학·입학정원·입시 전형 일반)
            (("입학정원", "입학 정원", "모집정원", "모집 정원",
              "의예과 입학", "치의학과 입학", "약학과 입학",
              "수의예과 입학", "간호학과 입학", "음악과 입학", "미술과 입학",
              "체육교육과 입학", "무용과 입학", "사범대학 입학",
              "공과대학 입학", "예술대학 입학",
              "수시 모집", "정시 모집", "학생부 종합", "학생부 교과",
              "논술 전형", "실기 전형", "면접 전형", "농어촌 전형",
              "기초생활수급자 전형", "지역인재 전형", "특성화고 전형",
              "만학도 전형", "군 위탁 전형", "재외국민 전형",
              "입학사정관", "입학 전형료", "전형료", "입학 합격자",
              "입학 등록 절차", "신입생 학과 배정", "신입생 등록금"),
             "입학본부", "https://ipsi.cnu.ac.kr"),
            # 학생 자치·위원회·총학생회 (cnustudent)
            (("학생회 선거", "총학생회 사이트", "단과대 학생회",
              "학과 학생회", "학생 자치", "학생복지위원회",
              "등록금심의위원회", "학사위원회", "교무위원회",
              "위원회 회의록", "대학평의원회", "교지편집위원회",
              "대학자치"),
             "학생자치·총학생회", "http://cnustudent.cnu.ac.kr"),
            # 학내 차량·주차 등록
            (("학내 차량", "차량 등록", "주차 등록", "주차증", "교내 주차"),
             "캠퍼스 편의시설 (주차)",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050101.html"),
            # 출입증
            (("출입증",),
             "AI정보화본부 (출입통제)", "https://cic.cnu.ac.kr"),
            # 보건진료소 (건강검진·백신·치과·외상센터·응급·의료비)
            (("학생 건강검진", "건강 검진", "건강검진", "백신 접종", "치과 진료",
              "외상센터", "응급 상황", "보건진료소",
              "학생 의료비", "응급실 비용", "의료비",
              "응급실 위치"),
             "보건진료소", "https://health.cnu.ac.kr"),
            # 안전관리본부 (실험실·방사선·화학물질·동물실험·연구안전·재난·소방)
            (("실험실 안전", "방사선 안전", "화학물질 관리", "동물실험 윤리",
              "연구안전", "안전교육",
              "학교 안전사고", "소화기 위치", "소방 훈련",
              "재난 대응", "재난 매뉴얼", "안전 매뉴얼"),
             "안전관리본부", "https://safety.cnu.ac.kr"),
            # 연구윤리·IRB·임상시험 (산학협력단)
            (("IRB", "irb", "임상시험 신청", "연구윤리 교육", "표절 예방"),
             "산학협력단 (연구윤리·IRB)", "https://iuc.cnu.ac.kr"),
            # 자전거·친환경·분리수거·폐기물 (편의시설·캠퍼스 운영)
            (("학내 자전거", "자전거 보관소", "자전거 등록",
              "친환경 캠퍼스", "그린 캠퍼스", "분리수거", "폐기물 처리"),
             "캠퍼스 편의·운영",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050101.html"),
            # 채용·인사 (조교·교수·직원·시설직·비정규직·휴직·승진·호봉·복지·자녀학비)
            (("조교 채용", "교수 채용", "직원 채용", "시설직 채용",
              "비정규직 처우", "교직원 채용",
              "교수 휴직", "교수 연구년", "교수 안식년",
              "교수 정년", "교수 승진", "교수 호봉", "교수 급여",
              "교수 복지", "교수 보험", "교수 자녀 학비",
              "직원 호봉", "직원 승진", "직원 휴직", "직원 복지",
              "직원 채용 시험", "사무직 시험", "기능직 시험",
              "시설관리직", "청소 용역", "경비 용역"),
             "충남대 채용·인사 안내",
             "https://plus.cnu.ac.kr/html/kr/sub01/sub01_010701.html"),
            # 등록금 위원회·인상률·인하 (학사지원과/등록과)
            (("등록금 인상률", "등록금 인하", "등록금 위원회",
              "등록금 심의"),
             "등록금 안내 (학사지원과)",
             "https://plus.cnu.ac.kr/html/hub/affairs/affairs_03.html"),
            # 규정·정관·회계규정·학사규정 조항·별표 (학사규정 안내)
            (("학칙 PDF", "학사 운영 규정 PDF", "도서관 규정 PDF",
              "인사 규정", "회계 규정", "정관 다운로드",
              "행정 규정 검색", "규정 발의", "규정 제정",
              "규정 변경 공고", "규정 검색",
              "학사규정 제", "학사규정 별표",
              "학칙 일반조항", "학칙 학적관리", "학칙 등록휴학",
              "학칙 수업", "학칙 졸업", "학칙 학위",
              "학칙 상벌", "학칙 학생자치", "학칙 부칙"),
             "학사규정 안내",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05030201.html"),
            # 조직도·인사·명단 (총장·부총장·본부장·처장·학장 등)
            (("총장 명단", "총장 인사말", "총장 비전",
              "부총장 명단", "본부장 명단",
              "단과대학장 명단", "대학원장 명단",
              "인사처 명단", "학생처 명단", "기획처 명단",
              "재무처 명단", "교무처 명단", "입학처 명단",
              "연구처 명단", "부속기관 명단",
              "평생교육원 명단", "산학협력단 명단",
              "도서관 명단", "박물관 명단"),
             "충남대 학사조직 (조직도)",
             "https://plus.cnu.ac.kr/html/kr/sub01/sub01_010601.html"),
            # 통계 자료 다운로드 (대학정보공시)
            (("학사 통계", "졸업생 진로 PDF", "졸업생 통계 다운로드",
              "대학정보공시", "학사 보고서",
              "학생 1인당", "교수 1인당",
              "1인당 도서", "1인당 면적", "1인당 비용",
              "1인당 학생 수", "1인당 강의 시수", "1인당 연구비",
              "1인당 논문", "1인당 특허", "1인당 장학금",
              "학생 수", "교수 수", "직원 수",
              "외국인 비율", "여학생 비율", "휴학생 비율",
              "졸업율", "졸업률", "중도탈락률", "중도 탈락",
              "신입생 충원율", "충원율",
              "평균 GPA", "학과별 경쟁률", "캠퍼스 크기"),
             "대학정보공시",
             "https://plus.cnu.ac.kr/html/kr/sub06/sub06_01.html"),
            # 대학혁신·국가/광역 사업단 (RISE·글로컬·DSC공유대학·미래기술 등)
            (("대학혁신 추진단", "RIS 사업", "RISE 사업",
              "대전세종충남 RISE", "글로컬대학", "글로컬 사업",
              "ICT 사업", "DSC 공유대학", "공유대학 강의",
              "광역 사업단", "미래모빌리티", "미래에너지",
              "첨단바이오", "반도체 사업단", "양자기술",
              "항공우주 사업단", "농수산식품", "스마트팜",
              "한국학 사업단", "인문사회 사업단"),
             "충남대 사업단·혁신본부",
             "https://plus.cnu.ac.kr/html/kr/sub01/sub01_010103.html"),
            # 환경·지속가능·ESG·사회적 책임
            (("신재생 에너지", "기후변화 연구", "ESG 활동",
              "지속가능 발전", "사회적 책임", "환경 연구",
              "통일 연구"),
             "충남대 사회적 책임·ESG",
             "https://plus.cnu.ac.kr/html/kr/sub01/sub01_010103.html"),
            # 다양성·평등 정책 (인권센터 broad)
            (("다양성 정책", "평등 정책", "학생 다양성",
              "캠퍼스 다양성"),
             "인권센터 (다양성·평등)", "https://fovu.cnu.ac.kr"),
            # 외국인 채용·해외 학기·글로벌 인턴
            (("외국인 교수 모집", "외국인 직원 모집",
              "글로벌 인턴십", "해외 학기 인정"),
             "국제교류본부", "https://cnuint.cnu.ac.kr"),
            # 영어 학위·강의·코칭·발표
            (("영어 강의 목록", "영어 학위 과정", "영어 학과 진학",
              "영어로 진행", "영어로 수업",
              "영어 코칭", "영어 멘토", "영어 발표 대회",
              "영어 글쓰기 대회", "영어 토론 대회",
              "학내 영어 동아리", "영어 능력 향상",
              "영어 교환학생 인증"),
             "교수학습지원센터 (영어 학습)",
             "https://ile.cnu.ac.kr/ile/ctl/study03.do"),
            # 강의평가·교수평가·만족도 (학사지원과)
            (("교수 평가", "강의 평가 결과", "강의 평가 점수",
              "교수 신뢰도", "교수 강의 후기",
              "학생 만족도", "학생 행복도",
              "학생 설문조사"),
             "강의평가 안내 (학사지원과)",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05030201.html"),
            # 정보 공개·회의록 공개·민원 (정보공개 청구)
            (("정보 공개 요청", "정보 공개 처리", "정보 공개 청구",
              "정보 공개 결과", "행정 정보 공개", "학사 정보 공개",
              "회의록 공개", "민원 답변", "민원 처리",
              "민원 위원", "학생 청원", "학생 민원"),
             "정보공개 (학사지원과)",
             "https://plus.cnu.ac.kr/html/hub/affairs/affairs_05.html"),
            # 공청회·학생 토론회·건의함 (총학생회)
            (("공청회", "학생 토론회", "학생 의견함",
              "학생 건의함", "학생 의견 수렴",
              "학사 부정 보고"),
             "학생자치·총학생회", "http://cnustudent.cnu.ac.kr"),
            # 위원회 명단·학생 위원 (학내 위원회)
            (("위원회 명단", "위원회 직무", "교무 위원회 명단",
              "평의원회 명단", "총학생회 명단", "학생자치단 명단",
              "학생회 회장 명단", "학생복지 위원", "학생 학사 위원",
              "학생 교무 위원", "학생 평의원", "학생 도서관 위원",
              "학생 식당 위원", "학생 기숙사 위원", "학생 동아리 위원",
              "학생 학과 위원", "학생 학년 대표", "학생 자치 회의",
              "학생 운영위"),
             "학생자치·위원회 (총학생회)",
             "http://cnustudent.cnu.ac.kr"),
            # 교내 근로·국가근로 (장학금 안내 — 이미 있으나 키워드 보강)
            (("교내 근로", "국가근로", "근로 장학", "교내 알바"),
             "학생과 장학금 안내",
             "https://plus.cnu.ac.kr/html/hub/support/support_030104.html"),
            # 학생 옴부즈·권익보호
            (("학생 옴부즈", "옴부즈", "권익 보호", "권익보호"),
             "인권센터 (권익 보호)", "https://fovu.cnu.ac.kr"),
            # 학생회비·동아리 예산/회계 (cnustudent)
            (("학생회비 환불", "학생회비 사용", "학생회비 감사",
              "동아리 예산", "동아리 보조금", "동아리 활동 보고서",
              "회계 보고", "예산 사용"),
             "총학생회 (회계·예산)", "http://cnustudent.cnu.ac.kr"),
            # 캠퍼스 명소·즐길거리·사이버 투어
            (("캠퍼스 잔디밭", "충대 호수", "도서관 분수대",
              "박물관 야외", "교내 정원", "단풍 명소",
              "캠퍼스 꽃길", "학내 사진 명소", "인스타 명소",
              "교내 핫플", "캠퍼스 즐길거리", "휴식 공간", "산책로"),
             "캠퍼스 사이버 투어",
             "https://plus.cnu.ac.kr/html/cyber/main.html"),
            # 식당 디테일 (메뉴·알레르기·비건·위생) — 생협/편의시설
            (("식당 알레르기", "비건 메뉴", "채식 메뉴",
              "식당 결제 방법", "외부인 식당", "식당 위생", "식중독",
              "식당 영업일", "단과대 학생식당"),
             "편의시설 (식당 운영)",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050101.html"),
            # 증명서 발급 (우편·자동발급기·인터넷·진위확인)
            (("증명서 우편", "증명서 자동 발급기", "증명서 모바일",
              "졸업증명서 우편", "증명서 해외 발송",
              "영문 진위확인", "진위확인 코드", "증명서 발급 수수료",
              "증명서 인터넷 발급"),
             "증명서 발급 (학사지원과)",
             "https://plus.cnu.ac.kr/html/hub/affairs/affairs_020303.html"),
            # 성명 변경 (영문·한글) — 학사지원과
            (("영문 성명 변경", "한글 성명 변경", "성명 변경",
              "이름 변경", "학적 영문 표기"),
             "학적부 (성명·표기)",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05030201.html"),
            # 교양 (영역·필수·선택·인공지능·캡스톤·영어·수학·진로)
            (("핵심 교양", "균형 교양", "인성교양", "자유 교양",
              "교양 필수", "교양 선택", "교양 영역",
              "인공지능 교양", "글쓰기 교양", "영어 교양", "수학 교양",
              "진로 교양", "캡스톤 교양", "일반 교양"),
             "교양교육원",
             "https://plus.cnu.ac.kr/html/kr/sub02/sub02_020207.html"),
            # 총동창회 (동문회·동문 멘토·동문 회비·동문 사이트)
            (("동문회", "총동창회", "동문 멘토", "동문 회비",
              "동문 사이트", "동문 행사", "동문 장학금",
              "동문 채용", "동문 사진첩", "동문 추천",
              "졸업생 평생회원", "동창회비"),
             "충남대 총동창회", "https://icnu.cnu.ac.kr"),
            # 부속학교 (충남대는 사대 부속학교 운영 안 함 — 정직 안내)
            (("사대부속", "사대부고", "사대부중", "사대부초",
              "사대부속유치원", "사범대 부속", "충남대 부속학교"),
             "부속학교 (운영 없음)",
             "https://plus.cnu.ac.kr/html/kr/sub02/sub02_0204.html"),
            # 자격시험·국가고시 지원 (취업지원본부)
            (("교사 임용", "행정고시", "의사 국가고시", "약사 국가고시",
              "변호사 시험", "회계사 시험", "세무사 시험",
              "PEET 시험", "LEET 시험", "법학 적성", "의학 적성",
              "치의학 적성", "약학 적성"),
             "취업지원본부 (국가고시·자격시험)",
             "https://career.cnu.ac.kr"),
            # 졸업생 진로 통계 (대학정보공시)
            (("졸업생 취업 통계", "졸업생 진로 통계", "졸업생 진학 통계",
              "학과별 취업률", "졸업생 만족도", "대학원 진학률"),
             "대학정보공시 (졸업생 통계)",
             "https://plus.cnu.ac.kr/html/kr/sub06/sub06_01.html"),
            # 입학 홍보·박람회·학교 소개 자료 (입학본부)
            (("입학 홍보", "입학 박람회", "학교 소개 자료",
              "학교 소개 책자", "캠퍼스 안내 책자", "홍보 영상"),
             "입학본부 (홍보)", "https://ipsi.cnu.ac.kr"),
            # 사회봉사 학점 인정 (이미 학생과 사회봉사 entry — kws 보강)
            (("사회 활동 학점", "봉사 학점 인정"),
             "사회봉사 (학생과)",
             "https://plus.cnu.ac.kr/html/hub/support/support_030205.html"),
            # 학사일정 디테일 (변경·캘린더·PDF·운영위 일정)
            (("학사 일정 변경", "학교 행사 캘린더", "학사 일정 PDF",
              "학사 일정 다운로드", "학사 운영 위원회 일정"),
             "학사일정",
             "https://plus.cnu.ac.kr/_prog/academic_calendar/?site_dvs_cd=kr&menu_dvs_cd=05020101"),
            # 학내 이동·대여 (카풀·킥보드·자전거·페이) — 캠퍼스 편의
            (("학내 카풀", "카쉐어링", "자전거 대여", "전동킥보드",
              "학내 페이코", "충남대 페이", "학내 충전", "학내 USB",
              "USB 충전"),
             "캠퍼스 편의·운영",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050101.html"),
            # 학습장비 대여 (스마트폰·태블릿·카메라·프로젝터·마이크·음향)
            (("스마트폰 대여", "태블릿 대여", "카메라 대여",
              "프로젝터 대여", "마이크 대여", "음향 장비",
              "행사 장비", "노트북 대여 장기"),
             "교수학습지원센터 (학습장비 대여)",
             "https://ile.cnu.ac.kr/ile/ctl/study03.do"),
            # 학내 인쇄 디테일 (컬러·무료·할인·매수·스캐너·팩스) — 도서관
            (("학내 컬러 프린터", "무료 인쇄", "인쇄 할인",
              "출력 매수 제한", "학내 스캐너", "학내 팩스",
              "복사소", "자동 인쇄"),
             "중앙도서관 (출력 서비스)", "https://library.cnu.ac.kr"),
            # 학내 무선/유선 인터넷 (cic 이미 — kws 보강)
            (("무료 와이파이 비번", "유선 인터넷"),
             "AI정보화본부", "https://cic.cnu.ac.kr"),
            # 학내 회의실·대강당·공연장 대관 — 정심화회관 (이미) + plus.cnu 편의시설
            (("회의실 예약", "강의실 임대", "공연장 대관",
              "대강당 예약", "세미나실 예약", "학생 라운지",
              "휴게실 위치"),
             "정심화국제문화회관 (대관)", "https://jsh.cnu.ac.kr"),
            # 학과 일반 (사무실·조교·시간표·교수진·전공 필수) — 학과 홈 안내
            (("학과 사무실", "학과 조교", "학과 행정실",
              "학과 사무실 전화", "학과 사무실 위치",
              "학과 대표 교수", "학과 학과장",
              "학과 행정 직원", "학과 학생회 위치",
              "학과별 전공 교수", "학과별 교수진",
              "학과별 졸업 학기", "학과별 4년 과정",
              "학과별 추천 진로", "학과별 졸업 후 진로",
              "학과별 부전공", "학과별 이중전공", "학과별 복수전공",
              "학과별 전공 필수", "학과별 전공 선택", "학과별 졸업 필수",
              "학과 시간표", "전공 시간표"),
             "학과/대학원 안내",
             "https://plus.cnu.ac.kr/html/kr/sub02/sub02_0201.html"),
            # 교통·오시는길 (시내버스·지하철·BRT·셔틀·택시·캠퍼스 위치)
            (("시내버스 노선", "시내버스 정류장",
              "셔틀버스 노선", "셔틀 정류장", "주말 셔틀", "야간 셔틀",
              "셔틀 첫차", "셔틀 막차", "셔틀 요금",
              "셔틀 이용", "셔틀 분실물",
              "학교까지 BRT", "학교까지 지하철", "학교까지 택시",
              "학교 오시는 길", "오시는 길",
              "어떻게 가나요", "대전 위치 정확", "대전 위치",
              "충남대 가는", "충남대 위치"),
             "오시는길 (교통 안내)",
             "https://plus.cnu.ac.kr/html/kr/sub01/sub01_01080301.html"),
            # 외국인 생활 (생활관·통역·은행·휴대폰·의료·문화체험) — 국제교류본부
            (("외국인 생활관", "외국인 통역", "외국인 한국어 멘토",
              "외국인 은행", "외국인 전화", "외국인 휴대폰",
              "외국인 의료", "외국인 단체",
              "외국인 한국 문화", "외국인 김치 체험", "외국인 한복",
              "외국인 학생회"),
             "국제교류본부 (외국인 생활)", "https://cnuint.cnu.ac.kr"),
            # 자격증 (사회복지사·보육교사·청소년지도사·정보처리기사·SQL·회계)
            (("사회복지사 자격", "보육교사 자격", "평생교육사 자격",
              "청소년 지도사", "정보처리기사", "SQL 자격",
              "빅데이터 자격", "회계 자격"),
             "취업지원본부 (자격증)", "https://career.cnu.ac.kr"),
            # 학사일정 디테일 (중간/기말/보강/휴강/공휴일/휴업일·학기별)
            (("중간고사 기간", "기말고사 기간", "보강 주간",
              "휴강일", "학교 공휴일", "학사 휴업일",
              "학기 시작일", "학기 종료일", "개교기념일 휴업",
              "1학기 시작", "2학기 시작", "1학기 종료", "2학기 종료",
              "1학기 중간고사", "2학기 중간고사",
              "1학기 기말고사", "2학기 기말고사",
              "1학기 등록금 납부", "2학기 등록금 납부",
              "1학기 수강신청", "2학기 수강신청",
              "1학기 복학", "2학기 복학", "1학기 휴학", "2학기 휴학",
              "여름 계절학기 일정", "겨울 계절학기 일정",
              "졸업식 1학기", "졸업식 2학기",
              "입학식 1학기", "입학식 2학기"),
             "학사일정",
             "https://plus.cnu.ac.kr/_prog/academic_calendar/?site_dvs_cd=kr&menu_dvs_cd=05020101"),
            # 학적 (자퇴·퇴학·재입학·학적 유지)
            (("자퇴 후 재입학", "퇴학 처리", "학교 떠난 후 학적",
              "휴학 후 학적 유지", "졸업 후 재학증명",
              "졸업 후 도서관"),
             "학사지원과 (학적)",
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05030201.html"),
        )
        # '학생회관'은 식사 맥락(학식/메뉴/끼니)이면 시설 안내가 아니라 학식 질의 →
        # _STANDALONE_SITES 매칭에서 제외하고 cafeteria 경로로 흘려보낸다.
        _meal_ctx = bool(_re.search(
            r"학식|메뉴|식단|점심|저녁|아침|조식|중식|석식|식사|먹", query))
        # M1) 멀티 intent면 단일 토픽 URL 빠른경로 전체를 건너뛰어 종합 답변(academic RAG)으로.
        for kws, label, url in (() if _is_multi else _STANDALONE_SITES):
            if "학생회관" in kws and _meal_ctx:
                continue  # 식사 맥락 학생회관 질의 → 시설 안내 스킵
            if any(k in query for k in kws):
                from .schemas import Reference
                static = (
                    f"충남대 {label} 관련 정보(소개·모집·강좌·신청 등)는 다음 공식 페이지에서 "
                    f"확인해 주세요:\n{url}\n\n자료에 세부 정보가 적재되어 있지 않아 통합 안내는 "
                    "위 사이트의 안내를 직접 확인하시는 게 정확합니다."
                )
                return P(Intent.ACADEMIC, static=static, refined=query,
                         references=[Reference(title=label, source_url=url)])
        # 동아리 관련 질의: 충남대는 동아리 통합 공식 페이지가 없으므로 가입 채널 정직 안내.
        if "동아리" in query and not _re.search(r"공지|소식|알림", query):
            from .schemas import Reference
            static = (
                "충남대학교는 일반 동아리를 통합 안내하는 공식 통합 페이지를 별도로 운영하지 "
                "않습니다(중앙·단과대·학과·자치 동아리가 개별 모집·운영). 가입·모집 정보는 "
                "아래 경로에서 직접 확인해 주세요:\n\n"
                "- 총학생회: https://cnustudent.cnu.ac.kr/cnustudent/index.do\n"
                "- 소속 단과대학 학생회: 해당 단과대학 홈페이지 → 학생회 메뉴\n"
                "- 창업동아리(공식 지원 사업): https://withu.cnu.ac.kr (학사지원시스템 → "
                "개인비교과프로그램에서 '창업동아리' 검색)"
            )
            return P(Intent.ACADEMIC, static=static, refined=query,
                     references=[
                         Reference(title="총학생회",
                                   source_url="https://cnustudent.cnu.ac.kr/cnustudent/index.do"),
                         Reference(title="학사지원시스템 (창업동아리)",
                                   source_url="https://withu.cnu.ac.kr")])
        # 단과대학 일반 안내(소개·연혁·학장·공지) 라이브 fetch.
        cov = self._read_college_overview(query)
        if cov:
            from .schemas import Reference
            body_text, page_url, label = cov
            prompt = (
                f"당신은 충남대학교 {label} 안내 봇입니다. 아래 [공식 홈페이지 본문]은 "
                "라이브로 가져온 본문입니다. 본문에 적힌 내용만 사용해 간결히 답하세요. "
                "본문에 없는 정보는 '본문에 명시되어 있지 않으니 공식 페이지를 확인하세요'라고 "
                "안내하세요. 답변은 한국어로만 작성하고, 키릴·아랍 등 외국문자는 한 글자도 "
                "섞지 마세요(정상적인 영문 고유명사·약어는 허용). 중복·모순 항목은 추측하거나 "
                "설명하지 말고 한 번만 깔끔히 정리하세요. '따라서', '이 외에도' 같은 군더더기 "
                "맺음·권유 문장은 쓰지 마세요.\n\n"
                f"[공식 홈페이지 본문]\n{body_text}\n\n[질문]\n{query}"
            )
            return P(Intent.ACADEMIC, max_tokens=500, refined=query, prompt=prompt,
                     references=[Reference(title=f"{label} (공식)", source_url=page_url)])
        # 외부 도메인(국제교류본부 등) 라이브 fetch — 학사요람에 없는 카테고리(외국인·교환학생·어학연수).
        ext = self._read_external_topic(query)
        if ext:
            from .schemas import Reference
            body_text, page_url, label = ext
            prompt = (
                f"당신은 충남대학교 {label} 안내 봇입니다. 아래 [공식 페이지 본문]은 "
                "라이브로 가져온 본문입니다.\n"
                "1) 질문에 관련된 내용만 간결히 정리해 안내하세요.\n"
                "2) 본문에 명시되지 않은 정책·금액·자격은 추측·창작 금지. "
                "구체적인 절차·요건은 '자세한 사항은 공식 페이지에서 확인하세요'로 안내.\n"
                "3) 답변은 한국어로만 작성. 본문의 영문 약어·고유명사는 그대로 사용.\n\n"
                f"[공식 페이지 본문]\n{body_text}\n\n[질문]\n{query}"
            )
            return P(Intent.ACADEMIC, max_tokens=500, refined=query, prompt=prompt,
                     references=[Reference(title=label, source_url=page_url)])
        # 학사일정/시험기간/방학 류 → plus.cnu 학사일정 캘린더 라이브(공지 오라우팅 방지).
        cal = self._read_academic_calendar(query)
        if cal:
            from .schemas import Reference
            page_text, page_url = cal
            prompt = (
                f"[오늘 날짜] {_now():%Y-%m-%d (%a)} — '며칠 남음/몇 주차/몇 달 남음' 등 기간 "
                "계산이 필요하면 반드시 이 날짜를 기준으로 계산하고, 자료에 없으면 추측하지 마세요.\n"
                "당신은 충남대학교 학사일정 안내 봇입니다. 아래 [학사일정 캘린더]는 plus.cnu의 "
                "공식 학사일정 페이지에서 가져온 본문입니다.\n"
                "1) 질문이 묻는 기간(예: 시험기간·방학·개강·종강·등록기간·수강신청 기간)에 "
                "해당하는 일정만 간결히 정리해 나열하세요. 자료에 없는 일정은 추측·창작 금지.\n"
                "2) 일정의 날짜와 명칭은 자료에 적힌 한국어 표기 그대로 사용. 한자/영어 변환 금지.\n"
                "3) '전체 학사일정 알려줘' 같은 일반 질의면 가장 가까운 학기 중심으로 주요 일정 "
                "5~10건만 추려 안내(전체를 그대로 옮기지 말 것).\n"
                "4) [인접 항목 결합 금지] 본문은 '06.22(월) 하기방학' '06.22(월)~07.10(금) 하기 "
                "계절학기' 처럼 **각 항목이 별개**로 나열됩니다. 같은 시작일을 공유하더라도 두 "
                "항목을 합쳐 새 기간을 만들지 마세요(예: '하기방학 6.22~7.10'은 환각). 본문에 "
                "**시작일만** 적혀 있으면 답변에도 시작일만 안내하고 종료일은 '본문에 명시되어 있지 "
                "않음'으로 밝히세요. 본문에 '기간 X~Y'로 적힌 항목은 그 항목의 명칭 그대로(예: "
                "'하기 계절학기 6.22~7.10') 인용하고, 다른 항목의 명칭으로 바꿔 말하지 마세요.\n\n"
                f"[학사일정 캘린더]\n{page_text}\n\n[질문]\n{query}"
            )
            return P(Intent.ACADEMIC, max_tokens=500, refined=query,
                     prompt=prompt,
                     references=[Reference(title="학사일정 캘린더 (plus.cnu)",
                                           source_url=page_url)])
        # 도서관 공지/소식 질의 → library.cnu 일반공지 게시판 라이브 fetch(NoticeService 우회).
        lib = self._read_library_notices(query)
        if lib:
            from .schemas import Reference
            body_text, page_url, items = lib
            prompt = (
                "당신은 충남대학교 도서관 공지 안내 봇입니다. 아래 [도서관 일반공지 목록]은 "
                "library.cnu.ac.kr 최신순입니다.\n"
                "1) '최근/최신' 류 질의면 맨 위 3~5건을 글머리표로 간결히 나열하세요. "
                "1건만 보여주지 마세요.\n"
                "2) 제목과 날짜는 자료의 한국어 표기 그대로 사용. 한자/일본어/영문 변환 금지.\n"
                "3) 자료에 없는 내용은 추측하거나 일반 지식으로 답하지 말고 "
                "'관련 공지를 찾을 수 없습니다'라고 답하세요. 자료의 정체를 '이건 도서관 공지입니다' "
                "처럼 메타로 설명하지 말고 바로 공지 목록만 제시.\n\n"
                f"[도서관 일반공지 목록]\n{body_text}\n\n[질문]\n{query}"
            )
            refs = [Reference(title=t, source_url=u) for t, _, u in items[:5]]
            refs.append(Reference(title="도서관 일반공지", source_url=page_url))
            return P(Intent.TEMPORAL_NOTICE, max_tokens=400, refined=query,
                     prompt=prompt, references=refs)
        # 교수/구성원 소개 질의 → 학과 구성원 페이지 라이브 fetch(라우터 전 가로챔; 공지로 오라우팅 방지)
        hist = self._read_dept_history(query)
        if hist:
            from .schemas import Reference
            page_text, page_url = hist
            prompt = (
                "당신은 충남대학교 학과 안내 봇입니다. 아래 [학과 연혁]은 해당 학과 공식 페이지에서 "
                "가져온 본문입니다. 본문에 직접 적혀 있는 사실(설립·신설·통합·개편 연도와 주체)만 "
                "사용해 답하세요. 본문에 없는 다른 학과 이력을 끌어와 추측하거나, 학과명을 변형·창작 "
                "하지 마세요. 본문에 답이 없으면 '해당 학과의 연혁 정보를 페이지에서 찾을 수 없습니다'"
                "라고만 답하세요.\n\n"
                f"[학과 연혁]\n{page_text}\n\n[질문]\n{query}")
            return P(Intent.ACADEMIC, max_tokens=400, refined=query, prompt=prompt,
                     references=[Reference(title="학과 연혁/소개", source_url=page_url)])
        col = self._read_college_structure(query)
        if col:
            from .schemas import Reference
            body_text, page_url = col
            prompt = (
                "당신은 충남대학교 단과대학 안내 봇입니다. 아래 [단과대학 조직]은 공식 사이트에서 "
                "가져온 최신 학부·학과 목록입니다. 질문이 '학부'를 묻는 경우 '[…학부]' 섹션의 항목만, "
                "'학과'를 묻는 경우 '[…학과]' 섹션의 항목만 정확히 나열하세요. 둘 다 묻거나 전체를 "
                "물으면 두 섹션 모두 안내. 다음 규칙을 반드시 지키세요:\n"
                "1) 자료의 학부·학과명을 한 글자도 변경·번역·한자화·축약하지 말고 글자 그대로 출력.\n"
                "2) 대괄호 라벨('[…학부]', '[…학과]', '(총 N개)')은 그대로 옮기지 말고, "
                "'OO대학에는 다음 N개 학부가 있습니다:' 같은 자연스러운 한국어 문장으로 풀어쓴 뒤 "
                "항목을 글머리표로 나열하세요.\n"
                "3) 자료에 없는 학부·학과명은 절대 추가·창작하지 마세요.\n\n"
                f"[단과대학 조직]\n{body_text}\n\n[질문]\n{query}")
            return P(Intent.ACADEMIC, max_tokens=400, refined=query, prompt=prompt,
                     references=[Reference(title="단과대학 학부·학과 목록",
                                           source_url=page_url)])
        fac = self._read_faculty(query)
        if fac:
            from .schemas import Reference
            page_text, page_url = fac
            prompt = (
                "당신은 충남대학교 학과 안내 봇입니다. 아래 [자료]에서 질문에 나온 교수의 "
                "정보(직위·전공/연구분야·연락처·연구실 위치·홈페이지·이메일 등)를 있는 그대로 간결히 "
                "정리해 답하세요. 자료에 '직위'·홈페이지·이메일·링크가 명시돼 있으면 그 값을 그대로 적고, "
                "자료에 없는 항목(특히 직위)은 절대 임의로 추정하지 말고 '자료에 표기되어 있지 않습니다'라고 "
                "밝히세요. 질문이 요구한 항목 중 일부가 자료에 없더라도(예: 연구실 학생 명단) 그 항목만 "
                "'페이지에 나와 있지 않습니다'라고 하고 나머지 확인되는 정보는 정상 제공하세요(부분적으로 "
                "없다고 전체를 거부하지 마세요). 답변은 한국어로만 작성하고, 질문에 쓰인 한국어 단어"
                "(예: '연구실', '홈페이지', '구성원', '연락처')는 영어로 번역·대체하지 말고 그대로 "
                "사용하세요. 자료에 영어로 표기된 학술 용어·약어(IoT·SDN 등)와 URL은 그대로 두세요. "
                "교수 이름 자체가 자료에 전혀 없을 때만 '해당 교수 정보를 찾을 수 없습니다'라고 답하세요.\n\n"
                f"[자료]\n{page_text}\n\n[질문]\n{query}")
            return P(Intent.ACADEMIC, max_tokens=512, refined=query, prompt=prompt,
                     references=[Reference(title="학과 구성원", source_url=page_url)])
        # 행사명/프로그램명류(영문 4+ 토큰: devday, hackathon, husscon 등)는 라우터가 academic으로
        # 오라우팅하는 경향 → 라우터 전에 공지 제목 매칭 시도(require_match=True). 매칭되면 그 공지로.
        # 단, M4a) 한영 혼합 학사질의(graduation/credit 등 + 충남대/학점 등)는 이 영문토큰 공지
        # 선점에 걸려 notice로 새던 버그가 있어, 이 경우 선점을 건너뛰고 아래 학사 라우팅(_kor_en_academic
        # 교정)으로 보낸다.
        _kor_en_academic = bool(
            _EN_ACADEMIC_HINT_RE.search(query) and _KOR_ACADEMIC_HINT_RE.search(query))
        if _re.search(r"[A-Za-z]{4,}", query) and not _kor_en_academic:
            pn_pre = self._plan_notice(query, require_match=True)
            if pn_pre:
                return P(Intent.TEMPORAL_NOTICE, prompt=pn_pre[0], max_tokens=400,
                         references=pn_pre[1], refined=query)
        # 셔틀/스쿨버스/통학버스: 5-way 분류의 정식 카테고리(라벨4)지만 라우터엔 독립 intent가
        # 없어 표면형('시간표' 등)이 조금만 달라도 OOS로 샜다. 광역 토큰으로 결정론 라우팅하여
        # 거부가 아닌 공식 안내(ACADEMIC)로 응답한다. 상세 배차표는 코퍼스에 없으므로 공식 페이지 안내.
        if _SHUTTLE_RE.search(query):
            from .schemas import Reference
            shuttle_url = "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050501.html"
            # is_fallback=False: 정상 안내(학사) 응답이므로 UI '거부' 배지가 붙으면 안 됨.
            return P(Intent.ACADEMIC, is_fallback=False, refined=query,
                     static=("충남대학교 셔틀버스(통학버스)의 노선·시간표·정류장 등 자세한 정보는 "
                             f"아래 공식 '셔틀버스 안내' 페이지에서 확인해 주세요:\n{shuttle_url}"),
                     references=[Reference(title="셔틀버스 안내", source_url=shuttle_url)])
        ir = self.router.get_intent(query)
        refined = ir.refined_query
        q = ir.refined_query or query
        # 학사 규정/정의 질의는 라우터가 OUT_OF_SCOPE/TEMPORAL_NOTICE로 오라우팅해도
        # 학사 경로로 강제 전환(공지·refusal 대신 학사 코퍼스 답변).
        from .module3_retriever import is_regulation_query
        _is_reg = is_regulation_query(query)
        # 학사 신호(몇 학년/교과목/커리큘럼 등)가 있고 공지 신호가 없으면 학사 질의로 본다.
        # 라우터가 OUT_OF_SCOPE/TEMPORAL_NOTICE로 오라우팅해도 양쪽 분기에서 동일하게 교정.
        # M4a) 한영 혼합 오거부 교정: 영문 학사어(graduation/credit 등)와 한국어/충남대 맥락
        # 학사어가 함께면 영문 때문에 OOS로 떨어지지 않도록 학사 질의로 본다(명백 케이스만).
        _kor_en_academic = bool(
            _EN_ACADEMIC_HINT_RE.search(query) and _KOR_ACADEMIC_HINT_RE.search(query))
        _looks_academic = _is_reg or _kor_en_academic or (
            _ACADEMIC_SIGNAL_RE.search(query) and not _NOTICE_SIGNAL_RE.search(query))
        if ir.intent == Intent.OUT_OF_SCOPE and not _looks_academic:
            pn = self._plan_notice(q, require_match=True)  # 행사명 등 공지 구제
            if pn:
                return P(Intent.TEMPORAL_NOTICE, prompt=pn[0], max_tokens=400,
                         references=pn[1], refined=refined)
            return P(Intent.OUT_OF_SCOPE, is_fallback=True, static=REFUSAL_MSG, refined=refined)
        if ir.intent == Intent.TEMPORAL_NOTICE and not _is_reg:
            # LLM이 '인공지능학부 미적분학 몇 학년' 같은 학사 질의를 공지로 오판해도,
            # 명시적 학사 신호가 있으면 학사로 교정. 공지 신호가 함께면 교정 안 함(진짜 공지 보존).
            if _ACADEMIC_SIGNAL_RE.search(query) and not _NOTICE_SIGNAL_RE.search(query):
                pass  # 학사 경로로 흘려보냄(아래 academic 처리)
            else:
                pn = self._plan_notice(q, require_match=False)
                if pn:
                    return P(Intent.TEMPORAL_NOTICE, prompt=pn[0], max_tokens=400,
                             references=pn[1], refined=refined)
                return P(Intent.TEMPORAL_NOTICE, is_fallback=True,
                         static=NOTICE_MSG, refined=refined)
        if ir.intent == Intent.CAFETERIA:
            is_1hall = ("1학" in query or "제1" in query)
            has_time = bool(_re.search(
                r"오늘|내일|모레|이번\s*주|이번주|어제|월요일|화요일|수요일|목요일|금요일|"
                r"토요일|일요일|점심|저녁|아침", query))
            # 시간 한정자 없는 '제1학생회관' 질의 → 정적 푸드코트 가격표(고정 메뉴 안내).
            if self.foodcourt_text and is_1hall and not has_time:
                return P(Intent.CAFETERIA, static=self.foodcourt_text, refined=refined)
            rr = self.cafeteria.retrieve(q, self.now_fn())
            if rr.is_fallback:
                # 캐시 미스 + '제1학생회관' 언급 → 푸드코트 정적표를 안내 폴백.
                if self.foodcourt_text and is_1hall:
                    return P(Intent.CAFETERIA, static=self.foodcourt_text, refined=refined)
                return P(Intent.CAFETERIA, is_fallback=True,
                         static=rr.fallback_message or REFUSAL_MSG, refined=refined)
            # 표는 코드(build_cafeteria_header)가 확정 출력 → LLM 환각('정보 없음')·
            # 토큰 잘림·멀티바이트 깨짐이 원천 불가능. 분석 질의일 때만 LLM 보조를 덧붙인다.
            from .module4_generator import (
                build_cafeteria_header, build_cafeteria_prompt, is_meal_analysis_query)
            table = build_cafeteria_header(rr.menus, rr.meal_date_label or "", query)
            if not is_meal_analysis_query(query):
                # 단순 조회('내일/오늘 메뉴') → 표만 static 반환(LLM 미사용 = 환각 불가).
                return P(Intent.CAFETERIA, static=table, refined=refined)
            # 분석 질의('가장 비싼' 등) → 표를 prefix 로 박고 LLM 한두 문장 분석을 prompt 로.
            return P(Intent.CAFETERIA, max_tokens=256, refined=refined,
                     static_prefix=table + "\n\n",
                     prompt=build_cafeteria_prompt(q, rr.menus, rr.meal_date_label or ""))
        # academic — 커리큘럼 질의는 학과 '학부 교과과정' 페이지를 우선 근거로(요람 대학원 청크 회피)
        from .module3_retriever import is_regulation_query
        is_reg = is_regulation_query(query)
        curric = self._read_curriculum(query)
        if curric and not is_reg:
            # 순수 커리큘럼 질의는 학부 교과과정 페이지만으로 충분.
            from .schemas import Reference
            page_text, page_url = curric
            prompt = (
                "당신은 충남대학교 학사 안내 봇입니다. 아래 [학부 교과과정]에 근거해 "
                "학년-학기별 주요 교과목을 정리해 답하세요. 표의 교과목명·학점을 그대로 쓰고 "
                "지어내지 마세요. 학부 과정이며 대학원(고급/특론/논문연구) 과목은 제외하세요.\n"
                "[중복·모순 정리] 같은 교과목번호나 과목명이 여러 번 나오면 한 번만 쓰세요. "
                "추측이나 메타발화(예: '~인 것 같습니다')는 하지 말고, 모순되는 정보는 생략하세요.\n"
                "[정렬] 1학년 1학기 → 1학년 2학기 → 2학년 1학기 → … 순으로 오름차순 정리하세요.\n"
                "[외국문자 금지] 키릴·아랍 등 외국문자를 쓰지 말고 올바른 한국어 글자로 쓰세요(예: '어드벤처디자인').\n"
                "[간결성] '이 외에도 학생들은…' 같은 군더더기 맺음 문단은 쓰지 마세요.\n\n"
                f"[학부 교과과정]\n{page_text}\n\n[질문]\n{query}")
            return P(Intent.ACADEMIC, max_tokens=512, refined=refined, prompt=prompt,
                     references=[Reference(title="학부 교과과정", source_url=page_url)])
        # 규정/요건 질의면 학사요람 boost된 청크 + 학부 교과과정 페이지를 모두 grounding으로 결합.
        self._progress("학사 인덱스 검색 중…")
        rr = self.academic.retrieve(q, top_k=8 if is_reg else None)
        from .schemas import Reference
        live = None if is_reg else self._read_dept_relevant(query)
        # DB(요람)에 청크가 없거나 약함 → 라이브 페이지로 즉시 폴백
        if rr.is_fallback or not rr.chunks:
            if live:
                ptext, purl = live
                prompt = (
                    "당신은 충남대학교 안내 봇입니다. 아래 [학과/서비스 페이지]에 근거해서만 답하세요. "
                    "페이지에 없으면 '관련 정보를 찾을 수 없습니다'라고 답하고 지어내지 마세요.\n"
                    "[중복·모순 정리] 같은 교과목번호나 과목명이 여러 번 나오면 한 번만 쓰세요. "
                    "추측이나 메타발화는 하지 말고, 모순되는 정보는 생략하세요.\n"
                    "[정렬] 학년-학기별 내용은 1학년 1학기 → 1학년 2학기 → … 순으로 오름차순 정리하세요.\n"
                    "[외국문자 금지] 키릴·아랍 등 외국문자를 쓰지 말고 올바른 한국어 글자로 쓰세요(예: '어드벤처디자인').\n"
                    "[간결성] '이 외에도 학생들은…' 같은 군더더기 맺음 문단은 쓰지 마세요.\n\n"
                    f"[학과/서비스 페이지]\n{ptext}\n\n[질문]\n{query}")
                return P(Intent.ACADEMIC, max_tokens=512, refined=refined, prompt=prompt,
                         references=[Reference(title="관련 페이지", source_url=purl)])
            return P(Intent.ACADEMIC, is_fallback=True, refined=refined,
                     static=rr.fallback_message or "관련 정보를 찾을 수 없습니다.")
        from .module4_generator import _dedup_refs, build_academic_prompt
        extra = self._read_top_pages(rr.chunks)  # adaptive 읽기(상위 출처 페이지)
        # 규정 질의 + curric: curric(한국어 학부 교과과정 페이지)만 grounding으로 사용.
        # 학사요람·영문 학과 청크가 섞이면 영문/대학원 과목/환각 위험 → 제외.
        # 핵심 사실(130학점)은 별도 [학칙 사실] 섹션으로 prompt에 직접 주입.
        if curric and is_reg:
            from .schemas import RetrievedChunk
            synth = RetrievedChunk(
                doc_id="curric-live",
                content=curric[0],
                title="학부 교과과정 (라이브)",
                source_url=curric[1],
                score=1.0,
            )
            rr_chunks = [synth]
            extra = (
                (extra + "\n\n" if extra else "")
                + "[학칙 사실 — 충남대 학사요람 제59조]\n"
                "충남대학교 일반 학사 졸업소요학점은 130학점을 원칙으로 한다. 사범대학·약학대학·"
                "의과대학·수의과대학·건축학 5년제 등 별도 학과만 예외(140·142·150·164·166·168·232 등). "
                "인공지능학과·컴퓨터인공지능학부 등 공과대학 일반 학부 산하 학과는 별도 예외가 없으므로 "
                "원칙(130학점)이 적용된다."
            )
        else:
            rr_chunks = list(rr.chunks)
        # 애매할 수 있으니 매칭된 학과/서비스 라이브 페이지도 컨텍스트에 합침(전체적으로 live 보강)
        if live:
            extra = ((extra + "\n\n") if extra else "") + f"[{live[1]}]\n{live[0]}"
        # 출처: PDF는 (PDF) 표기 + 클릭 가능한 학과/서비스 웹페이지 추가(요람 PDF는 6.7MB라 무거움)
        refs = [Reference(
            title=(r.title or "") + (" (PDF)" if (r.source_url or "").endswith(".pdf") else ""),
            source_url=r.source_url) for r in _dedup_refs(rr.chunks)]
        # 규정 질의면 학과/서비스의 클릭 가능한 라이브 페이지(졸업요건 게시판·학사규정)를 ref로 추가.
        # 본문 텍스트로는 빈약(첨부 zip)이라 grounding엔 안 쓰지만 사용자 클릭 경로로는 가장 권위.
        if is_reg:
            for url in self._regulation_live_links(query):
                if url and not any(url in (r.source_url or "") for r in refs):
                    refs.append(Reference(title="졸업요건 / 학사규정 (라이브)", source_url=url))
        # 클릭 가능한 라이브 페이지/학과 홈 출처 추가(중복 URL 제외)
        adds = []
        if curric:
            adds.append(("학부 교과과정", curric[1]))
        if live:
            adds.append(("학과/서비스 페이지", live[1]))
        site = self._resolve_site(query)
        if site:
            adds.append((site[0], site[1]))
        for ttl, url in adds:
            if url and not any(url in (r.source_url or "") for r in refs):
                refs.append(Reference(title=ttl, source_url=url))
        return P(Intent.ACADEMIC, max_tokens=512, refined=refined,
                 prompt=build_academic_prompt(q, rr_chunks, extra, now=_now()),
                 references=refs)

    def handle(self, query: str) -> CNUBotResponse:
        import time as _t
        _t0 = _t.time()
        p = self._plan(query)
        _t_plan = _t.time() - _t0
        print(f"[timing] _plan(검색/라이브fetch): {_t_plan:.1f}s "
              f"intent={p['intent'].value if hasattr(p['intent'],'value') else p['intent']} "
              f"prompt_len={len(p['prompt']) if p.get('prompt') else 0}")
        if p["prompt"]:
            _t1 = _t.time()
            answer = self.generator.llm.generate(p["prompt"], max_new_tokens=p["max_tokens"])
            print(f"[timing] LLM 생성: {_t.time()-_t1:.1f}s (max_tokens={p['max_tokens']})")
            answer = _strip_glued_latin(answer, query)
            # 분석 학식 질의: 코드가 만든 확정 표를 LLM 분석 앞에 prefix 로 결합.
            if p.get("static_prefix"):
                answer = p["static_prefix"] + (answer or "")
        else:
            answer = p["static"]
        # 명시적 거부문에는 사이트 URL을 덧붙이지 않는다('범위 밖' + 정답 URL 동시 출력 모순 방지).
        if answer != REFUSAL_MSG:
            suffix = self._url_suffix(query, answer, p["is_fallback"])
            if suffix:
                answer = (answer or "").rstrip() + suffix
        return CNUBotResponse(answer=answer, references=p["references"], intent=p["intent"],
                              is_fallback=p["is_fallback"], refined_query=p["refined"])

    def handle_stream(self, query: str):
        """SSE: status* → meta → delta* → (refs) → done.
        _plan은 워커 스레드에서 돌리고, 헬퍼들이 푸시한 progress 메시지를 큐에서 꺼내 status로 전송.
        plan 완료 후 LLM 토큰 스트림으로 본문 답변."""
        import queue as _queue
        q: _queue.Queue = _queue.Queue()
        plan_holder: dict = {}

        def _run_plan():
            _progress_local.q = q
            try:
                plan_holder["p"] = self._plan(query)
            except Exception as e:
                plan_holder["err"] = e
            finally:
                _progress_local.q = None
                q.put(None)  # sentinel

        t = _threading.Thread(target=_run_plan, daemon=True)
        t.start()
        # plan 진행 중 progress 메시지를 status 이벤트로 흘려보냄
        while True:
            msg = q.get()
            if msg is None:
                break
            yield {"type": "status", "text": msg}
        t.join()
        if "err" in plan_holder:
            yield {"type": "error", "message": f"{type(plan_holder['err']).__name__}: {plan_holder['err']}"}
            return
        p = plan_holder.get("p")
        if not p:
            yield {"type": "error", "message": "plan failed"}
            return
        yield {"type": "meta", "intent": p["intent"].value, "is_fallback": p["is_fallback"]}
        acc: list[str] = []
        if p["prompt"]:
            for tok in self.generator.llm.generate_stream(
                    p["prompt"], max_new_tokens=p["max_tokens"]):
                acc.append(tok)
                yield {"type": "delta", "text": tok}
        else:
            acc.append(p["static"] or "")
            yield {"type": "delta", "text": p["static"] or ""}
        suffix = self._url_suffix(query, "".join(acc), p["is_fallback"])
        if suffix:
            yield {"type": "delta", "text": suffix}
        if p["references"]:
            yield {"type": "refs", "references": [
                {"title": r.title, "source_url": r.source_url} for r in p["references"]]}
        yield {"type": "done"}

    def _read_curriculum(self, query: str):
        """학과 커리큘럼 질의면 그 학과 '학부 교과과정' 페이지를 라이브로 읽어 (본문, URL) 반환.
        요람의 대학원 청크 대신 학부 커리큘럼 페이지를 근거로 쓰기 위함."""
        if not _re.search(
                r"커리큘럼|교과\s*과정|교육\s*과정|교과목|무슨\s*과목|어떤\s*과목|"
                r"뭐\s*배|배우는|배워|배운다|수강|이수\s*체계|"
                r"졸업\s*요건|졸업\s*학점|이수\s*학점|이수\s*기준",
                query):
            return None
        site = self._resolve_site(query)
        if not site or "cnu.ac.kr" not in site[1]:
            return None
        self._progress("학부 교과과정 페이지 확인 중…")
        import httpx
        from bs4 import BeautifulSoup

        from .notice import fetch_page_text
        try:
            r = httpx.get(site[1], headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
                          verify=False, follow_redirects=True)
            soup = BeautifulSoup(r.text, "lxml")
            # P9: 라벨/경로 다양성 — 구체적 키워드 우선순위로 탐색(학부 우선, 대학원 제외)
            def _ok(h, t):
                return "grad" not in h and "school_of_edu" not in h and "대학원" not in t
            tiers = (
                lambda h, t: "curriculum" in h,
                lambda h, t: "교과과정" in t or "교육과정" in t,
                lambda h, t: any(k in h for k in ("process", "subject", "course", "gwajeong"))
                or any(k in t for k in ("교과목", "커리큘럼", "이수체계")),
            )
            cand = None
            for tier in tiers:
                for a in soup.find_all("a", href=True):
                    h = a.get("href").lower()
                    t = a.get_text(strip=True)
                    if tier(h, t) and _ok(h, t):
                        cand = str(httpx.URL(str(r.url)).join(a.get("href")))
                        break
                if cand:
                    break
            if not cand:
                return None
            txt = fetch_page_text(cand, max_chars=3500)
            # 실제 커리큘럼 텍스트인지 검증(학점 패턴). JS로 비어있거나 셸뿐이면 None → 요람 폴백.
            if not txt or len(_re.findall(r"\d-\d-\d|\d/\d/\d", txt)) < 5:
                return None
            return txt, cand
        except Exception:
            return None

    # 직위 키워드(긴 것 우선): 페이지 섹션/탭 라벨에서 직위를 귀속할 때 사용
    _POS_RE = _re.compile(
        r"(명예교수|초빙교수|특임교수|석좌교수|산학협력중점교수|산학협력교수|연구교수|"
        r"겸임교수|부교수|조교수|정교수|교수|강사|조교|교원)")
    _DETAIL_RE = _re.compile(r"mode=view|articleNo=|wr_id=|nttId=|seq=|key=", _re.I)
    _LAB_RE = _re.compile(
        r"sites\.google|github|\.io(/|$)|tistory|notion|wixsite|blog|cs-cnu|/~|lab\.", _re.I)
    _ENTITY_NAME_RE = _re.compile(r"([가-힣]{2,4})\s*[\(（]")

    def _entity_card(self, fsoup, base, query, *, section_re=None):
        """페이지에서 질의에 등장한 한국어 고유명사 항목을 포함한 '카드'(최소 컨테이너)를 찾고
        그 카드에 묶인 것들(상세링크·섹션 라벨·홈페이지/이메일 링크·본문)을 귀속해 반환.
        교수 소개 외에도 카드 단위 인물·시설·항목 페이지에 재사용 가능한 일반 헬퍼.
        Return: {name, detail_url, section_raw, section_match, links, body} | None.
        section_match는 section_re가 주어졌을 때 section_raw에서 매칭된 부분(예: '교수').
        """
        import httpx
        join = lambda h: str(httpx.URL(base).join(h))  # noqa: E731
        for blk in fsoup.find_all(["li", "tr", "div", "article"]):
            bt = " ".join(blk.get_text(" ").split())
            if not bt or len(bt) > 600:
                continue
            m = self._ENTITY_NAME_RE.match(bt)
            if not m or m.group(1) not in query:
                continue
            a = blk.find("a", href=self._DETAIL_RE)
            detail = join(a.get("href")) if a else None
            section_raw = None
            for sel in ("li.active", "li.on", ".tab .active", ".tabs .on", "li.first.active"):
                el = fsoup.select_one(sel)
                if el:
                    t = el.get_text(" ", strip=True)
                    if t and len(t) < 30:
                        section_raw = t
                        break
            if not section_raw:
                for prev in blk.find_all_previous(["h2", "h3", "h4", "strong", "th", "caption"]):
                    t = prev.get_text(" ", strip=True)
                    if t and len(t) < 40:
                        section_raw = t
                        break
            section_match = None
            if section_re and section_raw:
                mm = section_re.search(section_raw.replace(" ", ""))
                if mm:
                    section_match = mm.group(0)
            links, seen = [], set()
            for a in blk.find_all("a", href=True):
                h = a.get("href").strip()
                if not h or h in seen:
                    continue
                seen.add(h)
                if h.lower().startswith("mailto:"):
                    links.append(("이메일", h[7:]))
                elif self._LAB_RE.search(h):
                    links.append(("홈페이지", join(h) if h.startswith("/") else h))
            return {"name": m.group(1), "detail_url": detail,
                    "section_raw": section_raw, "section_match": section_match,
                    "links": links, "body": bt}
        return None

    def _read_college_overview(self, query: str):
        """단과대학 소개·연혁·개요 질의 → 해당 단과대학 홈 라이브 fetch.
        '구성·리스트' 질의는 _read_college_structure가 처리하므로 여기선 일반 안내 질의만."""
        target_name, target_url = None, None
        # _COLLEGE_DOMAINS 별칭 매칭 (긴 별칭 우선)
        all_aliases = [(alias, url) for aliases, url in _COLLEGE_DOMAINS for alias in aliases]
        all_aliases.sort(key=lambda x: -len(x[0]))
        for alias, url in all_aliases:
            if alias in query:
                target_name, target_url = alias, url
                break
        if not target_url:
            return None
        # 구성·리스트 질의는 _read_college_structure가 우선 → 여기선 회피
        if _re.search(r"학부|학과", query) and _re.search(
                r"리스트|목록|구성|뭐\s*있|어떤|모두|모든|얼마|몇|뭐가|뭐야", query):
            return None
        if not _re.search(
                r"소개|개요|연혁|학장|뭐\s*하|어떤\s*대학|어떤\s*학과|"
                r"공지|소식|취업|진로|입학",
                query):
            return None
        self._progress(f"{target_name} 홈페이지 본문 확인 중…")
        from .notice import fetch_page_text
        body = fetch_page_text(target_url, max_chars=3000)
        if not body:
            return None
        return body, target_url, target_name

    def _read_external_topic(self, query: str):
        """외부 도메인(국제교류·산학·건강·학부 등) 라이브 fetch로 grounding. 학사요람에 없는 카테고리.
        Return (body_text, page_url, label) or None. 라이브 fetch 본문이 충분히 풍부한 사이트만.
        """
        # (질의 패턴, 라이브 URL, 라벨)
        SOURCES = (
            (_re.compile(r"외국인\s*유학생|외국인\s*학생|국제\s*유학|inbound"),
             "https://cnuint.cnu.ac.kr", "국제교류본부 (외국인 유학생)"),
            (_re.compile(r"교환학생|파견학생|해외\s*파견|outbound"),
             "https://cnuint.cnu.ac.kr", "국제교류본부 (교환·파견학생)"),
            (_re.compile(r"어학\s*연수|국외\s*연수|해외\s*어학"),
             "https://cnuint.cnu.ac.kr", "국제교류본부 (어학연수)"),
            (_re.compile(r"산학협력|산단|산학|기업\s*협력|기술이전"),
             "https://iuc.cnu.ac.kr", "산학협력단"),
            (_re.compile(r"건강검진|보건진료|보건소|건강관리|의무실|예방접종"),
             "https://health.cnu.ac.kr", "건강관리실"),
            (_re.compile(r"자유전공학부|지식융합학부|자유전공"),
             "https://liberalarts.cnu.ac.kr", "자유전공학부 / 지식융합학부"),
            (_re.compile(r"국가안보융합|국가안보|국토안보"),
             "https://soins.cnu.ac.kr", "국가안보융합학부"),
            (_re.compile(r"국제학부|cnusis|International\s*School", _re.I),
             "https://cnusis.cnu.ac.kr", "국제학부"),
            (_re.compile(r"창의융합대학|창의융합|융복합대학"),
             "https://creative.cnu.ac.kr", "창의융합대학"),
            (_re.compile(r"장애학생|장애\s*지원|돔아|doumi", _re.I),
             "http://doumi.cnu.ac.kr", "장애학생지원센터"),
            (_re.compile(r"체육시설|수영장|헬스장|체력단련|웰니스|짐넌"),
             "https://gymn.cnu.ac.kr", "체육시설"),
            (_re.compile(r"박물관|전시|소장품|museum", _re.I),
             "https://museum.cnu.ac.kr", "충남대학교 박물관"),
            # SW 중심대학 사업단
            (_re.compile(r"SW\s*중심대학|소프트웨어\s*중심대학|SW\s*사업단|swuniv", _re.I),
             "http://swuniv.cnu.ac.kr", "SW중심대학사업단"),
            # plus.cnu 공식 안내 페이지 (학사요람 보충)
            (_re.compile(r"부전공\s*안내|부전공\s*이수\s*범위|부전공\s*신청\s*절차"),
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_051207.html", "부전공 안내(plus.cnu)"),
            (_re.compile(r"복수전공\s*안내|복수전공\s*이수\s*범위|복수전공\s*신청\s*절차"),
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_051206.html", "복수전공 안내(plus.cnu)"),
            (_re.compile(r"융복합창의전공\s*안내|융복합\s*창의\s*전공"),
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_051205.html",
             "융복합창의전공 안내(plus.cnu)"),
            (_re.compile(r"전공과정\s*이수\s*체계|이수\s*체계도"),
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_051204.html",
             "전공과정 이수체계도(plus.cnu)"),
            (_re.compile(r"학사규정\s*안내|학사규정\s*일반|학사\s*규정\s*전반"),
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05030201.html",
             "학사규정 안내(plus.cnu)"),
            (_re.compile(r"편의시설\s*안내|복지시설|학생\s*복지\s*시설"),
             "https://plus.cnu.ac.kr/html/kr/sub05/sub05_05050102.html",
             "편의·복지시설 안내(plus.cnu)"),
            # 충남대 일반 안내(발전·상징·UI)
            (_re.compile(r"학교\s*발전|대학\s*발전|발전\s*계획|비전|미래\s*전략"),
             "https://plus.cnu.ac.kr/html/kr/sub01/sub01_010103.html",
             "충남대 발전계획 (plus.cnu)"),
            (_re.compile(r"학교\s*상징|대학\s*상징|UI|교가|교색|교화|교목|마스코트"),
             "https://plus.cnu.ac.kr/html/kr/sub01/sub01_010104.html",
             "충남대 상징 (plus.cnu)"),
            # 충남대 연혁(연도별 사건) — 학교 역사 질의
            (_re.compile(r"충남대.{0,5}(연혁|역사)|대학.{0,5}연혁|언제.{0,5}(설립|개교)"),
             "https://plus.cnu.ac.kr/html/kr/sub01/sub01_0102.html",
             "충남대 연혁 (plus.cnu)"),
            # 학사·본부 조직도 — '행정 조직', '부서' 질의
            (_re.compile(r"학사\s*조직|본부\s*조직|행정\s*조직|조직도|부서\s*안내|"
                         r"부총장|총괄지원본부|교무처|학생처|기획처"),
             "https://plus.cnu.ac.kr/html/kr/sub01/sub01_010601.html",
             "충남대 학사조직 (plus.cnu)"),
            # AI정보화본부 — WIFI·IP·정보보호·원격지원·실습실 등 IT 인프라
            (_re.compile(r"AI정보화|정보화본부|WIFI|wifi|IP\s*신청|"
                         r"정보보호|원격\s*지원|실습실\s*점검|스마트\s*PC"),
             "https://cic.cnu.ac.kr", "AI정보화본부"),
            # 안전관리본부 — 연구·실험·실습 안전, 안전교육
            (_re.compile(r"연구\s*안전|실험\s*안전|실습\s*안전|안전교육|"
                         r"안전관리|안전관리시스템|연구실\s*안전"),
             "https://safety.cnu.ac.kr", "안전관리본부"),
            # HUSS 인문사회 디지털 융합 사업단
            (_re.compile(r"HUSS|huss|인문사회\s*디지털|디지털\s*융합전공|"
                         r"인문사회\s*융합", _re.I),
             "https://huss.cnu.ac.kr", "HUSS 인문사회 디지털 융합 사업단"),
            # 행정대학원
            (_re.compile(r"행정대학원|행정학과\s*대학원|공공행정\s*대학원|GSPA", _re.I),
             "https://gspa.cnu.ac.kr", "행정대학원"),
            # 인재개발원·취업지원 시스템
            (_re.compile(r"인재개발원|취업지원\s*센터|취업\s*정보|취업\s*시스템|"
                         r"채용\s*공고|취업\s*박람회"),
             "https://job.cnu.ac.kr", "인재개발원 취업지원"),
            # 미래창업원·창업지원
            (_re.compile(r"미래창업원|창업지원\s*센터|창업\s*아이디어|"
                         r"창업\s*경진|창업\s*경연|BOSS\s*특강"),
             "https://startup.cnu.ac.kr", "미래창업원"),
        )
        for pat, url, label in SOURCES:
            if not pat.search(query):
                continue
            self._progress(f"{label} 라이브 확인 중…")
            from .notice import fetch_page_text
            body = fetch_page_text(url, max_chars=3000)
            if not body:
                return None
            return body, url, label
        return None

    def _read_academic_calendar(self, query: str):
        """'학사일정/시험기간/방학/개강/등록기간/수강신청 기간' 류 질의 →
        plus.cnu 학사일정 캘린더 페이지 라이브 fetch. 공지 게시판으로 오라우팅 방지."""
        if not _re.search(
                r"학사\s*일정|학사\s*캘린더|학기\s*일정|학년도\s*일정|"
                r"시험\s*기간|중간\s*고사|기말\s*고사|방학|개강|종강|"
                r"등록\s*기간|등록금\s*납부\s*기간|"
                r"수강\s*신청|수강\s*변경|수강\s*취소|예비\s*수강|"
                r"성적\s*발표|학위\s*수여식|입학식|졸업식",
                query):
            return None
        self._progress("학사일정 캘린더 확인 중…")
        url = ("https://plus.cnu.ac.kr/_prog/academic_calendar/"
               "?site_dvs_cd=kr&menu_dvs_cd=05020101")
        # GitHub 모드(Colab): 라이브 fetch가 막히므로 사전 크롤된 academic_calendar.json 사용.
        # 이 JSON 은 (날짜, 이벤트명) 쌍을 보존해 '날짜만 나오고 명칭 없음' 문제를 해결한다.
        from . import github_data
        if github_data.is_enabled():
            cal = github_data.fetch_json("academic_calendar.json")
            if cal and cal.get("months"):
                lines = []
                for mm in cal["months"]:
                    lines.append(f"[{mm.get('month','')}]")
                    for it in mm.get("items", []):
                        d = it.get("date", ""); ev = it.get("event", "")
                        lines.append(f"- {d} {ev}".rstrip())
                return "\n".join(lines), url
            return None
        from .notice import fetch_page_text
        txt = fetch_page_text(url, max_chars=9000)  # 1~12월 전체 일정 커버
        if not txt:
            return None
        return txt, url

    def _read_library_notices(self, query: str):
        """'도서관 + 공지/소식' 질의 → library.cnu.ac.kr 일반공지 게시판 라이브 fetch.
        dept_registry에 도서관이 없어 NoticeService는 default(CS)로 흘러가 학부 공지를
        잘못 가져오는 문제를 우회. (제목 목록, 게시판 URL) 반환 — 매칭 없으면 None."""
        if not ("도서관" in query
                and _re.search(r"공지|소식|알림|새\s*글|최신|최근|올라온", query)):
            return None
        self._progress("도서관 공지 게시판 확인 중…")
        import httpx
        from bs4 import BeautifulSoup
        url = "https://library.cnu.ac.kr/bbs/list/1"  # 일반공지
        try:
            r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"},
                          timeout=8, verify=False, follow_redirects=True)
            soup = BeautifulSoup(r.text, "lxml")
            items: list[tuple[str, str, str]] = []  # (title, posted, href)
            seen = set()
            for a in soup.find_all("a", href=True):
                t = a.get_text(" ", strip=True)
                if not t or len(t) < 8 or len(t) > 140 or t in seen:
                    continue
                row = a.find_parent(["tr", "li", "div"])
                row_txt = " ".join(row.get_text(" ").split()) if row else ""
                m = _re.search(
                    r"20\d{2}[.\-]\d{1,2}[.\-]\d{1,2}|\d{4}-\d{2}-\d{2}", row_txt)
                if not m:
                    continue
                seen.add(t)
                full = str(httpx.URL(str(r.url)).join(a.get("href")))
                items.append((t[:120], m.group(0), full))
                if len(items) >= 8:
                    break
            if not items:
                return None
            body = "\n".join(f"- ({d}) {t}" for t, d, _ in items)
            return body, url, items
        except Exception:
            return None

    def _regulation_live_links(self, query: str) -> list[str]:
        """규정/요건 질의에 매칭되는 충남대 라이브 페이지 URL 목록(클릭 가능 ref용).
        ①질의 학과의 홈에서 '졸업요건/학사규정/학칙' 앵커를 라이브 탐색해 발견된 URL.
        ②질의가 학사규정/학칙/등록금/휴학 등 일반 정책이면 plus.cnu의 학사 정책 일반 페이지.
        충남대 도메인에 일반화: 학과별 도메인 패턴이 달라도 anchor 텍스트 기준으로 동작."""
        urls: list[str] = []
        site = self._resolve_site(query)
        if site and "cnu.ac.kr" in site[1]:
            try:
                import httpx
                from bs4 import BeautifulSoup
                r = httpx.get(site[1], headers={"User-Agent": "Mozilla/5.0"},
                              timeout=8, verify=False, follow_redirects=True)
                soup = BeautifulSoup(r.text, "lxml")
                # 앵커 텍스트가 졸업요건/학사규정/학칙 류이면 채택(짧은 메뉴 라벨만)
                wanted = ("졸업요건", "졸업 요건", "학사규정", "학사 규정",
                          "학칙", "학사안내", "학사 안내")
                for a in soup.find_all("a", href=True):
                    t = a.get_text(" ", strip=True)
                    if not t or len(t) > 20:
                        continue
                    if any(w in t for w in wanted):
                        u = str(httpx.URL(str(r.url)).join(a["href"]))
                        if u not in urls:
                            urls.append(u)
            except Exception:
                pass
        # 일반 학사 정책(plus.cnu 학사규정 메뉴 진입점)
        if _re.search(r"학사\s*규정|학칙|등록금|휴학|복학|전과|학적|장학", query):
            urls.append("https://plus.cnu.ac.kr/html/kr/sub05/sub05_0503.html")
        return urls[:3]

    def _read_dept_history(self, query: str):
        """학과 설립/연혁 질의 → 학과 사이트의 연혁·소개 페이지를 라이브로 읽어 (본문, URL).
        '스템프로봇' 같은 인접 학과 변형 환각 방지를 위해 그 학과 페이지 본문만 그라운딩."""
        if not _re.search(
                r"언제\s*(?:생긴|생겼|만들|만들어졌|설립|신설|개설|만든)|"
                r"연혁|역사|설립(?:\s*연도|일자|시기)|신설\s*(?:연도|시기)|개편\s*시기",
                query):
            return None
        if not _re.search(r"학과|학부|전공", query):
            return None
        site = self._resolve_site(query)
        if not site or "cnu.ac.kr" not in site[1]:
            return None
        self._progress(f"{site[0]} 연혁/소개 페이지 확인 중…")
        import httpx
        from bs4 import BeautifulSoup

        from .notice import fetch_page_text
        try:
            r = httpx.get(site[1], headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
                          verify=False, follow_redirects=True)
            soup = BeautifulSoup(r.text, "lxml")
            cand = None
            # 우선순위: 연혁 텍스트 > history href > 학과/학부 소개 텍스트
            tiers = (
                lambda h, t: "연혁" in t,
                lambda h, t: "history" in h.lower() or "yeonhyeok" in h.lower(),
                lambda h, t: ("학과소개" in t or "학부소개" in t or "전공소개" in t
                              or "introduction" in h.lower() or "intro" in h.lower()),
            )
            for tier in tiers:
                for a in soup.find_all("a", href=True):
                    h = a.get("href", "")
                    t = a.get_text(" ", strip=True)
                    if tier(h, t):
                        cand = str(httpx.URL(str(r.url)).join(h))
                        break
                if cand:
                    break
            if not cand:
                return None
            txt = fetch_page_text(cand, max_chars=3500)
            return (txt, cand) if txt else None
        except Exception:
            return None

    def _read_college_structure(self, query: str):
        """단과대학 학부/학과 구성 질의 → 해당 단과대학 공식 사이트 라이브 fetch.
        이름 suffix(…학부 / …학과)로 분류해 (본문, URL) 반환. _COLLEGE_DOMAINS 매핑 사용."""
        target_name, target_url = None, None
        # 더 긴 별칭이 먼저 매칭되도록 길이 내림차순으로 검사
        all_aliases = [(alias, url) for aliases, url in _COLLEGE_DOMAINS for alias in aliases]
        all_aliases.sort(key=lambda x: -len(x[0]))
        for alias, url in all_aliases:
            if alias in query:
                target_name, target_url = alias, url
                break
        if not target_url:
            return None
        if not (_re.search(r"학부|학과", query)
                and _re.search(r"리스트|목록|구성|뭐\s*있|어떤|모두|모든|얼마|몇|뭐가|뭐야",
                               query)):
            return None
        self._progress(f"{target_name} 사이트에서 학부·학과 목록 확인 중…")
        import httpx
        from bs4 import BeautifulSoup
        try:
            r = httpx.get(target_url, headers={"User-Agent": "Mozilla/5.0"},
                          timeout=8, verify=False, follow_redirects=True)
            s = BeautifulSoup(r.text, "lxml")
            seen_norm, schools, depts = set(), [], []
            for a in s.find_all("a", href=True):
                name = a.get_text(" ", strip=True)
                if not name or not (4 <= len(name) <= 25):
                    continue
                if name in _COLLEGE_NOISE or "대학원" in name:
                    continue
                norm = name.replace("·", "").replace(" ", "")
                if norm in seen_norm:
                    continue
                if name.endswith("학부"):
                    seen_norm.add(norm)
                    schools.append(name)
                elif name.endswith("학과") and name != "입학과":
                    seen_norm.add(norm)
                    depts.append(name)
            if not (schools or depts):
                return None
            body = (f"[{target_name} 학부] (총 {len(schools)}개)\n"
                    + ("\n".join(f"- {x}" for x in schools) if schools else "(없음)")
                    + f"\n\n[{target_name} 학과] (총 {len(depts)}개)\n"
                    + ("\n".join(f"- {x}" for x in depts) if depts else "(없음)"))
            return body, target_url
        except Exception:
            return None

    def _read_faculty(self, query: str):
        """교수/구성원 소개 질의면 학과 구성원 페이지를 라이브로 읽어 (본문, URL) 반환.
        특정 교수명이 질의에 있으면 그 교수의 상세 페이지를 우선 근거로, 직위는 섹션/탭에서 귀속."""
        if not (_re.search(r"교수|교원|구성원|선생님", query)
                and _re.search(r"소개|누구|어떤|정보|연구|알려|프로필|profile|어느|뭐", query)):
            return None
        self._progress("학과 구성원 페이지 확인 중…")
        site = self._resolve_site(query)
        # 학과 미지정이면 기본 학과(CS) 구성원에서 탐색(없으면 그라운딩이 '찾을 수 없음' 처리)
        host = site[1] if (site and "cnu.ac.kr" in site[1]) else "https://computer.cnu.ac.kr"
        import httpx
        from bs4 import BeautifulSoup

        from .notice import fetch_page_text
        try:
            r = httpx.get(host, headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
                          verify=False, follow_redirects=True)
            soup = BeautifulSoup(r.text, "lxml")
            cand = None
            for a in soup.find_all("a", href=True):
                h = a.get("href").lower()
                t = a.get_text(strip=True)
                if ((any(k in h for k in ("faculty", "professor", "member", "people"))
                     or "구성원" in t or "교수진" in t or ("교수" in t and len(t) < 8))
                        and "outmemer" not in h and "honor" not in h and "/en/" not in h):
                    cand = str(httpx.URL(str(r.url)).join(a.get("href")))
                    break
            if not cand:
                return None
            fr = httpx.get(cand, headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
                           verify=False, follow_redirects=True)
            fsoup = BeautifulSoup(fr.text, "lxml")
            base = str(fr.url)

            card = self._entity_card(fsoup, base, query, section_re=self._POS_RE)
            if card:
                body = (fetch_page_text(card["detail_url"], max_chars=3500)
                        if card["detail_url"] else None) or card["body"]
                parts = [f"[교수 상세 정보 — {card['name']}]"]
                if card["section_match"]:
                    parts.append(f"직위: {card['section_match']}")
                parts.append(body)
                if card["links"]:
                    parts.append("[링크]\n" + "\n".join(
                        f"- {k}: {v}" for k, v in card["links"]))
                return ("\n".join(parts), card["detail_url"] or cand)

            # 특정 교수 미지정(예: '교수진 알려줘') → 전체 목록 + 전체 링크 부록
            txt = fetch_page_text(cand, max_chars=3500)
            if not txt:
                return None
            seen_l, lines = set(), []
            for a in fsoup.find_all("a", href=True):
                h = a.get("href").strip()
                if h in seen_l:
                    continue
                is_mail = h.lower().startswith("mailto:")
                is_lab = bool(self._LAB_RE.search(h))
                if not (is_mail or is_lab):
                    continue
                par = a.find_parent(["li", "td", "div", "tr", "article"])
                name = " ".join(par.get_text(" ").split())[:30] if par else ""
                seen_l.add(h)
                lines.append(f"- {name} | {'이메일' if is_mail else '홈페이지'}: "
                             f"{h.replace('mailto:', '')}")
            if lines:
                txt += "\n\n[구성원 외부 링크(이름 | 종류: 주소)]\n" + "\n".join(lines[:40])
            return (txt, cand)
        except Exception:
            return None

    def _read_dept_relevant(self, query: str):
        """DB에 없거나 약할 때 일반 폴백: 학과 홈에서 질의 키워드와 맞는 링크를 찾아 라이브 fetch."""
        site = self._resolve_site(query)
        if not site or "cnu.ac.kr" not in site[1]:
            return None
        self._progress(f"{site[0]} 페이지에서 관련 정보 검색 중…")
        import httpx
        from bs4 import BeautifulSoup

        from .notice import fetch_page_text
        qtoks = [t.lower() for t in _re.findall(r"[A-Za-z0-9]{4,}|[가-힣]{3,}", query or "")
                 if t.lower() not in ("충남대", "충남대학교", "알려줘", "대해서", "어떻게", "무엇")]
        try:
            r = httpx.get(site[1], headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
                          verify=False, follow_redirects=True)
            soup = BeautifulSoup(r.text, "lxml")
            cand = None
            for a in soup.find_all("a", href=True):
                t = a.get_text(strip=True)
                if t and qtoks and any(tok in t for tok in qtoks):
                    cand = str(httpx.URL(str(r.url)).join(a.get("href")))
                    break
            url = cand or site[1]
            # 카드-귀속 우선: 질의에 한국어 고유명사가 있고 페이지에 그 이름의 카드가 있으면
            # 카드 단위로 묶인 정보(섹션 라벨·링크·상세본문)를 귀속해 사용.
            try:
                pr = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
                               verify=False, follow_redirects=True)
                psoup = BeautifulSoup(pr.text, "lxml")
                card = self._entity_card(psoup, str(pr.url), query)
                if card:
                    body = (fetch_page_text(card["detail_url"], max_chars=3500)
                            if card["detail_url"] else None) or card["body"]
                    parts = [f"[{card['name']} 카드]"]
                    if card["section_raw"]:
                        parts.append(f"섹션: {card['section_raw']}")
                    parts.append(body)
                    if card["links"]:
                        parts.append("[링크]\n" + "\n".join(
                            f"- {k}: {v}" for k, v in card["links"]))
                    return ("\n".join(parts), card["detail_url"] or url)
            except Exception:
                pass
            txt = fetch_page_text(url, max_chars=3500)
            return (txt, url) if txt else None
        except Exception:
            return None

    def _read_top_pages(self, chunks, limit: int = 2) -> str | None:
        """상위 출처 페이지(요람 PDF 제외, 서로 다른 URL)를 라이브로 읽어 본문 결합."""
        from .notice import fetch_page_text
        self._progress("관련 문서 페이지 읽는 중…")
        seen: set[str] = set()
        parts: list[str] = []
        for c in chunks:
            u = c.source_url or ""
            if not u.startswith("http") or u.lower().endswith(".pdf") or u in seen:
                continue
            seen.add(u)
            body = fetch_page_text(u)
            if body:
                parts.append(f"[{c.title or u}]\n{body}")
            if len(parts) >= limit:
                break
        return "\n\n".join(parts) if parts else None


def build_real_orchestrator() -> Orchestrator:
    """실제 모델 로드 (KURE@cuda:1, Qwen 4bit@cuda:0 + warmup). lifespan 에서만 호출."""
    from .module1_indexer import KUREEmbedder

    from .module2_hf_handler import HFIntentLLM

    embedder = KUREEmbedder("nlpai-lab/KURE-v1", "cuda:1")
    # 승격된 실 학사 인덱스(21,560 청크) 서빙. 없으면 mock 폴백.
    if Path(ACADEMIC_INDEX_PATH).exists():
        # 요람 PDF 적재(23,735) + canonical 부스팅(요람 경쟁서 본부 canonical 보호)
        academic = AcademicRetriever(ACADEMIC_INDEX_PATH, ACADEMIC_META_PATH,
                                     embedder=embedder, top_k=5, canonical_boost=0.04)
    else:
        academic = AcademicRetriever(INDEX_PATH, META_PATH, embedder=embedder, top_k=3)
    cafeteria = CafeteriaRetriever(cache_path=MEAL_CACHE_PATH)
    llm = HFAnswerLLM("Qwen/Qwen2.5-7B-Instruct", "cuda:0")  # M2/M4 공유 인스턴스
    _ = llm.generate("안녕")  # warm-up: 첫 쿼리 지연 완화 (제로화 아님)
    router = CNUHybridIntentRouter(llm=HFIntentLLM(backend=llm))  # 동일 Qwen 재사용
    foodcourt = render_foodcourt(FOODCOURT_PATH) if Path(FOODCOURT_PATH).exists() else None
    from .notice import NoticeService

    notice = NoticeService() if (_PKG / "data" / "dept_registry.json").exists() else None
    return Orchestrator(router, academic, cafeteria, CNUGenerator(llm),
                        foodcourt_text=foodcourt, notice=notice, now_fn=_now)


# ===== 게이트웨이: 세션 + 추론 워커(서브프로세스) 라이프사이클 =====
# 모델은 별도 워커 프로세스(inference_worker)에서 로드. connect 시 spawn,
# 마지막 disconnect/타임아웃 시 killpg → bitsandbytes 4bit VRAM까지 OS 레벨 완전 회수.
import os
import signal
import subprocess
import time

_WORKER_PORT = 8081
_WORKER_URL = f"http://127.0.0.1:{_WORKER_PORT}"
_SESSION_TTL = 90  # 초: 핑 끊긴 세션 만료 → 워커 kill

_sessions: dict[str, datetime] = {}
_sess_lock = threading.Lock()


_KST = timezone(timedelta(hours=9))


def _now() -> datetime:
    # 학식 '오늘/내일' 날짜 계산은 한국 시간 기준. Colab/서버가 UTC여도 KST로 고정
    # (UTC 사용 시 한국 새벽~오전엔 날짜가 하루 밀려 '내일'이 '오늘'로 나오던 버그).
    # naive 로 반환(tzinfo 제거): 캐시 timestamp 등 기존 naive datetime 과 빼기 호환.
    return datetime.now(_KST).replace(tzinfo=None)


class _WorkerManager:
    """추론 워커 서브프로세스 기동/종료. start=spawn+health대기, stop=killpg(VRAM 완전 회수)."""

    def __init__(self, port: int = _WORKER_PORT):
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, timeout: float = 45.0) -> bool:
        import httpx
        with self.lock:
            if self.is_running():
                return True
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self.proc = subprocess.Popen(
                ["python", "-m", "uvicorn", "cnubot.inference_worker:app",
                 "--host", "127.0.0.1", "--port", str(self.port)],
                env=env, preexec_fn=os.setsid)  # 독립 프로세스 그룹
            t0 = time.time()
            while time.time() - t0 < timeout:
                if self.proc.poll() is not None:  # 기동 실패
                    self.proc = None
                    return False
                try:
                    if httpx.get(f"http://127.0.0.1:{self.port}/health",
                                 timeout=1.0).status_code == 200:
                        return True
                except Exception:
                    pass
                time.sleep(1.0)
            return False

    def stop(self) -> bool:
        with self.lock:
            if self.proc is None:
                return False
            killed = False
            if self.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)  # 그룹째 → VRAM 회수
                    self.proc.wait(timeout=5)
                    killed = True
                except Exception:
                    pass
            self.proc = None
            return killed


_rm = _WorkerManager()


def _ensure_worker() -> None:
    if not _rm.is_running():
        _rm.start()


def _refresh_sessions() -> None:
    with _sess_lock:
        now = _now()
        for sid in list(_sessions):
            _sessions[sid] = now


def _touch(req) -> None:
    sid = getattr(req, "session_id", None)
    if sid:
        with _sess_lock:
            _sessions[sid] = _now()


def _cleanup_loop(stop: threading.Event) -> None:
    while not stop.wait(30):
        with _sess_lock:
            cutoff = _now() - timedelta(seconds=_SESSION_TTL)
            for sid in [s for s, t in _sessions.items() if t < cutoff]:
                del _sessions[sid]
            empty = not _sessions
        if empty and _rm.is_running():
            _rm.stop()  # 핑 끊긴 채 만료 → 워커 kill(VRAM 회수)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[AUTH] API_TOKEN = {API_TOKEN}  (X-API-Token 헤더로 전송)", flush=True)
    stop = threading.Event()
    threading.Thread(target=_cleanup_loop, args=(stop,), daemon=True).start()
    yield
    stop.set()
    _rm.stop()  # 게이트웨이 종료 시 워커도 정리


app = FastAPI(title="충남대 QA 게이트웨이", lifespan=lifespan)

# 웹앱을 다른 기기(노트북)에서 열어 호출할 수 있도록 CORS 허용(데모용 전체 허용).
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 토큰 인증: 공개 터널 노출 대비. /api/* 는 X-API-Token(또는 ?token=) 필수. /health는 공개.
# (미들웨어 대신 의존성 — BaseHTTPMiddleware가 SSE 스트림을 버퍼링하는 문제 회피)
import secrets as _secrets

from fastapi import Depends, Header, HTTPException

API_TOKEN = os.environ.get("CNU_API_TOKEN") or _secrets.token_urlsafe(16)


def _require_token(x_api_token: str | None = Header(default=None, alias="X-API-Token"),
                   token: str | None = None) -> None:
    if (x_api_token or token) != API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized: X-API-Token 필요")


class SessionReq(BaseModel):
    session_id: str
    timestamp: str | None = None


@app.get("/health")
def health() -> dict:
    _refresh_sessions()
    return {"ok": True, "ready": _rm.is_running(), "active_sessions": len(_sessions)}


@app.post("/api/v1/session/connect", dependencies=[Depends(_require_token)])
def session_connect(req: SessionReq) -> dict:
    with _sess_lock:
        _sessions[req.session_id] = _now()
    ready = _rm.start()  # 워커 spawn + 모델 로드(최초 ~30초)
    return {"ok": ready, "ready": ready, "session_id": req.session_id,
            "active_sessions": len(_sessions)}


@app.post("/api/v1/session/disconnect", dependencies=[Depends(_require_token)])
def session_disconnect(req: SessionReq) -> dict:
    with _sess_lock:
        _sessions.pop(req.session_id, None)
        empty = not _sessions
    unloaded = _rm.stop() if empty else False  # 마지막 세션 → 워커 kill(VRAM 회수)
    return {"ok": True, "active_sessions": len(_sessions), "unloaded": unloaded}


@app.post("/api/v1/cnu-bot/chat", response_model=CNUBotResponse, dependencies=[Depends(_require_token)])
async def chat(req: ChatRequest) -> CNUBotResponse:
    import httpx
    _ensure_worker()
    _touch(req)
    async with httpx.AsyncClient(timeout=180.0) as c:
        r = await c.post(f"{_WORKER_URL}/api/v1/cnu-bot/chat", json=req.model_dump())
    return CNUBotResponse(**r.json())


@app.post("/api/v1/cnu-bot/chat/stream", dependencies=[Depends(_require_token)])
async def chat_stream(req: ChatRequest):
    """워커의 SSE 스트림을 그대로 프록시(토큰 실시간 전달)."""
    import httpx
    from fastapi.responses import StreamingResponse

    _ensure_worker()
    _touch(req)

    async def gen():
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", f"{_WORKER_URL}/api/v1/cnu-bot/chat/stream",
                                 json=req.model_dump()) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/v1/webhook/ingest", dependencies=[Depends(_require_token)])
async def ingest(p: IngestPayload) -> dict:
    import httpx
    _ensure_worker()
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(f"{_WORKER_URL}/api/v1/webhook/ingest", json=p.model_dump())
    return r.json()
