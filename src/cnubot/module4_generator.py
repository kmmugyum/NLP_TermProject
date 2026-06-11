"""모듈 4-A: Generator. RetrievalResult → CNUBotResponse.

- is_fallback=True → LLM 추론 우회, fallback_message 그대로 반환 (할루시네이션 차단).
- 학사: chunks 출처 그라운딩 프롬프트 → references 채움.
- 학식: menus 마크다운 표 프롬프트.
AnswerLLM 은 주입식(HF/vLLM 무관). HFAnswerLLM 은 GPU0 4bit 구현체.
"""
from __future__ import annotations

import os
import re
import threading
from datetime import datetime
from typing import Protocol

_CJK_RE = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff65-\uff9f]")  # 한자+일본어 가나 차단(드리프트). 한글 U+AC00~ 제외

from .schemas import (
    CNUBotResponse,
    DailyMenu,
    Intent,
    NoticeItem,
    Reference,
    RetrievalResult,
    RetrievedChunk,
)

# 응답 끝에 누수되는 7B 환각 메타태그(예: "관리[SystemMessage]: ...") 컷.
# 끝부분(trailing)만 제거. 정상 본문에 흔한 "시스템 공학"·"시스템 프로그래밍" 같은
# 한국어 어구를 깨뜨리지 않도록, (1) 대괄호로 감싼 태그 [System...] 형태,
# (2) 영문 역할명 뒤에 콜론이 붙은 "System:"/"Assistant:" 형태에만 한정한다.
_META_TAIL_BRACKET_RE = re.compile(
    r"\s*\(?\s*(관리|시스템)?\s*\[(System|SystemMessage|Assistant|User|시스템)[^\]]*\]\s*:?.*$",
    flags=re.S,
)
_META_TAIL_COLON_RE = re.compile(
    r"\s*\(?\s*(관리|시스템)?\s*(SystemMessage|System|Assistant|User)\s*:.*$",
    flags=re.S,
)


def _strip_meta_tail(text: str) -> str:
    """디코딩된 응답 끝의 메타태그(SystemMessage/Assistant 등) 누수를 잘라낸다."""
    text = _META_TAIL_BRACKET_RE.sub("", text)
    text = _META_TAIL_COLON_RE.sub("", text)
    text = text.rstrip()
    # 공지를 나열(글머리표)한 뒤 끝에 '관련 공지를 찾을 수 없습니다'를 덧붙이는 7B 모순 꼬리 제거.
    lines = text.split("\n")
    has_list = any(re.match(r"\s*[-•*]\s|\s*\d+[.)]\s", ln) for ln in lines)
    if has_list:
        while lines and (not lines[-1].strip()
                         or re.fullmatch(r"\s*관련\s*공지를\s*찾을\s*수\s*없습니다[.\s]*",
                                         lines[-1])):
            if not lines[-1].strip():
                lines.pop()
            else:
                lines.pop()
                break
        text = "\n".join(lines).rstrip()
    return text


REFUSAL_MSG = "충남대학교 학사/학식 관련 질문에만 답변할 수 있습니다."
NOTICE_MSG = ("최신 공지사항·학사일정 등 실시간 정보는 저장된 자료로 답변드리기 어렵습니다. "
              "충남대학교 홈페이지(plus.cnu.ac.kr) 공지사항/학사일정에서 확인해 주세요.")
FACILITY_MSG = ("도서관 내부 시설의 세부 위치(화장실·층별 배치 등)는 평면도 이미지로 "
                "제공되어 텍스트로 안내드리기 어렵습니다. 도서관 홈페이지(library.cnu.ac.kr)의 "
                "층별안내를 확인해 주세요.")


class AnswerLLM(Protocol):
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str: ...


_CUMULATIVE_CREDITS_RE = re.compile(
    r"수료학년\s*(?:졸업소요\s*학점)?\s*(?:제\d학년\s*){2,}"
    r"(?:\d+학점\s*){2,}"
)


def _scrub_cumulative_table(content: str) -> str:
    """제59조 ⑤항 누적 수료학점 표(제1학년 33학점, …, 제4학년 130학점) 패턴을 마스킹.
    이 표가 LLM에 그대로 노출되면 학년·학기별 이수 분배로 잘못 해석돼 환각 유발."""
    return _CUMULATIVE_CREDITS_RE.sub("[누적 수료학점 표 — 답변에 옮기지 말 것]", content)


_KOR_WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")


def _date_line(now: datetime | None) -> str:
    """now가 있으면 프롬프트 맨 앞에 주입할 '오늘 날짜' 한 줄(날짜·기간 계산 grounding).
    요일은 한국어로 미리 계산해 명시한다(영문 %a 를 모델이 잘못 번역/환각하는 문제 방지)."""
    if now is None:
        return ""
    dow = _KOR_WEEKDAYS[now.weekday()]
    return (
        f"[오늘 날짜] {now:%Y년 %m월 %d일} {dow}요일 — 요일은 여기 적힌 '{dow}요일'을 그대로 쓰고 "
        "다시 계산하지 마세요. 날짜·기간(며칠/몇 주차/몇 달 남음 등) 계산이 필요하면 반드시 이 "
        "날짜를 기준으로 계산하고, 모르면 추측하지 마세요.\n"
    )


def build_academic_prompt(query: str, chunks: list[RetrievedChunk],
                          extra_context: str | None = None,
                          max_chunks: int = 5, max_chunk_chars: int = 1200,
                          max_total_chars: int = 7000,
                          now: datetime | None = None) -> str:
    # 프롬프트 슬림화: 청크 수·길이 상한으로 7B 모델 속도·정확도 확보.
    # (긴 컨텍스트는 생성 지연 + 핵심 누락 환각의 주원인)
    ctx_parts, total = [], 0
    for i, c in enumerate(chunks[:max_chunks]):
        body = _scrub_cumulative_table(c.content or "")[:max_chunk_chars]
        part = (f"[자료 {i + 1}] 제목: {c.title or '-'} | 출처: {c.source_url or '-'}\n"
                f"본문: {body}")
        if total + len(part) > max_total_chars:
            break
        ctx_parts.append(part)
        total += len(part)
    ctx = "\n\n".join(ctx_parts)
    if extra_context:  # adaptive: 상위 출처 페이지 본문(있으면 길이 제한해 추가)
        ctx += f"\n\n[상위 출처 페이지 전문]\n{extra_context[:2000]}"
    return (
        _date_line(now) +
        "당신은 충남대학교 학사·생활 안내 봇입니다. 아래 [참고 자료]에 근거해서만 답하세요.\n"
        "자료에 없는 내용은 지어내지 말고 '관련 정보를 찾을 수 없습니다'라고 답하세요.\n"
        "교과목/과목명 규칙: ①자료에 한국어 명칭이 있으면 한국어를 본문에, 영문은 괄호 병기. "
        "②자료에 영문 명칭만 있으면 **영문 원문 그대로** 쓰고 한국어 번역은 절대 만들지 마세요. "
        "예를 들어 'Theory of Computation', 'Counseling for Future Planning'을 '추측 계산 이론', "
        "'코딩 지원 강좌' 같은 한국어로 임의 번역하면 환각이며 금지. 영문만 있으면 그대로 두고 "
        "옆에 사견 번역을 붙이지 마세요.\n"
        "[누적 수료학점은 답변에 넣지 말 것] 학사요람 제59조 ⑤의 '제1학년 33학점, 제2학년 65학점, "
        "제3학년 98학점, 제4학년 130학점' 같은 누적 수료 인정학점 표는 **졸업학점 답변에 절대 "
        "옮기지 마세요**. 이 숫자를 학년별·학기별 이수 분배인 것처럼 임의로 나누는 행동(예: "
        "8학기에 130학점을 16.25씩, 33씩 분배)은 환각이며 금지. 졸업학점은 '총 130학점' 한 줄로 "
        "안내하고, 학년·학기 분류는 교과목 목록을 직접 그 학년-학기 라벨 아래에 나열하는 "
        "방식으로만 표현하세요.\n"
        "[학년-학기 표기 해석] 표의 'Grade' 또는 '학년' 컬럼이 'X-Y' 형식(예: '1-1', '1-2', "
        "'2-1', '2-2', '3-1', '4-2')으로 적혀 있으면, **X는 학년, Y는 학기**입니다. '1-1'은 "
        "1학년 1학기, '2-2'는 2학년 2학기, '4-1'은 4학년 1학기를 의미합니다. 절대 순번 "
        "(1,2,3,4번째 항목)으로 해석하지 마세요. **답변에 표기할 때는 'X-Y 학년' 같은 어색한 "
        "표기를 쓰지 말고 반드시 'X학년 Y학기'(예: '1학년 1학기', '2학년 2학기') 자연어로 "
        "적으세요.** 학년·학기는 반드시 오름차순으로 정렬하세요(1학년 1학기 → 1학년 2학기 → "
        "2학년 1학기 …). 뒤 학기를 앞 학기보다 먼저 쓰지 마세요. 자료에 1-1부터 4-2까지 8개 "
        "학기가 있으면 답변에도 8개 학기를 모두 포함하세요(임의로 빠뜨리지 마세요).\n"
        "[중복·모순 정리] 같은 교과목번호나 같은 과목이 자료에 여러 번 나오거나 과목명이 서로 "
        "다르게 적혀 모순될 때는, **추측하거나 어느 게 맞다고 설명(메타발화)하지 말고** 그 항목을 "
        "한 번만 깔끔히 적으세요. '중복된 이름으로 추측하므로', '~일 가능성이 크다' 같은 "
        "자기설명·추측 문구는 절대 답변에 쓰지 마세요. 자료가 모순되어 확실치 않으면 그 항목은 "
        "그냥 생략하세요. 동일 항목을 두 번 나열하지 마세요.\n"
        "[외국문자 금지] 답변은 한국어와 정상적인 영문 고유명사·약어(예: AI, IoT)만 사용하세요. "
        "키릴문자·아랍문자·태국어 등 한국어 답변에 어울리지 않는 외국문자를 한 글자라도 섞지 "
        "마세요(예: '디자чер' 같은 키릴 누수 금지). 한국어 과목명에 외국문자가 끼어 보이면 "
        "올바른 한국어 글자로 적으세요.\n"
        "[간결성] 군더더기 없이 핵심만 적으세요. '따라서 ~', '이 외에도 학생들은 ~', 같은 말을 "
        "반복하는 권유·맺음 문장은 쓰지 마세요. 목록형 답변은 목록만으로 끝내고 불필요한 요약 "
        "문단을 덧붙이지 마세요. 질문과 무관한 행정 절차(졸업사정 절차·명단 작성 등)는 나열하지 마세요.\n"
        "질문이 특정 학과·전공의 커리큘럼/과목을 물으면, 그 학과와 명백히 무관한 타 학과 과목"
        "(예: 인공지능 질문에 식물·생명과학·농학 과목)은 제외하고 관련 과목만 나열하세요. "
        "관련 과목은 자료에 있는 만큼 빠짐없이 나열하세요.\n"
        "[학부/대학원 기본값] 질문에 '대학원·석사·박사'가 명시되지 않으면 학부 과정으로 가정해 "
        "답하세요. 자료에 학부+대학원이 함께 들어 있더라도 답변엔 학부 과정만 포함하고 "
        "대학원 전용 과목·규정(논문연구·특론·특강·세미나·고급…·석박사·수료학점·"
        "학위청구논문·수학능력확인서·논문제출자격·종합시험 등)은 제외하세요.\n"
        "[표·목록 해석] 자료에 '대학명/학과명 → 학점' 표나 목록이 나오면 '질문 학과·학부와 "
        "정확히 일치하는 행'만 사용하세요. 다른 대학·학과 행(예: '수학교육과 142', '영어교육과 "
        "150', '건축학 5년제 166', '약학과 232', '의학과 164')의 숫자를 질문 학과 졸업학점으로 "
        "옮기지 마세요. 질문 학과가 표에 명시되어 있지 않으면 표 윗줄의 '원칙 X학점'을 적용하고, "
        "원칙도 없으면 '자료에 해당 학과의 정확한 졸업학점이 명시되어 있지 않습니다'라고 답하세요. "
        "예: 인공지능학과는 공과대학 컴퓨터인공지능학부 소속이며 제59조 예외 표에 별도 명시가 "
        "없으면 원칙(130학점)이 적용됩니다.\n"
        "[숫자·금액·일자 환각 금지] 학점·금액·연도·일자 같은 수치는 자료 본문에 그 숫자가 "
        "글자 그대로 적혀 있을 때만 사용하세요. 자료에 없는 수치는 절대 일반 지식·상식으로 "
        "채우지 말고 '자료에 정확한 수치가 명시되어 있지 않습니다. 정확한 정보는 충남대학교 "
        "공식 홈페이지의 해당 분야 안내 페이지를 확인해 주세요'처럼 일반 안내로 답하세요. "
        "거부 답변에 '학사요람 또는 학과 공지의 첨부파일' 같은 특정 출처 문구를 그대로 옮겨 "
        "엉뚱한 분야(예: 등록금) 답변에 붙이지 마세요. 인접 학과·학부의 표를 질문 학과 답인 양 "
        "옮기지 마세요(예: '인공지능학과'를 물었는데 '정보통신융합학부 졸업요건' 표를 끌어와 "
        "수치를 합성하는 행위 금지).\n"
        "[본문 한정·환각 금지] 답변에 적는 모든 숫자·기간·요건·점수·비율·절차·날짜는 **반드시 "
        "[참고 자료] 본문에 글자 그대로 명시된 것만** 인용하세요. 본문에 명시되어 있으면 그대로 "
        "인용해 정상적으로 답하면 됩니다(있는 근거를 거부하지 마세요). 본문에 없으면 '일반적으로 "
        "N개월/N주일/N점/N학년/평점 N 이상' 같은 일반 지식·상식·관행으로 **추측하거나 채워 넣지 "
        "마세요**. 본문에 없는 절차·요건·기간을 '따라서 ~됩니다'처럼 단정해 만들어내는 것은 "
        "환각이며 금지입니다. 핵심 수치·절차가 본문에 없으면 '자료에 명시되어 있지 않습니다 — "
        "정확한 내용은 학과 사무실 또는 충남대학교 공식 페이지에서 확인해 주세요'로 답하세요.\n"
        "[본문 관련성 점검] 답하기 전에 [참고 자료]가 질문의 **핵심 대상과 같은 주제인지** 먼저 "
        "확인하세요. 자료가 질문과 **다른 대상**을 다루면(예: '학사경고'를 물었는데 본문이 "
        "'생활관 벌점/상벌', 일반 학생을 물었는데 본문이 '의과대학·의예과' 전용 규정) 그 본문을 "
        "억지로 답으로 쓰지 말고 '질문에 정확히 해당하는 자료가 없습니다 — 공식 페이지에서 확인해 "
        "주세요'로 위임하세요. 검색돼 나온 본문이라고 해서 질문과 무관한 내용까지 사실처럼 "
        "단정하지 마세요. (단, 본문이 질문과 같은 주제이면 정상적으로 답하세요.)\n"
        "[개념 구분] 질문에 상반·유사 개념쌍(조기졸업↔졸업유예, 전과↔복수전공, 정규학기↔계절학기, "
        "수강신청↔수강정정)이 섞여 있으면, 먼저 두 개념의 차이를 한 줄로 구분한 뒤 **본문이 "
        "뒷받침하는 쪽만** 답하세요. 한쪽 개념의 규정(예: 계절학기 학점상한)을 다른 개념(정규학기)에 "
        "그대로 옮기지 마세요.\n"
        "[학과·입학년도 범위] 졸업요건·이수학점·인증기준 등이 학과나 입학년도(학번)별로 다른 "
        "질문이면, 먼저 '학과·입학년도별로 다릅니다'를 밝힌 뒤 본문에 있는 해당 학과 기준만 "
        "인용하세요. 특정 학과·학번의 수치(예: 토목과 토익 커트라인)를 질문자(예: 다른 공대생) "
        "전체에 해당하는 것처럼 단정하지 마세요. 본문에 질문 학과·학번 기준이 없으면 위임하세요.\n"
        "[잘못된 전제 교정] 질문이 특정 제도·특례(예: '직장인 출석특례', '경력 학점인정', "
        "'학생 정년', '외국인 면제')의 존재를 전제하더라도, [참고 자료]에 그 제도가 명시돼 "
        "있지 않으면 전제를 사실로 받아들이지 마세요. 먼저 '자료상 그런 제도(특례)는 확인되지 "
        "않습니다'라고 교정한 뒤, 무관한 절차(병가·공결 휴가신청 등)를 그 전제에 끼워맞추지 "
        "말고 일반 안내만 하세요. (단, 자료에 실제로 명시된 제도는 정상적으로 답하고, 근거가 "
        "없는 전제일 때만 교정하세요.)\n"
        "[수치 전제 수용 금지·학부/대학원 구분] 사용자가 '~ 맞지?', '~면 되지?', '~ 아냐?'처럼 "
        "특정 수치·구간·기준을 단정해 물어도 그 전제를 그대로 인정하지 말고 [참고 자료] 본문과 "
        "대조해, 다르면 본문 값으로 정정하세요(예: 사용자가 '대학원 4학점은 절반'이라 해도 본문이 "
        "'4학점 이상 전액'이면 전액으로 교정). 특히 학부와 대학원은 평점·학점·등록금 구간·요건이 "
        "서로 다를 수 있으니, 본문에 대학원 값이 따로 있으면 학부 값을 대학원에 옮겨 '동일하다'고 "
        "답하지 마세요. 본문에 한쪽(예: 대학원) 값이 없으면 '학부 기준은 …이며, 대학원 기준은 "
        "자료에 명시되어 있지 않습니다'처럼 구분해 답하세요.\n"
        "[지난 학년도 날짜 주의] 질문이 '신청 기간·일정·언제'처럼 현재/다음 일정을 묻는데 "
        "[참고 자료] 본문의 날짜가 지난 학년도(예: 2024년 등 올해보다 이전 연도)면, 그 지난 "
        "날짜를 현재 일정인 것처럼 단정하지 마세요. '자료의 일정은 지난 학년도 기준일 수 있어 "
        "최신 일정은 충남대학교 학사일정 페이지에서 확인해 주세요'라고 안내하세요. 또한 본문에 "
        "없는 일정을 개강일 등에서 'N주 후일 것'이라 추정하거나 '예상됩니다'로 창작하지 마세요.\n"
        "[일반지식 서술 누수 금지] '일반적으로 대학마다…', '보통은…', '대체로…' 같이 자료에 "
        "없는 내용을 일반 지식·관행으로 지어내 서술하는 것도 금지입니다. 자료에 근거가 없으면 "
        "정책·절차를 분기해 창작하지 말고 '자료에 명시되어 있지 않습니다 — 학과/공식 페이지 "
        "확인'으로 위임하세요.\n"
        "여러 자료에 걸친 정보는 종합해서 답하고, 자연스러운 한국어 문장으로만 쓰세요. "
        "'[문서 1]', '자료 2 참조' 같은 출처 번호 표기는 절대 쓰지 마세요(출처는 따로 표시됩니다).\n\n"
        f"[참고 자료]\n{ctx}\n\n[질문]\n{query}"
    )


def build_cafeteria_table(menus: list[DailyMenu]) -> str:
    rows = ["| 식당 | 끼니 | 대상 | 메뉴 |", "|---|---|---|---|"]
    for m in menus:
        rows.append(f"| {m.place} | {m.meal_type} | {m.target} | {', '.join(m.menu_list)} |")
    return "\n".join(rows)


# 질의의 끼니 키워드 → meal_type 매핑. 여러 표현을 표준 끼니로.
_MEAL_TYPE_KEYWORDS = {
    "조식": "조식", "아침": "조식", "아침밥": "조식",
    "중식": "중식", "점심": "중식", "점심밥": "중식",
    "석식": "석식", "저녁": "석식", "저녁밥": "석식",
}


def filter_menus_by_meal_type(menus: list[DailyMenu], query: str) -> list[DailyMenu]:
    """질의에 끼니 키워드(점심/석식 등)가 있으면 그 끼니만 남긴다. 없으면 전체 유지."""
    wanted = {mt for kw, mt in _MEAL_TYPE_KEYWORDS.items() if kw in (query or "")}
    if not wanted:
        return menus
    filtered = [m for m in menus if m.meal_type in wanted]
    return filtered or menus  # 해당 끼니가 표에 없으면 전체로 폴백(빈 표 방지)


def build_cafeteria_header(menus: list[DailyMenu], date_label: str = "",
                           query: str = "") -> str:
    """질의 날짜의 확정 식단표(코드 생성). LLM 을 거치지 않으므로 환각/잘림/깨짐이 원천 불가능.
    질의에 끼니 키워드가 있으면 그 끼니만 표시(점심→중식, 저녁→석식 등)."""
    when = date_label or "오늘"
    shown = filter_menus_by_meal_type(menus, query)
    return f"[{when} 식단표]\n{build_cafeteria_table(shown)}"


# '가장 비싼/싼', '추천', 끼니/식당 한정 등 표를 그대로 보여주는 것 이상으로
# 비교·추론이 필요한 질의. 이런 질의만 LLM 보조를 붙이고, 그 외에는 표만 반환한다.
_MEAL_ANALYSIS_RE = re.compile(
    r"가장|제일|비싼|싼|저렴|비교|추천|뭐가\s*좋|골라|어떤\s*게|메뉴\s*중|얼마")


def is_meal_analysis_query(query: str) -> bool:
    return bool(_MEAL_ANALYSIS_RE.search(query or ""))


def build_cafeteria_prompt(query: str, menus: list[DailyMenu], date_label: str = "") -> str:
    """분석 질의(가격 비교·추천 등) 전용 LLM 프롬프트. 표는 코드가 따로 출력하므로
    여기서는 LLM 에게 '한두 문장 분석'만 요청한다(표 재출력 금지 → 잘림 회피)."""
    when = date_label or "오늘"
    return (
        "당신은 충남대학교 학식 안내 봇입니다.\n"
        f"아래 [식단표]는 질문이 가리키는 날짜({when})의 실제 메뉴입니다.\n"
        "표는 이미 사용자에게 보여줬으니 **표를 다시 출력하지 말고**, 질문에 대한 "
        "분석만 한두 문장으로 답하세요.\n"
        "'가장 비싼/싼' 등은 메뉴 옆 괄호 안 숫자(원)를 직접 비교해 답하세요. 지어내지 마세요. "
        "메뉴 이름은 표에 적힌 글자 그대로 쓰세요.\n\n"
        f"[{when} 식단표]\n{build_cafeteria_table(menus)}\n\n[질문]\n{query}"
    )


def build_notice_prompt(query: str, items: list[NoticeItem], label: str,
                        body: str | None = None,
                        now: datetime | None = None) -> str:
    lines = "\n".join(f"- ({it.posted or '날짜미상'}) {it.title}" for it in items)
    extra = (f"\n\n[관련 공지 본문]\n{body}" if body else "")
    body_rule = (
        "[관련 공지 본문]이 있으면 본문에 실제로 적혀 있는 사실만(언제·어디서·대상·시험장 배정 등) "
        "정리해 답하세요. 본문에 없는 항목은 절대 일반 지식으로 채우지 말고 "
        "'본문에는 일정/응시장소 등만 있고 행사 내용은 안내돼 있지 않습니다'처럼 정직히 밝히세요.\n"
        if body else "")
    return (
        _date_line(now) +
        f"당신은 충남대학교 {label} 공지 안내 봇입니다. 아래 [최근 공지 목록]은 최신순입니다.\n"
        "다음 규칙을 지키세요:\n"
        "1) '최근/최신/요즘 공지' 같은 일반적 질의면 맨 위에서 **3~5건**을 글머리표로 나열하세요. "
        "1건만 보여주지 마세요. '가장 최신 1건'을 명시적으로 물을 때만 1건.\n"
        "2) '볼만한/관련 소식'이면 질문 주제에 맞는 항목 2~5건을 추려 안내.\n"
        "3) 공지 제목과 날짜는 자료의 한국어 표기 그대로(요일 '(일)/(월)/(화)…') 사용. "
        "절대 한자/일본어/영어로 변환하지 마세요(예: '일'→'日' 금지).\n"
        f"{body_rule}"
        "4) 목록/본문에 없는 내용은 추측하거나 일반 지식으로 답하지 말고 "
        "'관련 공지를 찾을 수 없습니다'라고만 답하세요(교수 소개·인물 정보 등도 마찬가지). "
        "날짜가 있으면 함께 알려주세요. **단, 위에서 공지를 하나라도 나열했다면 답변 끝에 "
        "'관련 공지를 찾을 수 없습니다'를 덧붙이지 마세요(나열과 모순됨).**\n"
        "5) [Anti-Meta] 질문이 가리키는 도메인(예: '도서관 공지')과 자료에 들어온 도메인(예: "
        "컴퓨터인공지능학부 공지)이 일치하지 않으면, '제공된 자료는 ~ 학부 공지이며…'처럼 "
        "자료의 정체를 변명·설명하지 마세요. 그 경우 곧바로 '관련 공지를 찾을 수 없습니다.' "
        "한 줄로만 답하세요.\n\n"
        f"[최근 공지 목록]\n{lines}{extra}\n\n[질문]\n{query}"
    )


def _dedup_refs(chunks: list[RetrievedChunk]) -> list[Reference]:
    seen, refs = set(), []
    for c in chunks:
        key = (c.title, c.source_url)
        if key in seen:
            continue
        seen.add(key)
        refs.append(Reference(title=c.title, source_url=c.source_url))
    return refs


class CNUGenerator:
    def __init__(self, llm: AnswerLLM):
        self.llm = llm

    def generate(self, query: str, retrieval: RetrievalResult,
                 extra_context: str | None = None) -> CNUBotResponse:
        # 우회: 거부/만료면 LLM 안 돌리고 메시지 그대로
        if retrieval.is_fallback:
            return CNUBotResponse(
                answer=retrieval.fallback_message or REFUSAL_MSG,
                intent=retrieval.intent, is_fallback=True,
            )
        if retrieval.intent == Intent.ACADEMIC:
            answer = self.llm.generate(
                build_academic_prompt(query, retrieval.chunks, extra_context),
                # 긴 답변(과목 목록 등) 잘림 방지. greedy라 짧은 답은 EOS에서 일찍 종료 → 속도 영향 적음.
                max_new_tokens=512 if extra_context else 256,
            )
            return CNUBotResponse(
                answer=answer, references=_dedup_refs(retrieval.chunks),
                intent=Intent.ACADEMIC,
            )
        if retrieval.intent == Intent.CAFETERIA:
            # 표는 코드(build_cafeteria_table)가 확정 출력 → LLM 환각('내일 정보 없음')·
            # 토큰 잘림·멀티바이트 깨짐이 원천 불가능. LLM 은 분석 질의일 때만 보조로 붙인다.
            menus = retrieval.menus
            label = retrieval.meal_date_label or ""
            table = build_cafeteria_header(menus, label)
            if not is_meal_analysis_query(query):
                # 단순 조회('내일/오늘 메뉴') → 표만 반환(LLM 미사용).
                return CNUBotResponse(answer=table, intent=Intent.CAFETERIA)
            # 분석 질의('가장 비싼' 등) → 표 + LLM 한두 문장 분석.
            note = self.llm.generate(
                build_cafeteria_prompt(query, menus, label),
                max_new_tokens=256,
            )
            answer = f"{table}\n\n{note}".strip() if note else table
            return CNUBotResponse(answer=answer, intent=Intent.CAFETERIA)
        # 방어: 예상 밖 intent
        return CNUBotResponse(answer=REFUSAL_MSG, intent=retrieval.intent, is_fallback=True)

    def generate_notice(self, query: str, items: list[NoticeItem], label: str,
                        focus: NoticeItem | None = None,
                        body: str | None = None,
                        now: datetime | None = None) -> CNUBotResponse:
        """temporal_notice: 라이브 공지 목록(+특정 공지 본문)을 LLM이 추려 답변."""
        answer = self.llm.generate(
            build_notice_prompt(query, items, label, body=body, now=now),
            max_new_tokens=400)
        # 본문 참조한 공지를 ref 맨 앞에
        head = [focus] if focus else []
        refs = [Reference(title=it.title[:70], source_url=it.url)
                for it in head + [x for x in items if x is not focus][:5]][:5]
        return CNUBotResponse(answer=answer, references=refs,
                              intent=Intent.TEMPORAL_NOTICE)


def _prequantized_dir(model_id: str) -> str | None:
    """사전 양자화 모델 저장 경로. Drive 마운트 시 Drive, 아니면 None(저장 생략).
    환경변수 CNU_QUANT_DIR 로 override 가능."""
    env = os.environ.get("CNU_QUANT_DIR")
    if env:
        return os.path.join(env, model_id.replace("/", "_") + "-4bit")
    drive = "/content/drive/MyDrive/hf_cache"
    if os.path.isdir("/content/drive/MyDrive"):
        return os.path.join(drive, model_id.replace("/", "_") + "-4bit")
    return None  # 로컬/비-Colab: 사전 저장 생략(매번 변환)


def _load_quantized_model(model_id: str, gpu_idx: int):
    """4bit 모델 로드. 사전 양자화본이 Drive에 있으면 변환 없이 로드(빠름),
    없으면 FP16→nf4 변환 후 Drive에 저장(다음 재시작부터 재사용)."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    qdir = _prequantized_dir(model_id)
    # 1) 사전 양자화본이 있으면 변환 없이 바로 로드 (quantization_config 가 모델에 내장됨)
    if qdir and os.path.isdir(qdir) and os.listdir(qdir):
        print(f"[quant] 사전 양자화 모델 로드(변환 생략): {qdir}")
        return AutoModelForCausalLM.from_pretrained(
            qdir, device_map={"": gpu_idx}, torch_dtype=torch.float16,
        )

    # 2) 없으면 FP16 받아 nf4 변환
    print("[quant] 4bit 변환 중(최초 1회)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": gpu_idx},
        torch_dtype=torch.float16,
    )
    # 3) Drive 에 저장 → 다음 재시작부터 변환 생략
    if qdir:
        try:
            os.makedirs(qdir, exist_ok=True)
            model.save_pretrained(qdir)
            print(f"[quant] 사전 양자화 모델 저장 완료: {qdir} (다음 재시작부터 재사용)")
        except Exception as e:
            print(f"[quant] 저장 실패(동작엔 무관): {type(e).__name__}: {e}")
    return model


class HFAnswerLLM:
    """Qwen2.5-7B-Instruct 4bit(bnb) @cuda:0. transformers 핀(4.51.3) 안전."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda:0",
        max_new_tokens: int = 512,
        block_hanzi: bool = True,
    ):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.max_new_tokens = max_new_tokens
        self.block_hanzi = block_hanzi
        gpu_idx = int(device.split(":", 1)[1]) if ":" in device else 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = _load_quantized_model(model_id, gpu_idx)
        self.model.eval()
        # greedy(do_sample=False) 사용 → 샘플링 파라미터 제거 (매 생성 경고 소거)
        gc = self.model.generation_config
        gc.temperature = gc.top_p = gc.top_k = None
        # GPU0 단일 가중치를 M2/M4가 공유 → model.generate 직전 한 줄만 직렬화.
        # 토크나이징/디코딩은 락 밖이라 다중 스레드 병렬.
        self._lock = threading.Lock()
        # 한자 토큰 id 사전 계산 (중국어 drift 차단용, §5 검증 기법)
        self._hanzi_ids: list[int] = []
        if block_hanzi:
            for tid in range(len(self.tokenizer)):
                if _CJK_RE.search(self.tokenizer.decode([tid])):
                    self._hanzi_ids.append(tid)

    def generate(self, prompt: str, max_new_tokens: int | None = None,
                 block_hanzi: bool | None = None) -> str:
        import torch

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        from transformers import LogitsProcessorList

        use_block = self.block_hanzi if block_hanzi is None else block_hanzi
        lp = None
        if use_block and self._hanzi_ids:
            ban = torch.tensor(self._hanzi_ids, device=self.model.device)

            def _mask(_ids, scores):  # 한자 토큰 로짓 → -inf
                scores[..., ban] = float("-inf")
                return scores
            lp = LogitsProcessorList([_mask])
        # T4 14GB OOM 회피: SDPA 의 memory-efficient 백엔드 강제 (math/flash 끔).
        # Qwen2.5-7B 4bit + 긴 RAG 프롬프트(>8k tok)에서 attention 메모리 O(n²)→O(n) 으로.
        # 품질 손실 없음 (수학적으로 동일), 평균 30% 빠름.
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with torch.no_grad(), self._lock, sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False, repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id, logits_processor=lp,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        # 멀티바이트(한글)가 토큰 경계에서 잘리면 U+FFFD(�)가 남는다 → 제거.
        text = text.replace("�", "").strip()
        # F3: 응답 끝 메타태그(관리[SystemMessage]: …) 7B 환각 누수 컷.
        return _strip_meta_tail(text)

    def generate_stream(self, prompt: str, max_new_tokens: int | None = None,
                        block_hanzi: bool | None = None):
        """토큰 스트리밍(SSE용). model.generate를 스레드에서 돌리고 streamer로 조각 yield.
        lock은 스레드 안에서 잡아 generate()와 직렬화 유지."""
        import threading

        import torch
        from transformers import LogitsProcessorList, TextIteratorStreamer

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        use_block = self.block_hanzi if block_hanzi is None else block_hanzi
        lp = None
        if use_block and self._hanzi_ids:
            ban = torch.tensor(self._hanzi_ids, device=self.model.device)

            def _mask(_ids, scores):
                scores[..., ban] = float("-inf")
                return scores
            lp = LogitsProcessorList([_mask])

        from torch.nn.attention import SDPBackend, sdpa_kernel
        def _run():
            # T4 OOM 회피: 위 generate() 와 동일하게 EFFICIENT_ATTENTION 강제
            with torch.no_grad(), self._lock, sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                    do_sample=False, repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.eos_token_id,
                    logits_processor=lp, streamer=streamer,
                )
        th = threading.Thread(target=_run)
        th.start()
        try:
            for tok in streamer:
                if tok:
                    yield tok
        finally:
            th.join()
