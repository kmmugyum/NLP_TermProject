# Deep Dive: 날짜 강건성 + 일반화 답변 (questioner ↔ critic ↔ writer)

생성: 2026-06-11, 연구서버(GPU 실측). 브랜치 `fix/env-compat-py310-torch251`.
방식: /team 구조 — **질문자(red-team) → 봇 eval → 비판자(critic) → writer 수정 → 재검증 → 비판자 최종**.

## Trace Findings (24개 미테스트 질문 probe)
질문자 agent가 5개 카테고리(날짜·OOD·엣지학사·모호·적대적) 24문항 설계 → 봇 실행 → 비판자 평가.
- **잘됨**: 엣지학사(휴학6학기·재수강성적삭제·복수전공vs부전공·장학3.25), OOD거부(날씨/라면/비트코인), 타기관거부(서울대).
- **결함(critic 규명)**:
  - **[CRITICAL] 날짜 grounding 부재**: `build_academic_prompt`/`build_notice_prompt`/calendar inline 프롬프트에 현재날짜 미주입. `_now()`(KST)는 식단/세션에만 사용. → "오늘 며칠?" 모름, "다음학기 몇 달?" **"약 2개월" 환각**(오늘 6/11).
  - **[HIGH] 되물음 부재**: 모호/빈 질문("그거 언제까지?"·"신청 언제부터?"·"?") → 환각·임의답. clarification 경로 grep 0건.
  - **[MED] 메타태그 누수**: 계절학기 답에 "관리[SystemMessage]:" 7B 환각 누수.
  - **[MAJOR, 후속]** 멀티intent first-match early-exit, 도서관 운영시간 OOS오거부, 도발→학식표 덤프, 한영혼합 오거부.

## 적용 수정 (이번 범위: 날짜강건성 + 안전한 일반화)
| F | 수정 | 위치 |
|---|---|---|
| F1 | 현재날짜(KST `_now()`) 주입 — `[오늘 날짜] YYYY-MM-DD` 한 줄 + 기간계산 기준 지시 | build_academic_prompt·build_notice_prompt·generate_notice + **calendar inline 프롬프트**(critic이 찾은 누락 경로) + 호출부 now 전달 |
| F2 | 모호/빈 질문 되물음 guard(빈입력·지시어only·주어없는신청) → 정적 되물음 | `_plan` 초입 |
| F3 | trailing 메타태그(`[SystemMessage]`류) 출력 scrub | HFAnswerLLM.generate |

안전선: 프롬프트+now plumbing+좁은 guard+끝부분 scrub만. retrieval/index/생성로직/모델 불변.

## 실측 before → after
| 질문 | before | after |
|---|---|---|
| 오늘 며칠? | "모름" | **"2026년 6월 11일 목요일"** ✅ |
| 다음학기 개강 몇 달? | "약 2개월"(환각) | **"3월 3일, 약 3달 8일"**(anchor 계산) ✅ |
| 종강까지 며칠? | "약 4개월, 추가정보 필요" | **"6/11 기준... 6/22 이전 추정"**(anchor 추론) ✅ |
| 그거/신청/?(모호) | 환각·임의답 | **"질문이 모호해요. 무엇에 대해..."** 되물음 ✅ |
| 계절학기(누수) | "관리[SystemMessage]:" 누수 | 깨끗 ✅ |
| 학식·졸업학점130·미적분=1학년 | 정상 | 동일(회귀 0) ✅ |

## 비판자 최종 판정
**ACCEPT-WITH-RESERVATIONS** — F1/F2/F3 검증·무회귀. F1 calendar 경로 보강으로 "다음학기 2개월" 환각 제거 완결.

## MAJOR 라우팅 4건 (2차 라운드, writer↔critic, 완료)
critic이 짚은 정확한 위치로 writer 수정 → MAJOR타깃+회귀18건 eval → critic 최종.
| M | 결함 | 수정(module4_api `_plan` 좁은 가드) | 결과 |
|---|---|---|---|
| M1 | 멀티intent first-match로 한쪽 드롭 | `_multi_intent()`(연결어+2신호) 시 URL 빠른경로 skip→academic RAG | 장문 신입생질문 구조화 답변 ✅ / 학식+학사 교차파이프라인은 학식 누락(비차단) |
| M2 | 도서관 운영시간 OOS 오거부 | `_LIBRARY_HOURS_RE`+"도서관" → library.cnu 안내 | ✅ |
| M3 | 도발("지어내잖아")→학식표 덤프 | `_META_PROVOKE_RE` 가드(cafeteria override보다 먼저)→차분한 출처 설명 | ✅ |
| M4a | 한영혼합 학사질의 notice 오라우팅 | 영문토큰 notice선점(line1262) **앞에** kor+en 학사 예외 hoist→academic | ✅ ("130 credits" 영어 답) |
| M4b | 감정표현('상담' 없으면) flat OOS | `_EMOTION_RE`→공감+학생상담센터 | ✅ |
- 회귀: OOD거부·엣지·서울대·학식표·130·미적분=1학년·학사일정·되물음·날짜·devday 공지선점 **전부 정상(0)**.
- 1차 writer가 M4a를 line1289(선점 뒤)에 둬 dead code였던 것을 critic이 규명→hoist로 활성화.

## 잔여(비차단)
- "몇 주차"는 temporal_notice 라우팅 + 주차 데이터 부재 → grounding만으론 한계.
- "다음 학기" 학기 모호성 + 2학기 개강 calendar 데이터 부족(retrieval/data).
- M1 학식+학사 교차파이프라인 통합답변은 cross-pipeline 기능(라우팅 버그 아님).
