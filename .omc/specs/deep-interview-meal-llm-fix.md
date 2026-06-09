# Deep Interview Spec: 학식 답변 LLM 환각·잘림 수정 (표는 코드로 통합 출력)

## Metadata
- Interview ID: meal-llm-fix-2026-06-09
- Rounds: 1 (+ Round 0 topology)
- Final Ambiguity Score: ~7%
- Type: brownfield
- Generated: 2026-06-09
- Threshold: 0.2 (20%) / Source: default
- Status: PASSED

## 이전 진단 반증 (중요)
직전 인터뷰는 "Colab이 live fetch를 못 한다"고 진단했으나, **실제 Colab 실행으로 반증됨**:
- 서버 응답이 "2026년 6월 10일(내일) 식단표"를 이미 포함 → **fetch는 정상 작동**
- 서버 로그에 `[meal]` 실패 라인 전혀 없음 → 크롤 성공
- 따라서 fetch 강건화 수정은 유효하나, **현재 증상의 원인이 아님**. 진짜 원인은 LLM 생성 단계.

## Topology
| Component | Status | Description | Coverage |
|-----------|--------|-------------|----------|
| "내일 메뉴 없음" 환각 | active | menus 있는데 LLM이 "정보 없음"이라 답함 | AC1, AC4 |
| "오늘 메뉴 잘림/깨짐" | active | max_new_tokens=384 초과 잘림 + 멀티바이트 `�` | AC2, AC3, AC4 |

## Goal
학식 답변에서 **메뉴 표 자체는 코드(`build_cafeteria_table`)가 확정 출력**하고, LLM은 요약·가격비교·날짜해석 같은 보조 역할만 하도록 생성 경로를 바꾼다. 이 단일 변경으로 (1) "내일 메뉴 없음" 환각, (2) 토큰 초과 잘림, (3) 멀티바이트 깨짐을 한꺼번에 근본 차단한다.

## 근본 원인 (코드 검증 완료)
1. **환각**: `build_cafeteria_prompt`(module4_generator.py:113)는 "내일이라 묻더라도 없다고 하지 마라"를 명시하지만, `do_sample=False`(greedy, line 284) 7B Qwen2.5 모델이 instruction을 무시 → **매번 동일하게** "정보 없습니다" 출력. retriever는 올바른 menus를 넘김(module3_retriever.py:454).
2. **잘림**: `generate(..., max_new_tokens=384)`(line 194). 오늘 표는 7개 식당 × 조/중/석식 × 직원/학생이라 384 토큰 초과 → 중간 잘림. 마지막 `�`는 한글 멀티바이트가 토큰 경계에서 잘린 디코딩 잔재.

## Constraints
- 모델은 greedy(`do_sample=False`) 7B → 프롬프트 설득력 약함, 코드 강제가 확실
- 기존 `build_cafeteria_table(menus)`(module4_generator.py:106) 재사용 (이미 정확한 표 생성)
- `meal_date_label`(예: "2026-06-10 (수)")은 retriever가 이미 제공 → 날짜 헤더에 사용
- SSE 스트리밍 경로(`generate_stream`)와도 호환되어야 함
- 멀티바이트 안전 디코딩 (토큰 경계 잘림 시 `�` 방지)

## Non-Goals
- live fetch / crawl 로직 변경 (이미 정상)
- 모델 교체나 파인튜닝
- 공지(notice)·학사(academic) 답변 경로

## Acceptance Criteria
- [ ] AC1: "내일 학식 메뉴" 질의 시 내일(다음날) 식단 표가 정상 표시됨 ("정보 없음" 환각 없음)
- [ ] AC2: "오늘 학식 메뉴" 질의 시 표 전체가 잘리지 않고 끝까지 표시됨
- [ ] AC3: 출력 끝에 `�` 같은 깨진 문자가 없음
- [ ] AC4: 메뉴 표는 build_cafeteria_table 출력과 100% 일치 (LLM이 메뉴명 변형/누락 못 함)
- [ ] AC5: 날짜 헤더가 질의 날짜(오늘/내일)와 일치
- [ ] AC6: SSE 스트리밍에서도 동일하게 작동

## 구현 방향 (통합 접근)
`CNUGenerator.generate`의 CAFETERIA 분기를 다음과 같이 바꾼다:
1. **표는 코드로 확정 출력**: `f"[{label} 식단표]\n{build_cafeteria_table(menus)}"`를 답변의 골격으로.
2. LLM은 (a) 질의가 단순 "메뉴 보여줘"면 표만 반환(LLM 생략 가능), (b) "가장 비싼/요약" 같은 분석 질의면 표 + LLM 보조 답변.
3. LLM 보조를 쓸 경우에도 표는 코드 출력이 우선이라 잘림/환각 무관.
4. **후처리 안전망**: LLM 답변에 "정보 없", "없습니다" 패턴이 있고 menus가 있으면 표로 대체.
5. **디코딩 방어**: `tokenizer.decode`에 `errors="ignore"` 또는 멀티바이트 경계 검증.

## Ontology
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| DailyMenu | core | place, meal_type, target, menu_list | build_cafeteria_table 입력 |
| RetrievalResult | core | menus, meal_date_label, is_fallback | retriever→generator |
| CNUGenerator | core | llm | CAFETERIA 분기 생성 |
| AnswerLLM | external | max_new_tokens, do_sample=False | greedy 7B Qwen2.5 |

## Interview Transcript
<details>
<summary>Q&A</summary>

### Round 0 — Topology
실행 로그/화면 분석: fetch 정상(6/10 데이터 도착), 두 문제 모두 LLM 단계. 사용자: 둘 다 맞음, "내일 없음"은 매번 동일.

### Round 1 — 수정 방향
내일 환각: 후처리 검증. 오늘 잘림: 토큰 늘리기+디코딩 방어. 통합: 표는 항상 코드로 출력(둘 다 근본 차단).
</details>
