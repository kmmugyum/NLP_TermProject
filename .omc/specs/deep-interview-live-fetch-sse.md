# Deep Interview Spec: Colab 라이브 fetch 강건화 + SSE 글자 렌더링

## Metadata
- Interview ID: live-fetch-sse-2026-06-09
- Rounds: 3
- Final Ambiguity Score: ~8%
- Type: brownfield
- Generated: 2026-06-09
- Threshold: 0.2 (20%)
- Threshold Source: default
- Initial Context Summarized: no
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.95 | 0.35 | 0.3325 |
| Constraint Clarity | 0.90 | 0.25 | 0.225 |
| Success Criteria | 0.92 | 0.25 | 0.230 |
| Context Clarity | 0.95 | 0.15 | 0.1425 |
| **Total Clarity** | | | **0.930** |
| **Ambiguity** | | | **0.07 (7%)** |

## Topology
| Component | Status | Description | Coverage / Deferral Note |
|-----------|--------|-------------|--------------------------|
| 학식 라이브 fetch 강건화 | active | "내일 학식 없음" fallback 문제 해결 | AC1~AC4 |
| SSE 글자 렌더링 | active | 답변이 통째로 뜨는 문제 → 글자 단위 타이핑 | AC5~AC6 |

## Goal
Colab 런타임에서 (1) "내일 학식" 질의 시 라이브 fetch가 실패해 6/1자 stale 캐시만 남아 fallback("없음")이 뜨는 문제를 **Colab→한국 서버 접속 강건화**로 해결하고, (2) SSE 응답이 Colab proxyPort 버퍼링으로 한꺼번에 뜨는 문제를 **클라이언트 측 글자 단위 타이핑 렌더링 강제**로 해결한다.

## 근본 원인 (데이터 검증 완료)
1. **fetch**: 로컬에서 `fetch_meal_html(target=내일)` 정상 동작 확인 (오늘·내일 모두 menus=9, target_date 정확). 코드 로직 무결. 디스크 캐시 `src/cnubot/data/cnu_meal_mock.json`의 timestamp는 **2026-06-01 (8일 전)**, days 키는 `[06-01, 06-02, 06-04, 06-05]`로 이번 주 날짜 전무. → Colab이 `mobileadmin.cnu.ac.kr` 접속에 실패해 라이브 크롤이 캐시를 갱신하지 못하고, stale 캐시에 없는 "내일" 날짜가 fallback 처리됨.
2. **SSE**: `server.py:138 /api/v1/cnu-bot/chat/stream`이 delta 청크를 흘려보내고, `chat.html:729` delta 핸들러가 `bodyEl.textContent = body`로 누적 렌더링하는 코드가 **이미 존재**. 그러나 Colab proxyPort가 `text/event-stream` 응답을 버퍼링해 청크가 한 번에 도착 → 통째로 표시.

## Constraints
- 실행 환경은 **Google Colab + proxyPort** (미국 리전, 한국 .ac.kr 접속 지연/차단 가능)
- fetch 재시도 총 60~90초 이내로 제한 (사용자 대기 시간 고려)
- 기존 `_net.py` monkey-patch 구조 유지 (httpx.get drop-in)
- fetch 최종 실패 시 **추측 메뉴 절대 표시 금지** — 명확한 안내만
- SSE 수정은 백엔드 프로토콜 변경 없이 **클라이언트 측 타이핑 효과**로 해결 (Colab 무관하게 항상 작동)

## Non-Goals
- 식단 데이터 소스(mobileadmin.cnu.ac.kr) 자체 변경
- 배치 모드(`/api/v1/batch/stream`) 수정
- 로컬/원격 서버 환경 대응 (Colab 전용 시연)

## Acceptance Criteria
- [ ] AC1: Colab에서 "내일 학식" 질의 시 실제 내일 날짜 메뉴가 표시됨 (stale fallback 아님)
- [ ] AC2: 서버 시작 시 라이브 크롤 시도/성공/실패가 로그에 명확히 출력됨 (`[meal]` 접두사, 실패 시 원인 포함)
- [ ] AC3: 시작 시 크롤이 실패해도 `retrieve()` 시점(`_auto_refresh_if_needed`)에 재시도하여 복구 가능 (3분 쿨다운은 유지하되 실패 로그 노출)
- [ ] AC4: 모든 재시도(총 60~90초) 후에도 실패하면 "현재 실시간 조회 불가, 홈페이지 확인" 안내 — 추측 메뉴 없음
- [ ] AC5: Colab proxyPort 버퍼링으로 SSE가 통째로 도착해도, 클라이언트가 받은 텍스트를 글자 단위로 타이핑 렌더링
- [ ] AC6: 타이핑 속도가 자연스러움 (너무 빠르거나 느리지 않음), 응답 완료 후 references 정상 표시

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| "live fetch 기능에 문제가 있다" (사용자 추측) | 로컬에서 실제 fetch 실행 | 코드는 정상, Colab 환경 접속 실패가 진짜 원인 |
| "SSE 렌더링을 적용하고 싶다" | chat.html 코드 검사 | SSE 렌더링 코드 이미 존재, Colab 프록시 버퍼링이 원인 |
| "내일만 안 됨 = 날짜 파싱 버그" | `_REL`/`resolve_target_date` 검사 | "내일"→+1 정상 매핑, 캐시에 날짜 없는 것이 원인 |
| stale 캐시면 자동 갱신될 것 | timestamp 확인 | 갱신 로직은 있으나 Colab fetch 실패로 미작동 |

## Technical Context
- **fetch 체인**: `Orchestrator.handle` → `CafeteriaRetriever.retrieve` (module3_retriever.py:441) → `_auto_refresh_if_needed` (392) → `crawl_week`/개별 `fetch_meal_html` → `_net.get` (monkey-patch, 3-stage retry 10/30/60s)
- **캐시 경로**: `MEAL_CACHE_PATH = src/cnubot/data/cnu_meal_mock.json` (module4_api.py:80)
- **stale 기준**: `stale_after_days=1` (module3_retriever.py:344)
- **SSE 백엔드**: server.py:138-174, `_SSE_CHUNK=4`, `_SSE_DELAY=0.03`
- **SSE 프론트**: chat.html:667-744, `fetch + getReader()`, delta 핸들러 729-736
- **검증 명령**: `CNU_NET_DISABLE=1 python3 -c "from cnubot.meal_crawler import fetch_meal_html, parse_meal_html, MEAL_URL; ..."` → 로컬 menus=9 확인됨

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| WeeklyMealCache | core domain | timestamp, week_start, days | CafeteriaRetriever가 보유 |
| MealCache | core domain | timestamp, target_date, menus | crawl_week이 일별 생성 |
| CafeteriaRetriever | core domain | cache_path, stale_after_days | retrieve/_auto_refresh |
| _net (fetch wrapper) | supporting | cache TTL, retry timeouts | httpx.get monkey-patch |
| SSE stream | external system | type=delta/status/meta/refs | server↔chat.html |
| Colab proxyPort | external system | 버퍼링 | SSE/fetch 양쪽 영향 |

## Interview Transcript
<details>
<summary>Full Q&A (3 rounds + Round 0)</summary>

### Round 0 — Topology
Q: 2개 컴포넌트(fetch 버그, SSE)로 봤는데 맞나? A: 둘 다 맞음. SSE 현상: 답변이 한꺼번에 뜸.

### Round 1 — 실행 환경
Q: 어디서 실행? A: Google Colab. 로그에 [meal] 크롤 메시지 없음, 서버 준비 완료만.

### Round 2 — 진단
로컬 fetch 테스트 → 오늘·내일 menus=9 정상. 캐시 timestamp 6/1 stale 확인. 근본 원인 = Colab 접속 실패.

### Round 3 — 수정 방향
Q: fetch 수정 방향? A: Colab 접속 강건화. Q: SSE 방향? A: 프론트 타이핑 효과 강제. Q: fetch 실패 UX? A: 재시도 후 명확한 안내.
</details>
