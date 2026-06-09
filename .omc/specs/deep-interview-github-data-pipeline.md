# Deep Interview Spec: GitHub 경유 데이터 파이프라인 (live fetch 제거)

## Metadata
- Interview ID: github-data-pipeline-2026-06-09
- Rounds: 3 (+ Round 0 topology)
- Final Ambiguity Score: ~7%
- Type: brownfield
- Threshold: 0.2 (20%) / Source: default
- Status: PASSED

## 배경 (진단으로 확정된 근본 문제)
Colab 챗봇이 CNU 서버(computer.cnu.ac.kr, me.cnu.ac.kr 등)를 라이브 크롤하는데,
**Colab(미국 Google Cloud IP) → 한국 .ac.kr 서버 접속이 504 Gateway Timeout**으로 실패한다.
- 실측: 로컬(한국 IP) fetch 1.6초 성공 vs Colab fetch 78초 후 504
- 원인: (1) 미-한 국제 경로 지연 (2) CNU 방화벽의 클라우드 IP throttling
- 영향: 공지 100% 거절, 학사 `_plan` 최대 319초, 모든 라이브 fetch가 78초씩 낭비
- **우리 코드로 504 자체는 못 고침** → 크롤 주체를 한국 IP 머신으로 옮기는 게 근본 해결

## Topology (4 components)
| Component | Status | Description |
|-----------|--------|-------------|
| 크롤러 (producer) | active | 연구서버(RTX A5000, 한국 IP)가 학식·공지 크롤 → JSON 생성 |
| GitHub 저장소 (transport) | active | 크롤 결과를 단일 repo에 커밋 (데이터 운반 채널) |
| Colab 소비자 (consumer) | active | raw.githubusercontent로 데이터 fetch, CNU 라이브 크롤 전부 제거 |
| 갱신 스케줄 | active | 데이터별 주기: 학식·공지=매일 cron, 학사 벡터=수동/가끔 |

## Goal
CNU 서버 직접 크롤을 **한국 IP 연구서버로 이전**하고, 결과를 GitHub를 통해 Colab으로
운반한다. Colab 챗봇은 GitHub raw 파일을 읽어 답변하며, **CNU 서버에 대한 모든 라이브
fetch를 제거**해 504 의존을 0으로 만든다. 최신성은 하루 1회 크롤로 충족한다.

## 데이터 3종 처리 (코드 검증 완료)
| 데이터 | 현재 | 크기 | 새 방식 |
|--------|------|------|---------|
| 학식(meal) | JSON 캐시 + 라이브 크롤 | 6KB | 연구서버 크롤 → GitHub → Colab raw fetch |
| 공지(notice) | 캐시 없음, 온디맨드 라이브 | - | 연구서버 크롤 → JSON 신규 생성 → GitHub → raw fetch |
| 학사(academic) | faiss 벡터 + 청크 (정적) | 146MB | **zip 동봉 유지**(거의 불변). 라이브 보강 페이지 fetch만 제거 |

## Constraints
- 최신성: 하루 1회 크롤로 충분 (학식 메뉴·공지 모두 하루 지연 허용)
- 저장: 단일 GitHub repo
- 크롤 머신: 연구서버 RTX A5000 (한국 IP, cron 자동화)
- 읽기: Colab은 raw.githubusercontent로 **필요 파일(학식·공지 JSON)만** fetch (전체 clone 회피)
- 벡터(146MB): 거의 안 변하므로 기존 zip 배포 유지 (매일 커밋 불필요)
- CNU 서버 의존 0: 학식·공지·학사 보강 페이지 라이브 fetch 전부 제거

## Non-Goals
- 분 단위/실시간 최신성 (하루 지연 허용)
- 학사 벡터를 매일 재생성
- GitHub Actions에서 크롤 (해외 IP라 504 동일 위험 → 한국 연구서버 사용)
- intent 분류·답변 생성 로직 변경 (이미 별도 수정됨)

## Acceptance Criteria
- [ ] AC1: Colab 서버 로그에 504 / 78초 fetch가 0건
- [ ] AC2: "공지 알려줘" 질의가 거절 없이 답변 (GitHub 데이터 기반)
- [ ] AC3: 학사·학식 응답이 수초 내 완료 (라이브 fetch 대기 제거)
- [ ] AC4: 연구서버 크롤 스크립트가 학식·공지 JSON을 생성하고 git push
- [ ] AC5: cron이 하루 1회 크롤·커밋 자동 실행
- [ ] AC6: Colab이 시작 시 raw fetch로 최신 데이터 로드, GitHub 실패 시 동봉 데이터로 graceful fallback
- [ ] AC7: CNU 라이브 fetch 코드 경로가 비활성화(또는 환경변수로 off)

## 구현 방향
### 1. 크롤러 (연구서버)
- 기존 `meal_crawler.crawl_week()`, `notice.NoticeService.collect()` 재사용 (한국 IP라 정상 동작)
- 출력: `data/meal_cache.json`, `data/notice_cache.json` (학과별 공지 묶음)
- `cron`: 매일 새벽 1회 → 크롤 → git add/commit/push

### 2. GitHub repo
- 단일 repo. 학식·공지 JSON은 작아서 히스토리 부담 적음
- 벡터는 repo에 두지 않음(zip 동봉) → 히스토리 비대화 방지

### 3. Colab 소비자
- 시작 시 `raw.githubusercontent.com/<repo>/main/data/*.json` fetch → 로컬 캐시로 저장
- `CafeteriaRetriever`/`NoticeService`를 **GitHub 우선, 라이브 크롤 제거** 모드로 전환
  (환경변수 `CNU_DATA_SOURCE=github` 같은 플래그)
- raw fetch 실패 시 zip 동봉 캐시로 fallback

### 4. 스케줄
- 학식·공지: 연구서버 cron 매일
- 학사 벡터: 학기 단위 수동 재빌드 (필요 시)

## Technical Context
- 라이브 fetch 위치: `meal_crawler.fetch_meal_html`, `notice._fetch_board`/`fetch_body`,
  `module4_api._read_dept_relevant`/`_read_curriculum`/`_read_top_pages`
- `_net.py` monkey-patch: 크롤러(연구서버)에선 유지, Colab에선 우회
- 504 발생 지점: 모든 `httpx.get(*.cnu.ac.kr)` 호출

## Ontology
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| MealCache/WeeklyMealCache | core | timestamp, days, menus | 크롤러 생성 → GitHub → Colab |
| NoticeCache (신규) | core | dept, items, fetched_at | 크롤러 생성 → GitHub |
| GitHub repo | transport | data/*.json | producer push, consumer raw fetch |
| 크롤러 cron | producer | schedule, git push | 연구서버(한국 IP) |
| Colab consumer | consumer | raw fetch, fallback | GitHub 읽기, CNU 크롤 제거 |

## Interview Transcript
<details>
<summary>Q&A</summary>

Round 0: 4 컴포넌트 토폴로지 확정. 최신성 하루 1회 충분.
Round 1: 학사 벡터는 정적, 504는 보강 fetch 탓. 대상=학식+공지+벡터, 크롤=연구서버.
Round 2: 단일 repo, 갱신 주기 데이터별 다르게.
Round 3: Colab은 필요 파일만 raw fetch(벡터는 zip 동봉), CNU 크롤 전부 제거.
</details>
