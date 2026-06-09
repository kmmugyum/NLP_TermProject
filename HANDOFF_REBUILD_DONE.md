# HANDOFF — v2 인덱스 재빌드 완료 (연구서버 → 로컬)

작성: 2026-06-09 19:41 (연구서버, 한국IP, GPU cuda:1)
상태: **재빌드·eval 통과 → repo push 완료. 로컬에서 라이브 인덱스 swap 대기.**

## 한 줄 요약
학사 환각의 직접 원인이던 라이브 인덱스(완전중복 44.6% + 교과과정 표 빈파이프 깨짐)를
**중복 0 / 빈파이프 0 / <200자 0** 의 v2 코퍼스로 재정비하고, 79개 학과 와이드 크롤로
커버리지를 보강한 새 KURE-v1 인덱스(`academic_v2.bin`)를 빌드했다. retrieval eval에서
미적분학 학년·1학년 교과목 질문이 OLD=⚠깨짐 → NEW=ok 로 개선됨을 확인했다.

## 산출물 (repo push 됨)
- `src/cnubot/storage/_academic_chunks_v2.json` — 11,425청크 (와이드크롤 반영)
- `src/cnubot/storage/academic_v2.bin.meta.json` — 메타(11,425, embed_dim 1024)
- `academic_v2_bin.zip` — 인덱스 본체 zip (원본 .bin 46.8MB)
- `RECHUNK_V2_REPORT.md`, 본 `HANDOFF_REBUILD_DONE.md`

## 실측 수치 (직접 측정, 추측 아님)
### 크롤 (STEP 0)
- 79/79 학과 도메인, 3,313쪽, 2,036초, skip 0

### 재청킹 (STEP 1) — `rechunk_v2.py "*_out.jsonl" "wide_dumps/*.jsonl"`
- 입력 104파일 / 12,713 raw 레코드 → **출력 11,425청크**
- drop 내역: page_noise 8,057 / page_dup 1,435 / chunk_dup 385 / short 476 / junk_table_rows 19,639

### 품질 검증 (STEP 2) — 독립 측정
- 청크 11,425 | **중복 0 | 빈파이프 0 | <200자 0** | 평균 984자 | 고유 URL 3,027

### 빌드 (STEP 3) — `build_index_v2.py` (KURE-v1, cuda:1)
- ok=true, num_chunks 11,425, num_skipped 0, embed_dim 1024, 466초
- `academic_v2.bin` 46.8MB / `.meta.json` 23MB
- ※ 라이브 `academic_real.bin`(23,642 vec)은 **미덮어씀** — 스테이징 academic_v2.bin로만 생성

### eval (STEP 4) — `eval_rebuild.py` OLD(23,642) vs NEW(11,425)
| 질문 | OLD | NEW |
|---|---|---|
| 미적분학 몇 학년 | 신소재과 오답 + ⚠깨짐 2건 | **[ok] 컴퓨터AI학부 미적분학1 = 1-1** |
| 1학년 교과목 | top만 ok, 2·3위 ⚠깨짐 | **[ok] 1-1/1-2 매핑 깨끗** |
| 졸업이수학점 | ok (130학점) | ok (130학점) |
| 졸업 전공학점 | ok | ok |
| 교환학생 자매대학 | 일반 페이지 | **[ok] 국제교류본부 교환학생 + 파견대학 표** |

## 로컬이 할 일 (라이브 swap)
1. repo pull → `academic_v2_bin.zip` unzip → `academic_v2.bin` 확보
2. 봇 라이브 인덱스를 `academic_real.bin` → `academic_v2.bin`(+meta)로 교체
   (반드시 기존 academic_real.bin 백업 후 swap)
3. 실제 봇 응답에서 "미적분학=2학년" 류 환각 재현 안 됨 확인 → 제출

## 미해결/주의
- v2는 게시판·공지 제외 정적 페이지 중심. 동적/공지성 질의 커버리지는 별도.
- NEW 벡터 수(11,425) < OLD(23,642): 중복·노이즈 제거 결과이며 정보 손실 아님(고유 URL 3,027 유지).
