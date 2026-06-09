# 재청킹 v2 리포트 — 학사 코퍼스 청킹 결함 수정

> 생성: 연구서버(한국IP). 측정된 두 결함(완전중복 39.5%, 빈 파이프 표 9.5%) 직격 수정.
> 입력: 기존 raw 크롤 덤프(`*_out.jsonl`, May 24 정적 페이지 크롤본) — **재크롤 아님**.
> 사유: 측정된 문제는 크롤이 아니라 순수 청킹 버그(청크-dedup 누락 + 표 압축 누락)였음.
>       정적 안내페이지라 staleness 무방, confound 없는 최경량 경로.

## Before / After (동일 진단 스크립트)

| 지표 | BEFORE (prod `_academic_chunks.json`) | AFTER (`_academic_chunks_v2.json`) |
|---|---|---|
| 청크 수 | 29,794 | 6642 |
| **완전 중복** | **39.5%** (11,782) | **0.0%** (0) |
| **빈 파이프 표** | **9.5%** (2,832) | **0.0%** (0) |
| <200자 청크 | 512 | 0 |
| 평균 / 최대 길이 | 1054 / 2001 | 1029 / 2000 |
| 고유 source_url | 3,838 | 1289 |
| 표 포함 청크 | — | 1425 |

## 청크 수 감소(29,794→6642)는 정당함
- 완전중복 ~11,782 제거
- nav/사이드메뉴 줄 241,810줄 제거(예: library 사이드바 217회 반복 → 소거)
- 게시판/공지 페이지 13,034건 드롭(프롬프트 범위: cron이 담당, 정적안내만 청킹)
- 잡표행(빈 파이프) 13,937줄 제거, <200자 파편 187 드롭
- "손실"로 보인 URL 대부분 SNS공유(`sns/connect.do?kakao`)·도서검색(`search/detail`)·신청폼(`_prog/apply_pass`) = 정적안내 아님

## 길이 분포
- 200-500자: 395
- 500-1000자: 679
- 1000-1500자: 5486
- 1500-2000자: 82

## 상위 소스 도메인 (top 10)
- socio.cnu.ac.kr: 1466
- library.cnu.ac.kr: 457
- plus.cnu.ac.kr: 419
- welfare.cnu.ac.kr: 355
- cnuint.cnu.ac.kr: 247
- egc.cnu.ac.kr: 240
- cit-bk21.cnu.ac.kr: 220
- instivet.cnu.ac.kr: 217
- gsph.cnu.ac.kr: 193
- grad.cnu.ac.kr: 185

## 표 청크 샘플 (빈 파이프 없음 확인)

**[1] 수의과대학 | 학사안내 | 대학원**
```
* 이 자료는 통합정보시스템의 정보를 실시간으로 제공하고 있습니다.
| 전공 | 과정 | 교과번호 | 과목명 | 과목영문명 | 이수구분 | 학점/이론/실습 |
| 예방수의학 | 석사과정 |
2107-1001
연구윤리 | Research Ethics | 공통 | 0/0/0 |
| 예방수의학 | 박사과정 |
2107-3001
연구윤리2 | Research Ethics 2 | 공통 | 0/0/0 |
| 예방수의학 | 석박사통합과정 |
2391-5025
수의학세미나1 | Seminar of Veterinary Science 1 | 전공 | 3/3/0 |
| 예방수의학 | 석박사통합과정 |
2391-5048
시스템 수의학 개론 | Introduction of systems veterinary medicine | 전공 | 3/3/0 |
| 예방수의학 | 석박사통합과정 |
2391-5049
질환모델 동물학 
```

**[2] 수의과대학 | 학사안내 | 대학원**
```
수의학세미나2 | Seminar of Veterinary Science 2 | 전공 | 3/3/0 |
| 예방수의학 | 석박사통합과정 |
2391-5074
질병발생의 수의과학적 이해 | Veterinary medicinal basis for disease | 전공 | 3/3/0 |
| 예방수의학 | 석박사통합과정 |
2391-5075
기초연구를 위한 수의학 이론과 실습 | Theory and practice of veterinary medicine for basic research | 전공 | 3/3/0 |
| 예방수의학 | 석박사통합과정 |
2391-5076
논문연구 2 | Thesis Research 2 | 전공 | 3/3/0 |
| 예방수의학 | 석박사통합과정 |
2391-5077
대학원 논문 연구 방법론 | Methodology in thesis research for postgradua
```
