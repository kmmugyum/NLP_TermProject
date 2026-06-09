# 로컬 세션 이어받기 프롬프트 — v2 인덱스 라이브 swap & 제출 (마지막 단계)

너는 **로컬 봇 서버**(라이브 인덱스 직접 접근, 봇 실행 가능) 세션이다.
연구서버가 학사 환각 수정용 v2 인덱스 재빌드·eval·push를 **완료**했다. 너는 이걸 라이브에
반영하고 **실제 봇 응답으로 환각이 사라졌는지 검증한 뒤 제출**한다. 이게 제출 직전 마지막 단계다.
작업 dir: `/root/jupyter/LogitWave/다시 공부/experiment/CNU QA bot`

## 배경 (연구서버가 끝낸 것 — 추측 말고 사실)
- 라이브 `cnubot/storage/academic_real.bin`(23,642 vec)이 학사 환각의 직접 원인(완전중복+표 빈파이프 깨짐).
- 연구서버가 79개 학과 와이드 크롤 완주 후 재청킹 → **11,425청크(중복0/빈파이프0/<200자0)** →
  KURE-v1로 `academic_v2.bin`(46.8MB, dim1024) 빌드 → retrieval eval에서
  미적분학 학년·1학년 교과목이 OLD=⚠깨짐 → NEW=[ok](1-1 매핑 깨끗)로 개선 확인.
- repo `kmmugyum/NLP_TermProject` main(커밋 `a75a206`)에 push됨:
  `academic_v2_bin.zip`(43.5MB), `src/cnubot/storage/academic_v2.bin.meta.json`,
  `src/cnubot/storage/_academic_chunks_v2.json`, `RECHUNK_V2_REPORT.md`, `HANDOFF_REBUILD_DONE.md`.
- ※ 연구서버는 라이브 인덱스를 **건드리지 않았다**. swap은 너(로컬) 책임.

## STEP A — 아티팩트 수령 & 무결성
```bash
cd /tmp && rm -rf ntp2 && gh repo clone kmmugyum/NLP_TermProject ntp2 -- --depth 1
cd ntp2 && git log -1 --format='%h %s'        # a75a206 인지 확인
python3 -c "import zipfile; z=zipfile.ZipFile('academic_v2_bin.zip'); print('bad:',z.testzip()); print(z.infolist()[0].file_size)"  # bad:None, 46796845
mkdir -p extracted && python3 -c "import zipfile; zipfile.ZipFile('academic_v2_bin.zip').extractall('extracted')"
ls -la extracted/academic_v2.bin                # 46796845 bytes
# meta vec 수 == 11425 확인
python3 -c "import json; m=json.load(open('src/cnubot/storage/academic_v2.bin.meta.json')); print('meta entries:', len(m) if isinstance(m,list) else len(m.get('chunks',m)))"
```
무결성/카운트 안 맞으면 **중단**하고 연구서버에 repo로 피드백(아래 STEP F).

## STEP B — 라이브 백업 (롤백 보장. 절대 생략 금지)
```bash
cd "/root/jupyter/LogitWave/다시 공부/experiment/CNU QA bot/cnubot/storage"
cp -av academic_real.bin           academic_real.bin.bak_pre_v2
cp -av academic_real.bin.meta.json academic_real.bin.meta.json.bak_pre_v2
ls -la academic_real.bin.bak_pre_v2 academic_real.bin.meta.json.bak_pre_v2
```

## STEP C — swap (봇 중지 상태에서)
봇/게이트웨이가 떠 있으면 먼저 내려라(워커가 인덱스 mmap 잡고 있을 수 있음).
```bash
cd "/root/jupyter/LogitWave/다시 공부/experiment/CNU QA bot/cnubot/storage"
cp -av /tmp/ntp2/extracted/academic_v2.bin            academic_real.bin
cp -av /tmp/ntp2/src/cnubot/storage/academic_v2.bin.meta.json  academic_real.bin.meta.json
# (선택) 청크 원본도 동기화
cp -av /tmp/ntp2/src/cnubot/storage/_academic_chunks_v2.json   _academic_chunks.json
```
- 인덱스 본체와 meta는 **반드시 쌍으로** 교체(벡터수 23,642→11,425 불일치 시 로딩/매핑 깨짐).
- 로더가 파일명을 다르게 기대하면(예: `academic_v2.bin`로 직접 로드) 코드 경로 확인 후 그 이름으로.

## STEP D — 실제 봇 E2E 환각 검증 (retrieval 아님, **생성 답변**으로)
봇 서버 기동(`source env.sh && python -m uvicorn cnubot.module4_api:app --host 0.0.0.0 --port 8080`)
후, 실제 답변에서 환각이 사라졌는지 직접 확인. 최소 아래 4개 + 회귀 가드:

**(1) 핵심 환각 — 반드시 통과**
- "컴퓨터인공지능학부 미적분학은 몇 학년 과목이야?" → **1학년(1-1) 미적분학1**으로 답해야 함.
  과거 "2학년"류가 나오면 **실패** → STEP F.
- "컴퓨터인공지능학부 1학년 교과목 알려줘" → 1-1/1-2 과목(컴퓨터프로그래밍1, 이산수학 등) 깨끗하게.
- "충남대 졸업 이수학점 최소 몇 학점?" → 130학점.

**(2) 회귀 가드 — 기존 기능 안 깨졌나 (project_cnubot_freeze.md의 RC-1~4)**
- 학식: "오늘 학식 뭐야" 류 정상 응답(OOS 오거부 없음).
- 공지 RAG/OOS-rescue, 도메인 URL 폴백, 한자/가나 차단이 여전히 동작.
- 진짜 OOS(성심당/날씨)는 여전히 거부.

각 질문 **실제 답변 텍스트를 캡처**해 기록. "아마 될 것" 금지 — 직접 띄워서 본다.

## STEP E — 제출 (D 전부 통과 시에만)
- 환각 4개 통과 + 회귀 0 확인되면 제출 진행.
- 백업(`*.bak_pre_v2`)은 제출 후에도 보존(롤백용).
- 결과를 `SUBMIT_DONE.md`(또는 EXPERIMENT_LOG.md)에 실제 답변 캡처와 함께 기록.

## STEP F — 실패 시 (롤백 + 피드백)
```bash
cd "/root/jupyter/LogitWave/다시 공부/experiment/CNU QA bot/cnubot/storage"
cp -av academic_real.bin.bak_pre_v2           academic_real.bin
cp -av academic_real.bin.meta.json.bak_pre_v2 academic_real.bin.meta.json   # 즉시 라이브 원복
```
- 무엇이 어떻게 실패했는지(질문·실제답변·기대답변) repo에 피드백 문서로 push해 연구서버에 전달.

## 원칙
- 추측 금지. 모든 판정은 **실제 봇 답변 캡처**로. 백업 없이 swap 금지.
- swap은 인덱스+meta 쌍으로. 벡터수(11,425) 일치 확인.
- 제출은 환각 4개 통과 AND 회귀 0일 때만. 하나라도 미달이면 롤백 후 보류.
