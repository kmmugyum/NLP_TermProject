# 로컬 세션 통합 프롬프트 — zip 제작 → Colab 실행 → 검증 → main 병합 (제출 마지막 단계)

너는 **로컬 맥**(GPU 없음, Google Drive 업로드 가능) 세션이다.
연구서버가 (1) 학사 환각 수정 v2 인덱스와 (2) py3.10.12/torch2.5.1 + Colab 환경호환 수정을
모두 끝내 `fix/env-compat-py310-torch251` 브랜치(repo `kmmugyum/NLP_TermProject`, gh=kmmugyum)에
push해뒀다. **아직 main 미병합.** 너의 일은 이 브랜치로 제출 zip을 만들어 Colab T4에서 실제
실행해 **(A) 에러 없이 동작 + (B) 학사 환각 사라짐 + (C) 기존 기능 회귀 없음**을 확인하고,
통과하면 main으로 병합하는 것이다. 이게 제출 직전 마지막 단계다.

## 배경 (연구서버가 끝낸 것 — 사실)
- **v2 인덱스**: 라이브 인덱스의 완전중복44.6%+표 빈파이프 깨짐(미적분학=2학년류 환각 원인)을
  79개 학과 와이드크롤 재청킹으로 정비 → 11,425청크(중복0/빈파이프0/<200자0), KURE-v1 빌드.
  repo 루트의 `academic_v2_bin.zip`(43.5MB)에 들어있음.
- **환경호환 수정**: ①requirements에 `torch==2.5.1 ; python_version<"3.11"`(py3.10 베이스 명시
  설치=PDF명세, Colab 3.12는 마커 비활성+사전설치 torch 유지) + lxml+PyMuPDF+olefile
  ②`ensure_academic_index()` 자립 부트스트랩  ③file_extractor fitz/olefile graceful 가드
  ④classifier.ipynb/ui_demo.ipynb 의존성 게이트에 lxml  ⑤chatbot.sh/realtime_chatbot.sh:
  setsid+하트비트(Colab 터미널 끊김에도 deps 설치 완주) + 단계 로깅(`▶ STEP n` 배너 +
  `outputs/run_*.log` 자동 기록)  ⑥학식 캐시 버그 수정(주간 캐시가 크롤 1일 뒤 통째로 막혀
  '오늘 학식'이 차단되던 것 → 날짜키 존재 여부로 판단)  ⑦classifier 멈춤 수정(orchestrator
  빌드 시 startup 라이브 크롤이 Colab 해외 IP에서 매달리던 것 제거 → 식단 질의 시에만 lazy 크롤)
  ⑧HF 캐시를 로컬 `/content/hf_cache`로(Drive FUSE에 2.3GB+ 쓰며 멈추던 것 회피·빠름) +
  `_build_orch`에 `[orch] ①~⑦` 단계 로그(모델 다운로드/로딩 진행 가시화). 연구서버
  py3.10.12/torch2.5.1/pl2.4.0 실측 통과. 상세 `VERIFY_ENV_REPORT.md`.
- **출력 품질·강건성 수정(deep-dive, writer↔critic 검증)**: ⑨답변 자연스러움(외국문자·중복·장황 정리)
  ⑩날짜 grounding(현재날짜 KST 주입 → "오늘 며칠"·"다음학기 몇 달" 환각 제거) ⑪모호/빈 질문 되물음
  ⑫메타태그 누수 제거 ⑬라우팅 일반화(멀티intent·도서관 운영시간·도발 방어·한영혼합·공감). 회귀 0
  실측. 상세 `OUTPUT_NATURALNESS_REPORT.md`·`.omc/specs/deep-dive-date-robustness.md`.
- **수동 swap 불필요**: Colab 첫 진입점 실행 시 `ensure_academic_index()`가 zip 안
  `academic_v2_bin.zip`을 자동 해제→`academic_real.bin`(11,425벡터) 생성. 즉 **v2 인덱스가
  zip만으로 자동 적용**된다(과거 수동 swap 가이드는 불필요해짐).

## STEP 1 — fix 브랜치 받기
```bash
cd /tmp && rm -rf ntp_sub && gh repo clone kmmugyum/NLP_TermProject ntp_sub -- -b fix/env-compat-py310-torch251 --depth 1
cd ntp_sub && git log -1 --format='%h %s'      # d483fdc(또는 이후) 확인
ls academic_v2_bin.zip src/classifier.ipynb chatbot.sh requirements.txt   # 존재 확인
```

## STEP 2 — 제출용 zip 제작 (git archive, 검증된 방식)
> 노트북 Cell 0 은 Drive 의 `Termproject_*.zip` 을 찾아 풀고 **첫 하위 폴더**를 PROJECT_ROOT 로
> 잡는다 → zip 내부는 `Termproject_김무겸/` 폴더 하나로 감싼다. `git archive` 가 추적 파일만 담아
> `.git`/`.venv`/`__pycache__`/캐시를 자동 제외하고 `academic_v2_bin.zip`·`outputs/.gitkeep` 은 포함.
```bash
cd /tmp/ntp_sub
PKG="Termproject_김무겸"
git archive --format=zip --prefix="$PKG/" -o "/tmp/${PKG}_final.zip" HEAD
ls -lah "/tmp/${PKG}_final.zip"          # ~52MB 예상
python3 -c "
import zipfile; z=zipfile.ZipFile('/tmp/${PKG}_final.zip'); n=z.namelist()
print('최상위:', sorted(set(x.split('/')[0] for x in n)))   # ['Termproject_김무겸']
for f in ['$PKG/chatbot.sh','$PKG/src/classifier.ipynb','$PKG/academic_v2_bin.zip','$PKG/requirements.txt']:
    print('OK' if f in n else 'MISSING', f)
"
```
> (연구서버에서 이 방식으로 52MB·65항목·구조 정상 검증 완료. 맥 기본 git 으로 동일 동작.)

## STEP 3 — Google Drive 업로드 + 노트북 열기
- `/tmp/Termproject_김무겸_final.zip` 을 브라우저로 **MyDrive 최상위**에 업로드
  (→ `/content/drive/MyDrive/Termproject_김무겸_final.zip`. 노트북이 이 패턴 자동 탐지).
- Colab 에서 노트북 열기: Colab → 파일 → 노트북 열기 → GitHub → `kmmugyum/NLP_TermProject`,
  브랜치 `fix/env-compat-py310-torch251`, `src/classifier.ipynb`.
- 런타임 유형 = **T4 GPU** 확인.

## STEP 4 — Colab T4 실행·검증 (실제 출력으로 판정, 추측 금지)

> **실행 순서 무관**: classifier.ipynb·chatbot.sh·realtime_chatbot.sh 중 무엇을 먼저 돌려도 됨.
> 두 `.sh`의 첫 실행 시 deps 설치(~3~5분)는 **setsid+하트비트**로 처리되어 Colab 터미널이
> 끊겨도(SIGHUP) 죽지 않고 백그라운드서 완주한다. 혹시 '.' 진행 중 화면이 끊기면 **같은 명령을
> 다시 실행**하면 됨(이미 설치된 건 건너뛰고 즉시 다음 단계로). 첫 모델 다운로드(Qwen 7B)는
> Drive 캐시(`hf_cache`)에 받혀 재실행 시 재다운로드 없음.

### (A) 에러 없이 동작
1. **classifier.ipynb** 셀 순서 실행 → Cell 0(Drive마운트·zip해제·deps설치, 첫 실행은
   Qwen 7B 다운로드로 수 분) → `outputs/cls_output.json` 생성 + 최소합격선 셀 ✅
   (건수 일치·label int 0~4).
2. **chatbot.sh** — 새 셀에서:
   ```python
   %cd /content/workspace/Termproject_김무겸     # Cell 0 출력에 찍힌 PROJECT_ROOT
   !bash chatbot.sh
   ```
   기대: 서버 기동→/health ready→배치 50건→`outputs/chat_output.json` 생성, 에러 로그 없음.
   (academic_real.bin 자동 부트스트랩 로그가 떠야 함.)
   진행 단계는 화면의 `▶ STEP n | ...` 배너로 확인, 전체 로그는 `outputs/run_chatbot.log`에 기록됨.

### (B) 학사 환각 사라짐 (UI 또는 realtime 질의로 실제 답변 캡처)
3. `컴퓨터인공지능학부 미적분학은 몇 학년 과목이야?` → **1학년(1-1) 미적분학1**.
   "2학년"류 나오면 실패 → 병합 보류·롤백 보고.
4. `컴퓨터인공지능학부 1학년 교과목 알려줘` → 1-1/1-2 과목(컴퓨터프로그래밍1, 이산수학 등) 깨끗.
5. `충남대 졸업 이수학점 최소 몇 학점?` → 130학점.

### (C) 기존 기능 회귀 없음
6. `오늘 학식 뭐야` → 식단 정상(OOS 오거부 없음).
7. `컴퓨터인공지능학부 최근 공지` → 공지 목록 정상.
8. `성심당 어디야` / `오늘 날씨` → 여전히 거부(진짜 OOS).

### (D) 강건성/일반화 (deep-dive 수정 — 선택 확인)
9. `오늘 며칠이야?` → 현재 날짜로 답(환각 날짜 없음).
10. `다음 학기 개강 몇 달 남았어?` → 오늘 기준 기간 추정(과거 "2개월" 환각 없음).
11. `그거 언제까지였지?` / `?` → "질문이 모호해요…" 되물음(임의 답 안 함).
12. `도서관 주말 몇 시까지 열어?` → library.cnu 안내(OOS 거부 아님).
13. `나 우울한데 위로 좀` → 공감+학생상담센터. `너 학식 다 지어내잖아?` → 차분한 출처 설명(학식표 덤프 아님).
14. `융합전공은 모듈 단위로 끊어 들으면 마이크로디그리로 학점 교차 인정돼?` → 학사 답변/위임.
    (과거 '단위로'의 '위로' 부분일치로 상담센터 메시지가 잘못 뜨던 버그 수정 회귀확인 — f0a7444)

### (빠른 일반 질문 5개) — 데모 시 가장 먼저 던져볼 평이한 질문
> 환각·회귀·라우팅을 한 번에 훑는 일상 질문. 모두 자연스러운 학사 답변이 나와야 정상.
1. `충남대 졸업하려면 몇 학점 들어야 해?`            → 130학점.
2. `복수전공이랑 부전공 차이가 뭐야?`                → 목적/이수범위/학점(복수39·부전공24~30) 구분 설명.
3. `성적 장학금 받으려면 평점 몇 이상이어야 해?`      → 3.25 이상.
4. `휴학은 최대 몇 학기까지 할 수 있어?`             → 통산 6학기(3년)류.
5. `오늘 학식 뭐 나와?`                              → 오늘 날짜 식단(OOS 오거부 없이).

## STEP 5 — 판정 & main 병합
- **(A) 두 출력파일 무에러 생성 AND (B) 환각 3~5 통과 AND (C) 회귀 6~8 정상** → 통과.
- 통과 시 main 병합:
  ```bash
  cd /tmp/ntp_sub
  gh pr create --base main --head fix/env-compat-py310-torch251 \
    --title "env compat + v2 index: py3.10.12/torch2.5.1 + Colab T4 무에러·환각수정" \
    --body "VERIFY_ENV_REPORT.md 참조. Colab 실행 통과(분류·채팅 무에러, 미적분학=1학년 확인)."
  gh pr merge --merge --delete-branch
  ```
- 실패 시: 어느 항목(A/B/C)·어떤 에러인지 **실제 답변/로그 캡처**와 함께 repo에 피드백 push,
  main 병합 보류.

## 원칙
- 추측 금지 — Colab 실제 실행 로그·출력파일·답변 캡처로만 판정.
- zip 안에 `academic_v2_bin.zip` 포함 필수(self-bootstrap = v2 인덱스 자동 적용 트리거).
- (A)·(B)·(C) 전부 통과 전 main 병합 금지.
