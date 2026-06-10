# 로컬 세션 프롬프트 — 제출용 zip 제작 → Colab 업로드·실행 검증

너는 **로컬 맥**(GPU 없음, Drive 업로드 가능) 세션이다. 연구서버가 환경호환 수정을
`fix/env-compat-py310-torch251` 브랜치에 push해뒀다(아직 main 미병합). 너의 일은
**그 브랜치 코드로 Colab 제출용 zip을 만들어 Drive에 올리고, Colab T4에서 실제 실행해
무에러를 확인**하는 것이다. 통과하면 main 병합. repo: `kmmugyum/NLP_TermProject`(gh=kmmugyum).

## 배경 (연구서버가 끝낸 것)
- py3.10.12/torch2.5.1/pl2.4.0 클린환경에서 검증·수정 완료. 수정 4건:
  ①requirements에 lxml+PyMuPDF+olefile  ②`ensure_academic_index()` 자립부트스트랩(zip→academic_real.bin)
  ③file_extractor fitz/olefile graceful 가드  ④classifier.ipynb/ui_demo.ipynb 의존성 게이트에 lxml.
- 상세: 브랜치의 `VERIFY_ENV_REPORT.md`. pl은 코드 미사용.
- **핵심**: repo의 `academic_v2_bin.zip`(46MB)이 zip 안에 포함되면, Colab에서 진입점 첫 실행 시
  `ensure_academic_index()`가 자동 해제→`academic_real.bin` 생성(수동 swap 불필요).

## STEP 1 — fix 브랜치 받기 (수정본 확보)
```bash
cd /tmp && rm -rf ntp_zip && gh repo clone kmmugyum/NLP_TermProject ntp_zip -- -b fix/env-compat-py310-torch251 --depth 1
cd ntp_zip && git log -1 --format='%h %s'      # 813db99 ... 확인
ls academic_v2_bin.zip src/classifier.ipynb chatbot.sh requirements.txt   # 존재 확인
```

## STEP 2 — 제출용 zip 제작 (git archive, 검증된 방식)
> 노트북 Cell 0 은 Drive 에서 `Termproject_*.zip` 을 찾아 풀고, **압축 안의 첫 하위 폴더**를
> PROJECT_ROOT 로 잡는다. 따라서 zip 내부는 `Termproject_김무겸/` 폴더 하나로 감싼다.
> `git archive` 가 가장 견고 — 추적 파일만 담아 `.git`/`.venv`/`__pycache__`/캐시를 자동 제외하고,
> `academic_v2_bin.zip`(추적됨)과 `outputs/.gitkeep`(출력폴더 보존)은 포함한다. rsync 불필요.
```bash
cd /tmp/ntp_zip
PKG="Termproject_김무겸"
git archive --format=zip --prefix="$PKG/" -o "/tmp/${PKG}_final.zip" HEAD
ls -lah "/tmp/${PKG}_final.zip"          # ~52MB 예상
# 검증: 단일 최상위 폴더 + 핵심 파일 + academic_v2_bin.zip 포함
python3 -c "
import zipfile; z=zipfile.ZipFile('/tmp/${PKG}_final.zip'); n=z.namelist()
print('최상위:', sorted(set(x.split('/')[0] for x in n)))
for f in ['$PKG/chatbot.sh','$PKG/src/classifier.ipynb','$PKG/academic_v2_bin.zip','$PKG/requirements.txt']:
    print('OK' if f in n else 'MISSING', f)
"
```
> (연구서버에서 위 git archive 방식으로 52MB·65항목·구조 정상 검증 완료. 맥 기본 git 으로 동일 동작.)

## STEP 3 — Google Drive 업로드
- `/tmp/Termproject_김무겸_final.zip` 을 브라우저로 **MyDrive 최상위**에 업로드
  (경로: `/content/drive/MyDrive/Termproject_김무겸_final.zip`). 노트북이 이 패턴을 자동 탐지.
- Colab에서 열 노트북도 준비: `src/classifier.ipynb` 를 Colab에 업로드하거나,
  GitHub fix 브랜치에서 바로 열기(Colab → 파일 → 노트북 열기 → GitHub → `kmmugyum/NLP_TermProject`,
  브랜치 `fix/env-compat-py310-torch251`, `src/classifier.ipynb`).

## STEP 4 — Colab T4에서 실행·검증
런타임 유형 = **T4 GPU** 확인 후:

### (1) 분류 진입점 — classifier.ipynb
- 셀 순서대로 실행. Cell 0이 Drive 마운트→zip 탐지·해제→PROJECT_ROOT 설정→deps 설치
  (faiss/bnb/lxml 등 ~1~2분, 첫 실행은 Qwen 7B 다운로드로 더 걸림).
- 기대: `cls_output.json` 생성, 최소합격선 셀이 ✅ (건수 일치·label int 0~4).

### (2) 채팅 진입점 — chatbot.sh
- 새 셀에서:
```python
%cd /content/workspace/Termproject_김무겸     # Cell 0이 푼 PROJECT_ROOT (출력에 찍힘)
!bash chatbot.sh
```
- 기대: 서버 기동→/health ready→배치 50건→`outputs/chat_output.json` 생성, 에러 로그 없음.
  (academic_real.bin 없으면 자동 부트스트랩 로그가 뜨고 진행돼야 함.)

### (3) 스모크 질문 (선택, UI/realtime)
- `컴퓨터인공지능학부 미적분학은 몇 학년?` → **1학년(1-1)** 답이면 인덱스도 정상.

## STEP 5 — 판정 & main 병합
- classifier `cls_output.json` + chatbot `chat_output.json` 둘 다 **에러 없이 생성**되면 통과.
- 통과 시 main 병합:
```bash
cd /tmp/ntp_zip
gh pr create --base main --head fix/env-compat-py310-torch251 \
  --title "env compat: py3.10.12/torch2.5.1 + Colab T4 무에러" --body "VERIFY_ENV_REPORT.md 참조. Colab 실행 통과."
gh pr merge --merge --delete-branch   # 또는 GitHub에서 머지
```
- 실패 시: 어느 진입점·어떤 에러인지(로그 캡처) repo에 피드백 push, main 병합 보류.

## 원칙
- 추측 금지 — Colab 실제 실행 로그/출력파일로 판정.
- zip 안에 `academic_v2_bin.zip` 포함 필수(자립 부트스트랩 트리거).
- 확정(Colab 통과) 전 main 병합 금지.
