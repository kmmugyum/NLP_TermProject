# Colab v2 인덱스 swap + 환각 검증 절차

> 로컬 맥은 GPU 없어 봇 실행 불가 → Colab(GPU)에서 swap·E2E 검증.
> STEP A 무결성은 로컬에서 통과(zip bad:None, bin 46,796,845B, meta 11,425).
> 봇 로드 경로: `storage/academic_real.bin` + `.meta.json` (이 이름으로 덮어씀).

## 전제
- Colab에서 Drive 마운트 + 프로젝트 zip 압축 해제 후, 그 폴더에서 진행.
- 또는 `git clone`한 폴더 + 벡터를 별도로 넣은 상태.

## STEP A' — Colab에서 무결성 재확인 (셀)
```python
import zipfile, json
PROJ = "/content/Termproject_김무겸 (1) 4"   # 실제 경로로
z = zipfile.ZipFile(f"{PROJ}/academic_v2_bin.zip")
print("zip bad:", z.testzip())                       # None
print("bin 크기:", z.infolist()[0].file_size)        # 46796845
m = json.load(open(f"{PROJ}/src/cnubot/storage/academic_v2.bin.meta.json"))
print("meta entries:", len(m))                       # 11425
```

## STEP B — 라이브 백업 (롤백 보장, 생략 금지)
```python
import shutil, os
S = f"{PROJ}/src/cnubot/storage"
for f in ["academic_real.bin", "academic_real.bin.meta.json"]:
    src = f"{S}/{f}"
    if os.path.exists(src):
        shutil.copy2(src, f"{src}.bak_pre_v2")
        print("백업:", f"{f}.bak_pre_v2")
```

## STEP C — swap (봇 중지 상태에서)
```python
# v2 인덱스 압축 해제
import zipfile
zipfile.ZipFile(f"{PROJ}/academic_v2_bin.zip").extractall(f"{PROJ}/_v2tmp")
import shutil
# 인덱스 + meta 쌍으로 교체 (academic_real.bin 이름으로)
shutil.copy2(f"{PROJ}/_v2tmp/academic_v2.bin", f"{S}/academic_real.bin")
shutil.copy2(f"{PROJ}/src/cnubot/storage/academic_v2.bin.meta.json",
             f"{S}/academic_real.bin.meta.json")
# 청크 원본도 동기화
shutil.copy2(f"{PROJ}/src/cnubot/storage/_academic_chunks_v2.json",
             f"{S}/_academic_chunks.json")
# 검증: 벡터수 11,425 일치
import json
print("교체 후 meta:", len(json.load(open(f"{S}/academic_real.bin.meta.json"))))  # 11425
```

## STEP D — 봇 띄워 실제 답변으로 환각 검증 (핵심)
서버 실행:
```python
%cd "/content/Termproject_김무겸 (1) 4"
!bash realtime_chatbot.sh
```
UI 접속 후 아래 질문, **실제 답변 텍스트를 캡처**:

### 핵심 환각 (반드시 통과)
1. `컴퓨터인공지능학부 미적분학은 몇 학년 과목이야?`
   → **1학년(1-1) 미적분학1**. "2학년"류 나오면 실패 → STEP F 롤백.
2. `컴퓨터인공지능학부 1학년 교과목 알려줘`
   → 1-1/1-2 과목(컴퓨터프로그래밍1, 이산수학 등) 깨끗하게.
3. `충남대 졸업 이수학점 최소 몇 학점?` → 130학점.

### 회귀 가드 (기존 기능 안 깨졌나)
4. `오늘 학식 뭐야` → 정상 식단표(OOS 오거부 없음).
5. `컴퓨터인공지능학부 최근 공지` → 공지 목록 정상.
6. `성심당 어디야` / `오늘 날씨` → 여전히 거부(진짜 OOS).

## STEP E — 제출 (D 전부 통과 시에만)
- 환각 1~3 통과 + 회귀 4~6 정상이면 제출.
- `*.bak_pre_v2`는 제출 후에도 보존.

## STEP F — 실패 시 즉시 롤백
```python
import shutil
for f in ["academic_real.bin", "academic_real.bin.meta.json"]:
    shutil.copy2(f"{S}/{f}.bak_pre_v2", f"{S}/{f}")
print("롤백 완료 — 라이브 원복")
```
실패 내용(질문·실제답변·기대답변)을 repo에 피드백 push.

## 원칙
- 추측 금지. 모든 판정은 실제 봇 답변 캡처로.
- swap은 인덱스+meta 쌍, 벡터수 11,425 일치 확인.
- 제출은 환각 통과 AND 회귀 0일 때만.
