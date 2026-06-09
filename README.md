# Termproject — Campus ChatBot (CNU)

충남대학교 재학생을 위한 AI 챗봇. Colab Free(T4 16GB)에서 단일 프로세스로 동작.

## 🛡 의존성 자동 복구

`chatbot.sh` 와 `classifier.ipynb` 의 첫 셀에 다음 안전망이 내장되어 있어,
조교가 별도 환경 셋업 없이 진입점만 실행해도 동작합니다:

| 단계 | 처리 |
|---|---|
| (a) pip 최신화 | `pip install -U pip setuptools wheel` |
| (b) Colab 사전설치 충돌 패키지 선제 제거 | `torchao`, `torchcodec` |
| (c) 1차 설치 (캐시 우회) | `--no-cache-dir --prefer-binary -r requirements.txt` |
| (d) 1차 실패 시 강제 재설치 | `--force-reinstall` |
| (e) `bitsandbytes` 단독 복구 | wheel sha256 mismatch 대비 |
| (f) 최종 sanity check | 누락 시 명시적 에러 |

각 단계는 idempotent — 재실행해도 안전.


> **chatbot.sh 동작**: foreground 로 test_chat 50건 추론 → `outputs/chat_output.json` 생성.
> (있으면) test_realtime 추론 → `outputs/realtime_output.json`. **UI 는 띄우지 않음** —
> UI 평가용 시연은 별도 영상으로 제출. UI 를 직접 띄워보고 싶으면
> `cd src && uvicorn server:app --port 8077` 로 별도 실행.

## 🎥 UI 시연 (영상 촬영용)

평가 진입점(`classifier.ipynb`, `chatbot.sh`) 과 별도로 UI 를 직접 띄우려면
**`ui_demo.ipynb`** 를 Colab 에서 순서대로 실행하세요.
1. Drive 마운트 + 의존성 자동 셋업
2. FastAPI 서버 백그라운드 실행 (`:8077`)
3. iframe 로 UI 표시 — 자유 질의 입력 시연 가능

## ⚠ 필수 환경

> **Google Colab Free 의 T4 GPU 런타임이 반드시 선택되어 있어야 합니다.**
> (런타임 → 런타임 유형 변경 → 하드웨어 가속기: **T4 GPU**)
>
> CPU only 런타임에서는 Qwen2.5-7B 4bit 양자화(bitsandbytes) 가 동작하지 않아
> `chatbot.sh` / `classifier.ipynb` 모두 실패합니다.

## 디렉터리

```
Termproject_김무겸/
├── data/                 # test_cls.json, test_chat.json, (test_realtime.json), train.json, valid.json
├── src/
│   ├── classifier.ipynb  # ★ 평가 진입점 — 5-way 분류 → outputs/cls_output.json
│   ├── chat_model.py     # LLM(Qwen2.5-7B 4bit) + RAG + 분기 핵심 모듈
│   ├── server.py         # FastAPI: chat.html mount + chat 엔드포인트
│   ├── chat.html         # ★ Chat UI (same-origin SSE)
│   ├── realtime_model.py # Optional: 실시간 정보 응답
│   └── cnubot/           # 내부 모듈 (인덱서·라우터·검색·생성·크롤러·NoticeService)
├── model/                # (자동 생성) Qwen 4bit + KURE 가중치 캐시
├── outputs/              # cls_output.json, chat_output.json, realtime_output.json
├── chatbot.sh            # ★ 평가 진입점 — chat 추론 + UI 런치
├── requirements.txt
└── README.md (이 파일)
```

## 평가 흐름

```bash
# 1) 의존성 설치 (Colab/로컬 공통)
pip install -r requirements.txt

# 2) 분류기 (40점)
jupyter nbconvert --to notebook --execute src/classifier.ipynb
# 결과: outputs/cls_output.json  ({question, label} 리스트)

# 3) 챗봇 + UI (60점)
bash chatbot.sh
# - outputs/chat_output.json 생성
# - (있으면) outputs/realtime_output.json 생성
# - http://0.0.0.0:8077 에 UI 노출
```

### Colab 셀 예시

```python
%cd /content/Termproject_김무겸
!pip install -q -r requirements.txt
# (a) 분류기
!jupyter nbconvert --to notebook --execute src/classifier.ipynb --output classifier_executed.ipynb
# (b) 챗봇
!bash chatbot.sh &
import time; time.sleep(60)  # 모델 로딩 대기
from google.colab.output import serve_kernel_port_as_window
serve_kernel_port_as_window(8077)
```

## 모델 / 데이터

### 모델 파일 다운로드 링크
zip 안에는 **가중치 미포함** (`model/.gitkeep` 만 존재). 아래 두 모델은 `chatbot.sh` / `classifier.ipynb` 첫 실행 시 `huggingface_hub` 가 자동 다운로드하여 `HF_HOME` 캐시에 저장합니다.

| 역할 | 모델 ID | 다운로드 URL | 크기 |
|---|---|---|---|
| LLM (분류·생성 공용) | `Qwen/Qwen2.5-7B-Instruct` | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct | ≈15 GB (4bit 양자화 후 ≈5 GB) |
| 임베더 (RAG) | `nlpai-lab/KURE-v1` | https://huggingface.co/nlpai-lab/KURE-v1 | ≈1 GB |

수동 사전 다운로드 (선택):
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir model/Qwen2.5-7B-Instruct
huggingface-cli download nlpai-lab/KURE-v1        --local-dir model/KURE-v1
```
Colab 권장: `HF_HOME=/content/drive/MyDrive/hf_cache` 설정 → 재실행 시 1~2분에 로드 (`classifier.ipynb` 셋업 셀에 자동 적용).

### 데이터 / 인덱스 (zip 동봉)
- **FAISS 인덱스**: `src/cnubot/storage/academic_real.bin` (학사요람 + 부속, 약 23k 청크 / 92 MB)
- **메타·청크**: `src/cnubot/storage/_academic_*.json`, `_attachments_extracted.json`
- **학식 캐시**: `src/cnubot/data/cnu_meal_mock.json` — 첫 질의 시 자동 크롤·핫스왑

## 분류기 (Task 1)

`src/chat_model.py:classify(q)` — Qwen2.5-7B Instruct 에 5-way 카테고리 시스템 프롬프트를 주입해 `0..4` 한 글자 출력.

- 0: 졸업요건
- 1: 학교 공지사항
- 2: 학사일정
- 3: 식단 안내
- 4: 통학/셔틀 버스

LLM 출력이 깨지면 키워드 휴리스틱 fallback.

## 챗봇 (Task 2)

`src/chat_model.py:respond(q)` → `cnubot.module4_api.Orchestrator.handle(q)` 호출:

1. **라우팅** (`CNUHybridIntentRouter`): academic / cafeteria / temporal_notice / out_of_scope
2. **검색** (`AcademicRetriever`): KURE 임베딩 + FAISS top-k, 학사요람 chunk 부스팅
3. **분기**:
   - 학사규정/정의 → `_DEFINITION_PREPEND_MAP` 강제 prepend
   - 외부 도메인 키워드 → `_STANDALONE_SITES` 즉시 URL 안내 (~120 entries)
   - 주관·외부정보·비공식 → `_REJECT_PATTERNS` 정중 거절
   - 학식 → `CafeteriaRetriever` (실시간 캐시 트리거)
   - 공지 → `NoticeService` 라이브 fetch
4. **생성** (`CNUGenerator`): 프롬프트 빌드 → Qwen → 한자 제거 등 후처리

## UI (Chat Interface)

`src/chat.html` — SSE 스트리밍·세션 connect/disconnect·인텐트 뱃지·한자 제거 후처리.
`server.py` 가 `/`(정적) + `/api/v1/*`(API) 를 같은 포트에서 같이 노출 → ngrok 불필요.

## 실시간 (Task 3, Optional)

`realtime_model.py` 가 `chat_model.respond` 를 그대로 호출. 학식 캐시 자동 갱신·공지 라이브 fetch·외부 도메인 URL 가이드가 모두 통합 응답에 반영됨.

## 알려진 제약

- 단일 GPU 환경: KURE 임베더와 LLM 이 같은 cuda:0 을 공유. T4 16GB 에서 약 7 GB 사용 (~9 GB 여유).
- 인터넷 차단된 평가 환경이면 학식·공지 자동 fetch 가 fail → 캐시된 마지막 식단/공지 반환.
- 첫 질의는 모델 warm-up 으로 30~60초 소요. 이후는 평균 5~15초.
