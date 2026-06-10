# 환경 검증 + 수정 리포트 (py3.10.12 / torch2.5.1 / pl2.4.0 + Colab T4)

작성: 2026-06-10, 연구서버(GPU). 브랜치: `fix/env-compat-py310-torch251` (main 병합 전 검증용).

## 목적
GitHub `NLP_TermProject`(CNU QA bot)가 **(A) python 3.10.12 + torch 2.5.1 + pytorch-lightning 2.4.0**
베이스와 **(B) Colab Free T4** 양쪽에서 모든 진입점이 에러 없이 동작하는지 검증하고 수정.

## 검증 환경 (실측)
- `uv`로 python **3.10.12** 정확히 설치 → 클린 venv
- `torch==2.5.1` (+cu124), `pytorch-lightning==2.4.0` 명시 설치 후 `requirements.txt` 설치
- GPU: RTX A5000(24GB), Qwen2.5-7B-Instruct 4bit(bitsandbytes 0.49.2) + KURE-v1 캐시 사용
- pytorch-lightning은 **코드 미사용**(import 0건, grep 확인) — 환경 존재만 보장, 충돌 없음

## 발견된 에러 → 수정
| # | 증상 | 원인 | 수정 |
|---|---|---|---|
| 1 | `FeatureNotFound: ... lxml` (학식/공지 크롤 실패) | `BeautifulSoup(..., "lxml")` 13곳 사용하나 `requirements.txt`에 `lxml` 없음(Colab은 사전설치라 미발견) | requirements.txt에 `lxml` 추가 |
| 2 | fresh `git clone`에서 인덱스 없음 → `FileNotFoundError: academic_real.bin` | 실제 벡터 `.bin`은 `.gitignore`, repo엔 `academic_v2_bin.zip`만 | `module4_api.ensure_academic_index()` 자립 부트스트랩 추가(zip 자동해제→`academic_real.bin`+meta). 모든 진입점이 거치는 `chat_model._build_orch`에서 호출 |
| 3 | PDF/HWP 첨부 추출 경로에서 `ImportError`(fitz/olefile) 잠재 크래시 | `import fitz`가 try 밖 + 미명시 | `file_extractor`의 import를 graceful 가드(미설치 시 `[PDF_ERROR:no_engine]` 반환, 크래시 X) + requirements에 `PyMuPDF`/`olefile` 명시 |
| 4 | 클린 Colab에서 lxml 누락 시 동일 크래시 | 노트북 의존성 게이트가 lxml 미점검 | `classifier.ipynb`/`ui_demo.ipynb` 의존성 import 게이트에 `lxml` 추가 → 누락 시 requirements 설치 트리거 |

## Acceptance Criteria 결과
| AC | 내용 | 결과 | 증거 |
|---|---|---|---|
| AC1 | py3.10.12+torch2.5.1+pl2.4.0 설치·전모듈 import | ✅ | 19개 cnubot 모듈+진입점 import 0실패, torch 2.5.1+cu124, bnb 0.49.2, CUDA 2장 |
| AC2 | classifier.ipynb → cls_output.json | ✅ | `classify_batch` 100건 무에러(43s), 라벨 0–4 정수, 분포 균형 |
| AC3 | chatbot.sh 서버+배치+realtime+UI | ✅ | 서버 29s 기동, /health 200, 배치 50/50→chat_output.json, `POST /api/v1/cnu-bot/chat` 정답, `GET /` chat.html 200, realtime_output.json append |
| AC4 | fresh clone 자립 부트스트랩 | ✅ | zip 자동해제→`academic_real.bin`(46.8MB), faiss ntotal=meta=11,425, dim 1024 |
| AC5 | Colab T4 동작 + 노트북 정비 | ✅(코드/노트북 정비) | 두 노트북 lxml 게이트 패치, self-bootstrap이 `/content` 탐색, torch 미고정으로 Colab 사전설치본 유지(2-타깃 호환). ※ Colab 실런타임 실행은 사용자측 확인 필요 |
| AC6 | 에러 수정 + 증거 | ✅ | 본 리포트 |

## 2-타깃 torch 전략 (충돌 없음)
- `requirements.txt`는 torch를 **고정하지 않음** → 각 환경의 적합 torch 사용:
  - 베이스(py3.10.12): 외부 제공 torch 2.5.1 유지(설치가 건드리지 않음, 19패키지 충돌 0 확인).
  - Colab(py3.12+cu128): 사전설치 torch 유지, `chatbot.sh`/노트북이 `torchao`/`torchcodec` 선제 제거.

## 보너스
realtime 답변에서 "컴퓨터인공지능학부 미적분학 = 1학년(1-1 미적분학1)" 정답 + intent=academic +
computer.cnu 출처 → v2 인덱스 환각수정이 라이브로 동작함을 부수 확인.

## 남은 1건(사용자 확인 필요)
Colab **실 런타임** 실행은 연구서버에서 대리 불가. 위 정비로 코드/노트북은 Colab-ready이며,
2-타깃 torch 전략·self-bootstrap·lxml 게이트로 클린 Colab에서도 동작하도록 했음.
최종 제출 전 Colab T4에서 `classifier.ipynb` + `chatbot.sh` 1회 실행 권장.
