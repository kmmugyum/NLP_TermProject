#!/usr/bin/env bash
# Campus ChatBot — 평가용 진입점
#   1) 의존성 설치
#   2) 서버 실행 (배치 모드)
#   3) 배치 추론: data/test_chat.json → outputs/chat_output.json
#   4) 배치 완료 후 Web UI에서 실시간 채팅 가능 → outputs/realtime_output.json 자동 저장
#
# Colab 환경 자동 감지: ngrok 없이 localtunnel 사용
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR/src"
mkdir -p "$SCRIPT_DIR/outputs"

# ============================================================
# 의존성 자동 셋업 (Colab Free / 로컬 양쪽 모두 안전)
# ============================================================
pip install -q -U pip setuptools wheel >/dev/null 2>&1 || true
pip uninstall -y -q torchao torchcodec >/dev/null 2>&1 || true

if ! python -c "import fastapi, uvicorn, transformers, faiss, bitsandbytes, sentence_transformers, peft, accelerate" >/dev/null 2>&1; then
    echo "[deps] 의존성 미설치/불완전 — pip install 진행 (~3~5분)"
    pip install -q --no-cache-dir --upgrade-strategy only-if-needed --prefer-binary \
        -r "$SCRIPT_DIR/requirements.txt" || {
        echo "[deps] 1차 설치 실패 — force-reinstall 재시도"
        pip install -q --no-cache-dir --force-reinstall \
            -r "$SCRIPT_DIR/requirements.txt"
    }
fi

if ! python -c "import bitsandbytes" >/dev/null 2>&1; then
    echo "[deps] bitsandbytes 단독 재시도 (Colab PyPI 미러 동기화 이슈)"
    pip install -q --no-cache-dir --force-reinstall "bitsandbytes>=0.46.1"
fi

python -c "import torch, transformers, bitsandbytes, sentence_transformers, faiss" || {
    echo "[FATAL] 의존성 검증 실패. 런타임 재시작 후 다시 실행하세요."
    exit 1
}
echo "[deps] OK"

# ============================================================
# HF 모델 캐시를 Google Drive 로 영속화 (재시작 시 재다운로드 방지)
#   Colab 은 세션 종료 시 ~/.cache 를 날리므로, Drive 에 캐시를 두면
#   7B 모델(~15GB)을 매 재시작마다 다시 받지 않는다.
#   classifier.ipynb / ui_demo.ipynb / realtime_chatbot.sh 와 동일 경로 공유.
# ============================================================
if [ -d "/content/drive/MyDrive" ]; then
    export HF_HOME="/content/drive/MyDrive/hf_cache"
    export HUGGINGFACE_HUB_CACHE="/content/drive/MyDrive/hf_cache/hub"
    mkdir -p "$HUGGINGFACE_HUB_CACHE"
    echo "[hf-cache] Drive 캐시 사용: $HF_HOME (재다운로드 방지)"
else
    echo "[hf-cache] Drive 미마운트 — 기본 캐시(~/.cache) 사용. 재시작 시 재다운로드 가능."
fi

# ============================================================
# GitHub 데이터 모드 (Colab 전용) — CNU 라이브 fetch 차단(504 회피)
#   Colab(미국 IP)은 *.cnu.ac.kr 접속 시 504 → 한국 연구서버가 크롤해 GitHub 에
#   올린 학식·공지 JSON 을 raw 로 읽는다. 라이브 크롤 0, 데이터 없으면 '정보 없음'.
#   Colab 감지 = Drive 마운트. 연구서버/로컬에선 미설정 → 기존 라이브 크롤 유지.
# ============================================================
if [ -d "/content/drive/MyDrive" ]; then
    export CNU_DATA_REPO="https://raw.githubusercontent.com/kmmugyum/NLP_TermProject/main"
    echo "[github-data] GitHub 데이터 모드 ON: $CNU_DATA_REPO (CNU 라이브 fetch 차단)"
    # 서버 시작 시 최신 학식·공지 JSON 을 github_data 캐시 디렉토리에 미리 받아둠
    # → 첫 질의 시 fetch 지연 제거. retriever 는 런타임에 이 캐시를 읽는다(CNU_DATA_LOCAL).
    # 실패해도 무관: github_data 가 런타임에 다시 raw fetch 시도(이중 안전망).
    export CNU_DATA_LOCAL="$SCRIPT_DIR/.cnu_data_cache"
    mkdir -p "$CNU_DATA_LOCAL"
    for f in meal_cache notice_cache; do
        if curl -fsSL "$CNU_DATA_REPO/data/$f.json" -o "$CNU_DATA_LOCAL/$f.json.tmp" 2>/dev/null; then
            mv "$CNU_DATA_LOCAL/$f.json.tmp" "$CNU_DATA_LOCAL/$f.json"
            echo "[github-data] $f.json 사전 캐시 완료"
        else
            rm -f "$CNU_DATA_LOCAL/$f.json.tmp"
            echo "[github-data] $f.json 사전 캐시 실패 — 런타임에 재시도됨"
        fi
    done
else
    echo "[github-data] GitHub 모드 OFF (Colab 아님) — 라이브 크롤 사용"
fi

# ============================================================
# 서버 실행 (배치 모드 ON)
# ============================================================
export CNU_BATCH_MODE=1
echo "[server] 서버 시작 (배치 모드)..."
python -m uvicorn server:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# 서버 준비 대기 — /health 의 ready:true(모델 로딩 완료)까지 대기.
echo "[server] 모델 로딩 대기 중... (최대 수 분)"
for i in $(seq 1 600); do
    if curl -s http://localhost:8000/health 2>/dev/null | grep -q '"ready": *true'; then
        echo "[server] 서버 준비 완료 (${i}초) — 모델 로딩 완료"
        break
    fi
    sleep 1
done

# ============================================================
# Colab 환경 감지 → URL 출력
# ============================================================
if [ -n "${COLAB_RELEASE_TAG:-}" ] || [ -d "/content" ]; then
    COLAB_URL=$(python -c "
from google.colab.output import eval_js
print(eval_js('google.colab.kernel.proxyPort(8000)'))
" 2>/dev/null || echo "")
    echo ""
    if [ -n "$COLAB_URL" ]; then
        echo "  ✅ Web UI: $COLAB_URL"
    else
        echo "  새 셀에서 실행: from google.colab.output import eval_js; print(eval_js('google.colab.kernel.proxyPort(8000)'))"
    fi
    echo ""
else
    echo "[server] Web UI: http://localhost:8000"
fi

# ============================================================
# 배치 추론: test_chat.json → chat_output.json
# ============================================================
echo "[batch] 배치 추론 시작: data/test_chat.json → outputs/chat_output.json"
curl -s -X POST http://localhost:8000/api/v1/batch/start \
     -H "Content-Type: application/json" | python -m json.tool 2>/dev/null || true

# 배치 완료 대기
echo "[batch] 배치 진행 중... (Web UI에서 실시간 확인 가능)"
while true; do
    STATUS=$(curl -s http://localhost:8000/api/v1/batch/status 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || echo "")
    if [ "$STATUS" = "done" ] || [ "$STATUS" = "error" ]; then
        break
    fi
    sleep 5
done

echo "[batch] 배치 완료!"

# 배치 모드 해제 → 실시간 채팅 가능
export CNU_BATCH_MODE=0

echo ""
echo "============================================"
echo "  배치 추론 완료!"
echo "  Web UI에서 실시간 채팅이 가능합니다."
echo "  채팅 내용은 outputs/realtime_output.json 에 자동 저장됩니다."
echo "============================================"
echo ""

# 서버 유지 (Ctrl+C로 종료)
echo "[server] 서버 실행 중... (Ctrl+C로 종료)"
wait $SERVER_PID
