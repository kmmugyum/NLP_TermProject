"""CNU Campus ChatBot — FastAPI 서버 (UI 안에서 test_chat 배치 진행 표시)."""
from __future__ import annotations
import json
import os
import threading
import time
import traceback
from pathlib import Path
from queue import Queue, Empty
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from chat_model import get_orchestrator, respond

_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _PKG_DIR.parent
_TOKEN = os.environ.get("CNU_API_TOKEN", "")
_SSE_CHUNK = 4
_SSE_DELAY = 0.03
_REALTIME_OUT = _PROJECT_DIR / "outputs" / "realtime_output.json"
_realtime_lock = threading.Lock()

# 배치 모드 여부 (환경변수로 제어)
_BATCH_MODE = os.environ.get("CNU_BATCH_MODE", "0") == "1"


def _append_realtime(user: str, model: str):
    """실시간 채팅 결과를 realtime_output.json 에 append."""
    with _realtime_lock:
        _REALTIME_OUT.parent.mkdir(parents=True, exist_ok=True)
        if _REALTIME_OUT.exists():
            try:
                data = json.loads(_REALTIME_OUT.read_text(encoding="utf-8"))
            except Exception:
                data = []
        else:
            data = []
        data.append({"user": user, "model": model})
        _REALTIME_OUT.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

app = FastAPI(title="CNU Campus ChatBot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _auth(t):
    if _TOKEN and t != _TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")


class SessionReq(BaseModel):
    session_id: str


class ChatReq(BaseModel):
    session_id: str
    query: str


@app.on_event("startup")
def _startup():
    get_orchestrator()


@app.get("/")
def root():
    return FileResponse(str(_PKG_DIR / "chat.html"))


@app.get("/health")
def health():
    return {"ok": True, "ready": True}


_sessions = set()
_lock = threading.Lock()


@app.post("/api/v1/session/connect")
def connect(req: SessionReq,
            x_api_token=Header(default=None, alias="X-API-Token")):
    _auth(x_api_token)
    with _lock:
        _sessions.add(req.session_id)
        n = len(_sessions)
    return {"ok": True, "ready": True, "session_id": req.session_id, "active_sessions": n}


@app.post("/api/v1/session/disconnect")
def disconnect(req: SessionReq, token=None,
               x_api_token=Header(default=None, alias="X-API-Token")):
    _auth(x_api_token or token)
    with _lock:
        _sessions.discard(req.session_id)
        n = len(_sessions)
    return {"ok": True, "active_sessions": n}


@app.get("/api/v1/mode")
def mode():
    """현재 서버가 배치 모드인지 실시간 전용 모드인지 반환.
    batch_mode: 서버 시작 시 배치 모드로 실행됐는지
    batch_status: idle/running/done/error
    chat_enabled: 실시간 채팅 가능 여부
    """
    with _batch_lock:
        status = _batch_state["status"]
    # 배치 모드가 아니거나, 배치가 완료/에러면 채팅 가능
    chat_enabled = (not _BATCH_MODE) or (status in ("done", "error"))
    return {"batch_mode": _BATCH_MODE, "batch_status": status, "chat_enabled": chat_enabled}


@app.post("/api/v1/cnu-bot/chat")
def chat(req: ChatReq,
         x_api_token=Header(default=None, alias="X-API-Token")):
    _auth(x_api_token)
    resp = get_orchestrator().handle(req.query)
    answer = resp.answer or ""
    # 실시간 채팅 결과 자동 저장
    _append_realtime(req.query, answer)
    return {
        "answer": answer,
        "intent": resp.intent.value if hasattr(resp.intent, "value") else str(resp.intent),
        "is_fallback": resp.is_fallback,
        "references": [{"title": r.title, "source_url": r.source_url}
                       for r in (resp.references or [])],
    }


@app.post("/api/v1/cnu-bot/chat/stream")
def chat_stream(req: ChatReq,
                x_api_token=Header(default=None, alias="X-API-Token")):
    _auth(x_api_token)

    def _gen():
        orch = get_orchestrator()
        yield f"data: {json.dumps({'type':'status','text':'생각 중…'}, ensure_ascii=False)}\n\n"
        resp = orch.handle(req.query)
        meta = {
            "type": "meta",
            "intent": resp.intent.value if hasattr(resp.intent, "value") else str(resp.intent),
            "is_fallback": resp.is_fallback,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
        ans = resp.answer or ""
        for i in range(0, len(ans), _SSE_CHUNK):
            piece = ans[i:i + _SSE_CHUNK]
            yield f"data: {json.dumps({'type':'delta','text':piece}, ensure_ascii=False)}\n\n"
            time.sleep(_SSE_DELAY)
        refs = [{"title": r.title, "source_url": r.source_url}
                for r in (resp.references or [])]
        if refs:
            yield f"data: {json.dumps({'type':'refs','refs':refs}, ensure_ascii=False)}\n\n"
        # 실시간 채팅 결과 자동 저장
        _append_realtime(req.query, ans)
        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# Batch 진행 — UI 안에서 test_chat 실행 + 진행 표시
# ============================================================
_batch_lock = threading.Lock()
_batch_state = {
    "status": "idle",   # idle | running | done | error
    "total": 0,
    "done_count": 0,
    "current_idx": -1,
    "current_user": "",
    "results": [],      # [{idx, user, model}]
    "started_at": 0.0,
    "finished_at": 0.0,
    "error": "",
    "input_path": "",
    "output_path": "",
}
_batch_subscribers: list[Queue] = []


def _broadcast(ev: dict):
    msg = f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    with _batch_lock:
        subs = list(_batch_subscribers)
    for q in subs:
        try:
            q.put_nowait(msg)
        except Exception:
            pass


def _save_results():
    out = _batch_state["output_path"]
    if not out:
        return
    try:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(
            json.dumps(
                [{"user": r["user"], "model": r["model"]} for r in _batch_state["results"]],
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        traceback.print_exc()


def _batch_worker(items, out_path: str):
    try:
        get_orchestrator()  # ensure loaded
        with _batch_lock:
            _batch_state["status"] = "running"
            _batch_state["total"] = len(items)
            _batch_state["done_count"] = 0
            _batch_state["results"] = []
            _batch_state["started_at"] = time.time()
            _batch_state["finished_at"] = 0.0
            _batch_state["error"] = ""
            _batch_state["output_path"] = out_path
        # 시작 시 기존 파일 truncate (빈 리스트)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("[]", encoding="utf-8")
        _broadcast({"type": "start", "total": len(items), "output_path": out_path})

        for i, it in enumerate(items):
            user = (it.get("user") or "").strip()
            with _batch_lock:
                _batch_state["current_idx"] = i
                _batch_state["current_user"] = user
            _broadcast({"type": "progress_begin", "idx": i, "total": len(items), "user": user})
            t0 = time.time()
            try:
                ans = respond(user) if user else ""
            except Exception as e:
                ans = f"[ERROR] {e}"
                traceback.print_exc()
            dt = time.time() - t0
            with _batch_lock:
                _batch_state["results"].append({"idx": i, "user": user, "model": ans})
                _batch_state["done_count"] = i + 1
                _save_results()  # 매 응답마다 저장 (중단 시 복원 가능)
            _broadcast({
                "type": "progress_end", "idx": i, "total": len(items),
                "user": user, "model": ans, "elapsed_sec": round(dt, 2),
            })

        with _batch_lock:
            _batch_state["status"] = "done"
            _batch_state["finished_at"] = time.time()
        _broadcast({
            "type": "done",
            "total": len(items),
            "output_path": out_path,
            "elapsed_sec": round(_batch_state["finished_at"] - _batch_state["started_at"], 2),
        })
    except Exception as e:
        traceback.print_exc()
        with _batch_lock:
            _batch_state["status"] = "error"
            _batch_state["error"] = str(e)
            _batch_state["finished_at"] = time.time()
        _broadcast({"type": "error", "error": str(e)})


@app.post("/api/v1/batch/start")
def batch_start(x_api_token=Header(default=None, alias="X-API-Token")):
    _auth(x_api_token)
    with _lock:
        if _batch_state["status"] == "running":
            return {"ok": False, "reason": "already_running",
                    "done_count": _batch_state["done_count"],
                    "total": _batch_state["total"]}
    in_path = _PROJECT_DIR / "data" / "test_chat.json"
    out_path = _PROJECT_DIR / "outputs" / "chat_output.json"
    if not in_path.exists():
        raise HTTPException(status_code=404, detail=f"input not found: {in_path}")
    try:
        items = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"input parse error: {e}")
    with _batch_lock:
        _batch_state["input_path"] = str(in_path)
        _batch_state["output_path"] = str(out_path)
    th = threading.Thread(target=_batch_worker, args=(items, str(out_path)), daemon=True)
    th.start()
    return {"ok": True, "total": len(items),
            "input_path": str(in_path), "output_path": str(out_path)}


@app.get("/api/v1/batch/status")
def batch_status():
    with _batch_lock:
        return dict(
            status=_batch_state["status"],
            total=_batch_state["total"],
            done_count=_batch_state["done_count"],
            current_idx=_batch_state["current_idx"],
            current_user=_batch_state["current_user"],
            input_path=_batch_state["input_path"],
            output_path=_batch_state["output_path"],
            error=_batch_state["error"],
            elapsed_sec=round(
                (_batch_state["finished_at"] or time.time()) - _batch_state["started_at"], 2
            ) if _batch_state["started_at"] else 0.0,
        )


@app.get("/api/v1/batch/stream")
def batch_stream():
    q: Queue = Queue(maxsize=1024)
    with _batch_lock:
        _batch_subscribers.append(q)
        # 신규 구독자에게 현재 누적 상태 먼저 push
        snap = {
            "type": "snapshot",
            "status": _batch_state["status"],
            "total": _batch_state["total"],
            "done_count": _batch_state["done_count"],
            "current_idx": _batch_state["current_idx"],
            "current_user": _batch_state["current_user"],
            "results": _batch_state["results"][-50:],
            "output_path": _batch_state["output_path"],
        }
    try:
        q.put_nowait(f"data: {json.dumps(snap, ensure_ascii=False)}\n\n")
    except Exception:
        pass

    def _gen():
        try:
            while True:
                try:
                    msg = q.get(timeout=15)
                    yield msg
                except Empty:
                    yield ": ping\n\n"  # keep-alive
        finally:
            with _batch_lock:
                if q in _batch_subscribers:
                    _batch_subscribers.remove(q)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
