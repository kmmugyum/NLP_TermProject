"""HTTP fetch utilities — Colab Free 환경 최적화.

자동 통합 (cnubot 패키지 import 시 활성):
  A. Persistent HTTP/2 client — TCP/TLS handshake 재사용 (연속 fetch 5~10배 가속)
  B. 3-stage retry (10/30/60s, exponential backoff) — 느린/지연되는 네트워크에 회복
  C. 디스크 캐시 (Drive 또는 /tmp, TTL 1시간) — 반복 호출 0초 + 실패 시 stale fallback

사용:
  기존 `httpx.get(url, ...)` 코드 그대로 — `_net` import 시점에 httpx.get 을
  monkey-patch 하여 투명하게 적용. 모듈 호출자는 수정 불필요.

환경변수:
  CNU_NET_CACHE     캐시 디렉터리 (기본: Drive hf_cache/fetch_cache, 없으면 /tmp)
  CNU_NET_CACHE_TTL TTL 초 (기본 3600)
  CNU_NET_DISABLE   "1" 이면 모든 가속 비활성 (디버깅용)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
_DISABLED = os.environ.get("CNU_NET_DISABLE") == "1"
_CACHE_TTL = int(os.environ.get("CNU_NET_CACHE_TTL", "120"))  # 2분 TTL (학식 등 실시간 데이터 반영)

# 서버 시작 시 기존 캐시 전체 삭제 (stale 데이터 방지)
_CLEAR_ON_START = os.environ.get("CNU_NET_CLEAR_CACHE", "1") == "1"
_CACHE_DIR_ENV = os.environ.get("CNU_NET_CACHE")
if _CACHE_DIR_ENV:
    _CACHE_DIR = Path(_CACHE_DIR_ENV)
elif os.path.isdir("/content/drive/MyDrive"):
    _CACHE_DIR = Path("/content/drive/MyDrive/hf_cache/fetch_cache")
else:
    _CACHE_DIR = Path("/tmp/cnu_fetch_cache")
try:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    _CACHE_DIR = Path("/tmp/cnu_fetch_cache")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RETRY_TIMEOUTS = (10, 30, 60)  # 3-stage

# 시작 시 stale 캐시 정리
if _CLEAR_ON_START and not _DISABLED:
    try:
        import glob
        cleared = 0
        for f in glob.glob(str(_CACHE_DIR / "*.json")):
            try:
                os.remove(f)
                cleared += 1
            except Exception:
                pass
        if cleared:
            print(f"[_net] 기존 HTTP 캐시 {cleared}개 삭제 (stale 방지)")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Persistent client (HTTP/2 우선, h2 없으면 HTTP/1.1)
# ---------------------------------------------------------------------------
def _build_client() -> httpx.Client:
    common = dict(
        verify=False,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=120),
    )
    try:
        return httpx.Client(http2=True, **common)
    except Exception:
        return httpx.Client(**common)


_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


# ---------------------------------------------------------------------------
# 캐시 헬퍼
# ---------------------------------------------------------------------------
def _cache_path(url: str, params=None) -> Path:
    key = url
    if params:
        try:
            key = f"{url}?{httpx.QueryParams(params)}"
        except Exception:
            pass
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{h}.json"


def _save_cache(path: Path, status_code: int, text: str, url: str,
                content_type: str = "") -> None:
    try:
        path.write_text(
            json.dumps({"t": time.time(), "status_code": status_code,
                        "text": text, "url": url,
                        "content_type": content_type}),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_cache(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class _CachedResponse:
    """httpx.Response 의 최소 호환 객체 (text, status_code, url, headers)."""

    def __init__(self, d: dict):
        self.status_code = d.get("status_code", 200)
        self.text = d.get("text", "")
        self._url = d.get("url", "")
        # headers 는 dict 만 흉내 — content-type 만 필요한 경우가 대부분
        self.headers = {"content-type": d.get("content_type", "text/html; charset=utf-8")}

    @property
    def url(self):
        return httpx.URL(self._url) if self._url else None

    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8")

    def raise_for_status(self):
        return None


# ---------------------------------------------------------------------------
# get / head 래퍼 (httpx.get drop-in)
# ---------------------------------------------------------------------------
# 원본 보관 (재진입 무한루프 방지)
_orig_get = httpx.get
_orig_head = httpx.head


def get(url, *, params=None, headers=None, timeout=None,
        verify=None, follow_redirects=None, **kw):
    """httpx.get drop-in. 캐시 hit 시 즉시 반환, miss 시 3-stage retry."""
    if _DISABLED:
        return _orig_get(
            url, params=params, headers=headers,
            timeout=timeout if timeout is not None else 60,
            verify=False if verify is None else verify,
            follow_redirects=True if follow_redirects is None else follow_redirects,
            **kw,
        )

    cf = _cache_path(url, params)
    # 캐시 hit
    cached = _load_cache(cf)
    if cached and time.time() - cached.get("t", 0) < _CACHE_TTL:
        return _CachedResponse(cached)

    # 3-stage retry
    last_err: Exception | None = None
    cli = _get_client()
    for i, ts in enumerate(_RETRY_TIMEOUTS):
        # caller 가 명시 timeout 주면 1차 시도에만 사용, 그 외엔 retry 단계별 ts
        eff_timeout = timeout if (timeout is not None and i == 0) else ts
        try:
            req_headers = headers
            r = cli.get(url, params=params, headers=req_headers, timeout=eff_timeout)
            _save_cache(cf, r.status_code, r.text, str(r.url),
                        r.headers.get("content-type", ""))
            return r
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError,
                httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError) as e:
            last_err = e
            if i < len(_RETRY_TIMEOUTS) - 1:
                time.sleep(0.5 * (2 ** i))  # 0.5s, 1s

    # 모두 실패 — stale cache 라도 반환
    if cached:
        return _CachedResponse(cached)
    assert last_err is not None
    raise last_err


def head(url, **kw):
    if _DISABLED:
        return _orig_head(url, **kw)
    try:
        return _get_client().head(url, **kw)
    except Exception:
        return _orig_head(url, **kw)


# ---------------------------------------------------------------------------
# Monkey-patch 적용
# ---------------------------------------------------------------------------
if not _DISABLED:
    httpx.get = get  # type: ignore[assignment]
    httpx.head = head  # type: ignore[assignment]
