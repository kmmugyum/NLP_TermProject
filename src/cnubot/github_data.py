"""GitHub raw 데이터 소스 — Colab(미국 IP)이 CNU 서버 504를 우회하는 핵심.

배경: Colab 은 *.cnu.ac.kr 접속 시 504 Gateway Timeout. 한국 IP 연구서버가 크롤해
      GitHub 에 올린 JSON 을 raw 로 읽어 라이브 크롤을 대체한다.

데이터 계약(연구서버 crawl_and_push.py 가 생성):
  data/meal_cache.json    {"fetched_at": ISO, "data": <WeeklyMealCache dict>}
  data/notice_cache.json  {"fetched_at": ISO, "data": {dept_label: [NoticeItem dict...]}}

환경변수:
  CNU_DATA_REPO   raw base, 예: "https://raw.githubusercontent.com/kmmugyum/NLP_TermProject/main"
                  미설정 시 None → 호출자는 기존 라이브 크롤로 폴백(하위호환).
  CNU_DATA_TTL    raw 재fetch 간격(초, 기본 1800=30분). Colab 세션 내 반복 fetch 절감.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

_RAW_BASE = os.environ.get("CNU_DATA_REPO")  # None 이면 GitHub 소스 비활성
_TTL = int(os.environ.get("CNU_DATA_TTL", "1800"))
_LOCAL_DIR = Path(os.environ.get("CNU_DATA_LOCAL", "/tmp/cnu_github_data"))
_LOCAL_DIR.mkdir(parents=True, exist_ok=True)


def is_enabled() -> bool:
    """GitHub 데이터 소스가 켜져 있나(CNU_DATA_REPO 설정 시)."""
    return bool(_RAW_BASE)


def _local_path(name: str) -> Path:
    return _LOCAL_DIR / name


def fetch_json(name: str) -> dict | None:
    """data/{name} 을 GitHub raw 에서 읽어 dict 반환. 실패 시 로컬 캐시, 그것도 없으면 None.
    name 예: 'meal_cache.json'. 반환은 계약의 'data' 필드(내부 dict)."""
    if not _RAW_BASE:
        return None
    cache = _local_path(name)
    # 로컬 캐시가 신선하면 재fetch 생략
    if cache.exists() and (time.time() - cache.stat().st_mtime) < _TTL:
        return _read_data(cache)
    url = f"{_RAW_BASE.rstrip('/')}/data/{name}"
    try:
        r = httpx.get(url, timeout=15.0, follow_redirects=True)
        r.raise_for_status()
        payload = r.json()
        cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"[github-data] {name} fetch 성공 (fetched_at={payload.get('fetched_at')})")
        return payload.get("data")
    except Exception as e:
        print(f"[github-data] {name} fetch 실패 → {type(e).__name__}: {e}")
        if cache.exists():  # stale 이라도 로컬 캐시 폴백
            print(f"[github-data] 로컬 캐시로 폴백: {cache}")
            return _read_data(cache)
        return None


def _read_data(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("data")
    except Exception:
        return None
