"""학식 수동 동기화 CLI (cron/데몬 없음 — 호출할 때만 크롤).

학사 코퍼스는 정적(요람=연1회)이라 일일 동기화 불필요 → 새 요람만 webhook 수동 적재.
일일 변동 대상은 학식뿐 → 이 스크립트가 그 유일한 갱신 경로.

  python -m cnubot.manual_sync                 # 크롤 → 디스크 캐시 핫스왑
  python -m cnubot.manual_sync --push http://127.0.0.1:8000  # + 가동 서버 in-memory swap

빈 메뉴(주말/공휴일)면 기존 캐시 보존하고 중단 — staleness fallback이 처리.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .meal_crawler import MEAL_URL, crawl_week
from .module1_indexer import _atomic_write_text
from .module4_api import MEAL_CACHE_PATH


def sync_meal(url: str = MEAL_URL, cache_path: str = MEAL_CACHE_PATH,
              push_base: str | None = None) -> int:
    wc = crawl_week(url)  # → WeeklyMealCache (이번 주 월~금)
    ndays = len(wc.days)
    if ndays == 0:
        print("[중단] 주간 크롤 결과 0일 (주말/공휴일/소스 변동?). "
              "기존 캐시 보존 — 덮어쓰지 않음.", file=sys.stderr)
        return 0

    _atomic_write_text(Path(cache_path), wc.model_dump_json())  # 디스크 핫스왑(서버 reload용)
    tot = sum(len(v) for v in wc.days.values())
    print(f"[디스크] {cache_path} 갱신 — {ndays}일 / 총 {tot}건 "
          f"(주 시작 {wc.week_start}, 적재 {wc.timestamp:%Y-%m-%d %H:%M})")
    print(f"  수집일: {sorted(wc.days)}")

    if push_base:  # 가동 중 서버에 즉시 in-memory swap
        import httpx

        payload = {"kind": "meal", "data": [wc.model_dump(mode="json")]}
        r = httpx.post(f"{push_base.rstrip('/')}/api/v1/webhook/ingest",
                       json=payload, timeout=15.0)
        r.raise_for_status()
        print(f"[서버] webhook in-memory swap → {r.json()}")
    return ndays


def main() -> None:
    ap = argparse.ArgumentParser(description="학식 수동 크롤 동기화")
    ap.add_argument("--url", default=MEAL_URL, help="학식 소스 URL(기본: cnu food)")
    ap.add_argument("--cache-path", default=MEAL_CACHE_PATH, help="디스크 캐시 경로")
    ap.add_argument("--push", default=None, metavar="BASE_URL",
                    help="가동 중 서버 base url (예: http://127.0.0.1:8000) → 즉시 in-memory swap")
    a = ap.parse_args()
    n = sync_meal(a.url, a.cache_path, a.push)
    sys.exit(0 if n else 1)


if __name__ == "__main__":
    main()
