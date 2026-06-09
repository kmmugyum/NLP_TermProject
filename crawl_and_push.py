#!/usr/bin/env python3
"""연구서버(한국 IP)용 크롤러 — 학식·공지를 크롤해 data/ JSON 생성 후 git push.

배경: Colab(미국 IP)은 CNU 서버 접속 시 504 Gateway Timeout.
      한국 IP 머신이 대신 크롤 → GitHub → Colab 은 raw fetch 만(라이브 크롤 제거).

사용 (연구서버 cron 예시, 매일 새벽 4시):
    0 4 * * *  cd /path/to/Termproject && python3 crawl_and_push.py >> crawl.log 2>&1

출력:
    data/meal_cache.json    — WeeklyMealCache (학식 주간)
    data/notice_cache.json  — {dept: [NoticeItem...]} (학과별 최신 공지)

환경변수:
    CNU_CRAWL_DEPTS   콤마구분 학과명(부분일치). 미지정 시 DEFAULT_DEPTS.
    CNU_GIT_PUSH      "0" 이면 git push 생략(로컬 테스트용).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

DATA_DIR = ROOT / "data"

# 자주 묻는 주요 학과 우선(전체 75개는 느림). 환경변수로 override.
DEFAULT_DEPTS = [
    "컴퓨터인공지능", "인공지능", "기계공학", "전기", "전자",
    "수학", "물리", "화학", "경영", "행정",
]


def crawl_meal() -> dict | None:
    """학식 주간 크롤 → WeeklyMealCache dict. 실패 시 None."""
    from cnubot.meal_crawler import MEAL_URL, crawl_week
    try:
        t0 = time.time()
        wc = crawl_week(MEAL_URL)
        if not wc.days:
            print("[meal] 수집 0일 — 크롤 실패로 간주, 기존 캐시 유지")
            return None
        print(f"[meal] {len(wc.days)}일 수집 ({time.time()-t0:.1f}s)")
        return json.loads(wc.model_dump_json())
    except Exception as e:
        print(f"[meal] 크롤 실패 → {type(e).__name__}: {e}")
        return None


def crawl_notices(depts: list[str]) -> dict | None:
    """학과별 최신 공지 크롤 → {dept_label: [item dict...]}. 전부 실패 시 None."""
    from cnubot.notice import NoticeService
    svc = NoticeService()
    out: dict[str, list] = {}
    for kw in depts:
        try:
            t0 = time.time()
            label, items = svc.collect(kw)
            if items:
                out[label] = [json.loads(it.model_dump_json()) for it in items]
                print(f"[notice] {label}: {len(items)}건 ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"[notice] {kw} 실패 → {type(e).__name__}: {e}")
    if not out:
        print("[notice] 전체 수집 0 — 기존 캐시 유지")
        return None
    return out


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fetched_at": datetime.now().isoformat(), "data": data}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {path} ({path.stat().st_size} bytes)")


def git_push(paths: list[Path]) -> None:
    """변경된 data 파일만 commit·push. 실패해도 크롤 결과는 디스크에 남음."""
    if os.environ.get("CNU_GIT_PUSH") == "0":
        print("[git] CNU_GIT_PUSH=0 → push 생략")
        return
    try:
        rel = [str(p.relative_to(ROOT)) for p in paths]
        subprocess.run(["git", "add", *rel], cwd=ROOT, check=True)
        # 변경 없으면 commit 이 실패하므로 먼저 확인
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
        if diff.returncode == 0:
            print("[git] 변경 없음 — commit 생략")
            return
        msg = f"data: 크롤 갱신 {datetime.now():%Y-%m-%d %H:%M}"
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT, check=True)
        subprocess.run(["git", "push"], cwd=ROOT, check=True)
        print(f"[git] push 완료: {msg}")
    except subprocess.CalledProcessError as e:
        print(f"[git] 실패 → {e}. 데이터는 디스크에 저장됨(다음 실행 시 재시도).")


def main() -> int:
    depts_env = os.environ.get("CNU_CRAWL_DEPTS")
    depts = [d.strip() for d in depts_env.split(",")] if depts_env else DEFAULT_DEPTS
    print(f"=== 크롤 시작 {datetime.now():%Y-%m-%d %H:%M} (학과 {len(depts)}개) ===")

    written: list[Path] = []
    meal = crawl_meal()
    if meal is not None:
        p = DATA_DIR / "meal_cache.json"
        _write_json(p, meal)
        written.append(p)

    notices = crawl_notices(depts)
    if notices is not None:
        p = DATA_DIR / "notice_cache.json"
        _write_json(p, notices)
        written.append(p)

    if not written:
        print("=== 수집 결과 없음 — git push 생략 ===")
        return 1
    git_push(written)
    print("=== 크롤 완료 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
