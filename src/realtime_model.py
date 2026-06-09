"""CNU Campus ChatBot — 실시간 정보 응답 (Optional Task 3).

학식·공지·셔틀·학사일정 등 동적 정보는 chat_model.respond() 가 이미
Orchestrator 내부에서 처리(meal_crawler 자동 트리거, notice live fetch).

본 모듈은 평가용 진입점: test_realtime.json → outputs/realtime_output.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from chat_model import respond


def realtime_batch(in_path: str, out_path: str) -> None:
    items = json.loads(Path(in_path).read_text(encoding="utf-8"))
    results = [{"user": it["user"], "model": respond(it["user"])} for it in items]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        realtime_batch(sys.argv[1], sys.argv[2])
    else:
        realtime_batch("../data/test_realtime.json", "../outputs/realtime_output.json")
    print("DONE")
