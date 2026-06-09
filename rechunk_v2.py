"""재청킹 v2 — 측정된 두 결함(완전중복 39.5%, 빈 파이프 표) 직격 수정.

기존 ingest_refined() 대비 추가:
  (A) 청크-레벨 content dedup  — 페이지단위 dedup만으로 안 잡히는 공통 nav/사이드바
      반복(예: library 사이드메뉴 217회)을 청크 정규화 해시로 제거.
  (B) 빈 파이프 표 압축        — `| |||||||` → 제거/단일화, 의미없는 표행 드롭,
      의미있는 표행(`| 과목 | 학점 |`)은 보존.
  (C) 길이 가드               — <200자 청크 폐기, 최대 2000자.

입력: 디렉토리의 *_out.jsonl raw 크롤 덤프 (url/title/content). 공지/게시판 제외.
출력: _academic_chunks_v2.json  (기존 _academic_chunks.json 덮어쓰지 않음)
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from collections import Counter

from cnubot.academic_ingest import (
    _subsplit, _is_noise_url, _is_noise_page,
    _META_RE, _PIPE_RE, _NAV_RE, _LINK_RE, _BRAND_RE,
)

# (B) 빈 셀 연속 압축: `| |`, `| | |`, ... (공백 포함 파이프 2개 이상 연속) → 단일 `|`
_EMPTY_PIPE_RE = re.compile(r"\|(?:\s*\|)+")
# 압축 후 의미 없는 표행: 파이프/공백/대시/콜론만 남은 줄
_JUNK_ROW_RE = re.compile(r"^[|\s\-:]*$")


def _norm(s: str) -> str:
    """dedup/표 판정용 정규화: 공백 squeeze."""
    return re.sub(r"\s+", " ", s).strip()


def _fix_table_line(line: str) -> str | None:
    """빈 파이프 압축. 의미 없는 표행이면 None(드롭)."""
    if "|" not in line:
        return line
    fixed = _EMPTY_PIPE_RE.sub("|", line)
    # 양끝 잉여 파이프 정리: '| a | b |' 형태는 유지, 단독/잔재는 드롭
    if _JUNK_ROW_RE.match(fixed):
        return None
    return fixed.strip()


CHUNK_CHARS = 1000
OVERLAP_PARAS = 1
MIN_CHUNK = 200          # (C) <200자 폐기 (프롬프트 규칙)
MAX_CHUNK = 2000         # (C) 최대 2000자


def ingest_v2(records: list[dict]) -> tuple[list[dict], dict]:
    page_seen: set[str] = set()      # 페이지단위 dedup
    chunk_seen: set[str] = set()     # (A) 청크단위 dedup
    out: list[dict] = []
    counter = 0
    stats = Counter()

    for r in records:
        url = r.get("url") or ""
        title = (r.get("title") or "").strip()
        if _is_noise_url(url) or _is_noise_page(url, title):
            stats["page_noise_drop"] += 1
            continue
        c = r.get("content") or ""
        if len(c.strip()) < 10:
            continue
        ph = hashlib.md5("".join(c.split()).encode()).hexdigest()
        if ph in page_seen:
            stats["page_dup_drop"] += 1
            continue
        page_seen.add(ph)
        stats["pages_kept"] += 1

        paras: list[str] = []
        for line in c.split("\n"):
            line = _NAV_RE.sub("", line).strip()
            line = _BRAND_RE.sub("", line).strip()
            if not line:
                continue
            # (B) 표 빈 파이프 압축 / 잡표행 드롭
            fixed = _fix_table_line(line)
            if fixed is None:
                stats["junk_table_rows"] += 1
                continue
            line = fixed
            if _PIPE_RE.match(line) or _META_RE.match(line):
                continue
            # nav 메뉴 줄: 마크다운 링크가 줄의 50%+ 차지
            if len("".join(_LINK_RE.findall(line))) / max(len(line), 1) > 0.5:
                stats["nav_link_rows"] += 1
                continue
            paras.extend(_subsplit(line, MAX_CHUNK))

        # greedy 패킹
        chunks: list[str] = []
        buf: list[str] = []
        n = 0
        for p in paras:
            buf.append(p)
            n += len(p)
            if n >= CHUNK_CHARS:
                chunks.append("\n".join(buf))
                buf = buf[-OVERLAP_PARAS:] if OVERLAP_PARAS else []
                n = sum(len(x) for x in buf)
        if buf:
            chunks.append("\n".join(buf))

        for text in chunks:
            text = text.strip()
            if len(text) < MIN_CHUNK:
                stats["short_chunk_drop"] += 1
                continue
            if len(text) > MAX_CHUNK:
                text = text[:MAX_CHUNK]
            # (A) 청크-레벨 dedup (정규화 해시)
            ch = hashlib.md5(_norm(text).encode()).hexdigest()
            if ch in chunk_seen:
                stats["chunk_dup_drop"] += 1
                continue
            chunk_seen.add(ch)
            tid = hashlib.md5(text.encode()).hexdigest()[:12]
            out.append({
                "doc_id": f"cnu_chunk_{tid}_{counter}", "content": text,
                "title": title or None, "source_url": url or None,
                "file_type": "html", "metadata": {},
            })
            counter += 1
    return out, dict(stats)


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*_out.jsonl"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "_academic_chunks_v2.json"
    files = sorted(glob.glob(pattern))
    records: list[dict] = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        records.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    print(f"[in] {len(files)} 파일, {len(records)} raw 레코드")
    chunks, stats = ingest_v2(records)
    json.dump(chunks, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[out] {len(chunks)} 청크 → {out_path}")
    print("[stats]", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
