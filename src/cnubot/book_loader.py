"""학사요람/규정 PDF 텍스트 → 조항(제O조) 경계 기반 청킹.

요람은 학칙/학사운영규정(제O조)이 핵심 → 조항 경계로 분할해 문맥 보존.
교과목 실러버스(학점 N-N-N 등) 라인은 노이즈라 제거.
"""
from __future__ import annotations

import hashlib
import re

_ARTICLE_RE = re.compile(r"제\s*\d+\s*조\s*[(（]")  # 제26조( …
_SYLLABUS_RE = re.compile(r"학점\s*\d+\s*-\s*\d+\s*-\s*\d+|교과목\s*개요|course\s*", re.I)

# PDF 목차/불릿 아티팩트 멸균: ● 런(65k개)·· 가운뎃점 점선·○◦■･ 불릿·PUA 글리프 제거.
# 의미 기호(①②③, ※)와 문장 마침표·소수점(1.75)·단일 불릿은 보존.
_PUA_RE = re.compile(r"[-\U000F0000-\U0010FFFD]")   # 깨진 글리프(U+F080F 등)
_LEADER_RUN_RE = re.compile(r"(?:[●○◦■·･]\s*){2,}")            # 불릿/가운뎃점 2+ 런
_DOTLEAD_RE = re.compile(r"\.{3,}")                            # ASCII 점선 3+ (소수점 보존)
_WS_RE = re.compile(r"[ \t　]{2,}")


def _sterilize(line: str) -> str:
    line = _PUA_RE.sub("", line)
    line = _LEADER_RUN_RE.sub(" ", line)
    line = _DOTLEAD_RE.sub(" ", line)
    return _WS_RE.sub(" ", line.replace("　", " "))


def chunk_academic_book(text: str, source_url: str, title: str,
                        max_chars: int = 1200, min_chars: int = 50) -> list[dict]:
    chunks: list[str] = []
    buf: list[str] = []
    size = 0

    def flush():
        nonlocal buf, size
        if buf:
            body = _sterilize("\n".join(buf))  # 줄 경계서 합쳐진 '●\n●' 런까지 멸균
            if len(body.strip()) >= min_chars:
                chunks.append(body)
        buf, size = [], 0

    for line in text.split("\n"):
        s = _sterilize(line).strip()  # ● 런/· 점선/PUA 글리프 멸균
        if not s or _SYLLABUS_RE.search(s):
            continue
        if _ARTICLE_RE.match(s) and buf:  # 새 조항 시작 → 단락 분할
            flush()
        buf.append(s)
        size += len(s)
        if size >= max_chars:
            flush()
    flush()

    out = []
    for i, body in enumerate(chunks):
        h = hashlib.md5(body.encode("utf-8")).hexdigest()[:12]
        out.append({
            "doc_id": f"cnu_book_{h}_{i}", "source_url": source_url,
            "title": title, "content": body,
        })
    return out
