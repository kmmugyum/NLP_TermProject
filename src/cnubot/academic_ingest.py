"""학사 데이터 적재 전처리 (기존 crawl4ai 덤프 → CNUExtractedChunk 호환 청크).

경량 원칙: md5 dedup + 문단경계 청킹 + 문단단위 overlap (단어 파손 0, 의존성 0).
giant 단일 문단(문단 1개가 chunk_chars 초과)만 공백경계 윈도우로 약식 분할.
실측(1955만자): 유령문자 2·바이너리 0 → 멸균/엔트로피/BPE 가드 전부 불필요(미탑재).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any


def _subsplit(p: str, limit: int) -> list[str]:
    """단일 문단이 limit 초과 시 공백 경계로 분할 (단어 중간 절단 회피)."""
    if len(p) <= limit:
        return [p]
    out, i = [], 0
    while i < len(p):
        end = min(i + limit, len(p))
        if end < len(p):
            sp = p.rfind(" ", i, end)  # 윈도우 내 마지막 공백
            if sp > i:
                end = sp
        seg = p[i:end].strip()
        if seg:
            out.append(seg)
        i = end
    return out


_META_RE = re.compile(r"^(작성자|조회수|등록일|첨부파일|이전글|다음글|목록)\s*[|:]")
_PIPE_RE = re.compile(r"^[|\s\-:]{3,}$")  # 파이프/대시만으로 된 표 잔재 줄
_NAV_RE = re.compile(r"\[URL복사\]|\[SNS[^\]]*\]|\[프린트\]|\[트위터\]|\[페이스북\]|\[카카오")
_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")  # 마크다운 링크 (plus.cnu 전역 메뉴 검출용)
_BRAND_RE = re.compile(r"The Strong CNU|THE STRONG CNU|미래\s*사회를\s*선도할\s*강한\s*대학|"
                       r"통합검색\s*검색\s*통합검색\s*닫기버튼|본문\s*바로가기")
_NOISE_URL = ("/notice", "mode=view", "board", "/article", "sub07")
# title/url 에 들어있으면 페이지 통째 드롭 (검색 잡음원)
_NOISE_KEYWORDS = ("개인정보처리방침", "갤러리", "게시판목록", "사이트맵", "로그인")


def _is_noise_url(url: str) -> bool:
    if "sub05" in url:  # 사용자가 발굴한 canonical 학사 가이드는 보존
        return False
    return any(x in url for x in _NOISE_URL)


def _is_noise_page(url: str, title: str) -> bool:
    return any(k in title or k in url for k in _NOISE_KEYWORDS)


def ingest_refined(
    records: list[dict], chunk_chars: int = 1000, overlap_paras: int = 1,
    min_chunk_chars: int = 50,
) -> list[dict[str, Any]]:
    """정제 강화본: 공지/게시판 URL 탈락 + 라인 boilerplate(작성자|·파이프표) 제거
    + 50자 미만 파편 컷 + content-hash doc_id. (1+2 결합)"""
    seen: set[str] = set()
    out: list[dict] = []
    counter = 0
    for r in records:
        url = r.get("url") or ""
        if _is_noise_url(url) or _is_noise_page(url, r.get("title") or ""):
            continue
        c = r.get("content") or ""
        if len(c.strip()) < 10:
            continue
        h = hashlib.md5("".join(c.split()).encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        paras: list[str] = []
        for line in c.split("\n"):
            line = _NAV_RE.sub("", line).strip()  # nav 링크 토큰 제거
            line = _BRAND_RE.sub("", line).strip()  # 전역 브랜딩 헤더 제거(임베딩 희석 방지)
            if not line or _PIPE_RE.match(line) or _META_RE.match(line):
                continue
            # nav 메뉴 줄 제거: 마크다운 링크가 줄의 50%+ (plus.cnu 전역 메뉴 = 89% 차지)
            if len("".join(_LINK_RE.findall(line))) / max(len(line), 1) > 0.5:
                continue
            paras.extend(_subsplit(line, chunk_chars))

        chunks: list[str] = []
        buf: list[str] = []
        n = 0
        for p in paras:
            buf.append(p)
            n += len(p)
            if n >= chunk_chars:
                chunks.append("\n".join(buf))
                buf = buf[-overlap_paras:] if overlap_paras else []
                n = sum(len(x) for x in buf)
        if buf:
            chunks.append("\n".join(buf))

        title = (r.get("title") or "").strip() or None
        for text in chunks:
            if len(text.strip()) < min_chunk_chars:
                continue
            tid = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
            out.append({
                "doc_id": f"cnu_chunk_{tid}_{counter}", "content": text,
                "title": title, "source_url": url or None,
                "file_type": "html", "metadata": {},
            })
            counter += 1
    return out


def ingest(
    records: list[dict], chunk_chars: int = 1000, overlap_paras: int = 1
) -> list[dict[str, Any]]:
    """raw 레코드(url/title/content) → CNUExtractedChunk dict 리스트.

    - md5(공백제거 본문) 기준 Exact Dedup
    - 문단(\\n) 경계로 chunk_chars 까지 greedy 패킹, overlap_paras 문단만큼 슬라이딩
    - doc_id: acad_{url해시10}_{청크순번} (페이지간 고유)
    """
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        c = r.get("content") or ""
        if len(c.strip()) < 10:
            continue
        h = hashlib.md5("".join(c.split()).encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        url, title = r.get("url", ""), r.get("title", "")
        paras: list[str] = []
        for line in c.split("\n"):
            line = line.strip()
            if line:
                paras.extend(_subsplit(line, chunk_chars))

        chunks: list[str] = []
        buf: list[str] = []
        n = 0
        for p in paras:
            buf.append(p)
            n += len(p)
            if n >= chunk_chars:
                chunks.append("\n".join(buf))
                buf = buf[-overlap_paras:] if overlap_paras else []
                n = sum(len(x) for x in buf)
        if buf:
            chunks.append("\n".join(buf))

        did = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
        for i, text in enumerate(chunks):
            out.append({
                "doc_id": f"acad_{did}_{i}",
                "content": text,
                "title": title or None,
                "source_url": url or None,
                "file_type": "html",
                "metadata": {},
            })
    return out
