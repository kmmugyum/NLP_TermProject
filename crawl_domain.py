#!/usr/bin/env python3
"""CNU 도메인 와이드 크롤러 — *.cnu.ac.kr 정적 페이지 BFS 수집 → 청크 JSON → (별도) 벡터 빌드.

매일 도는 crawl_and_push.py(학식·공지)와 **별개**. 이건 학과소개·교과과정·규정·시설
안내 같은 '잘 안 변하는 정적 페이지'를 모아 학사 코퍼스를 넓히는 무거운 주간 배치다.

⚠️ 안전 설계 (셋 다 필수, 하나라도 빠지면 안 끝남):
  1) visited set (query/fragment 제거 정규화) — 무한루프·페이지네이션 함정 차단
  2) BFS 깊이 상한(--depth, 기본 3)
  3) 페이지 총 상한(CRAWL_MAX_PAGES)

⚠️ 격리 원칙: 신규 청크는 storage/_domain_chunks.json + storage/domain.bin 으로 **별도** 빌드.
   기존 academic 코퍼스에 바로 병합하지 않는다 (eval 회귀 측정 후 수동 병합).

단계 분리:
  --stage crawl : 크롤 + 청크화 (httpx/bs4만 필요 → 어느 노드든)
  --stage build : 임베딩 + faiss (torch + KURE 필요 → GPU 노드)
  --stage all   : 둘 다 (한 노드에 torch+KURE 있을 때)

환경변수: CRAWL_MAX_PAGES, CRAWL_DELAY, CRAWL_DEPTH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# fetch 품질 가드 상수 재사용 (notice.py에 이미 검증된 것)
from cnubot.notice import _HEAD, _LOGIN_RE, _NONHTML_EXT  # noqa: E402

STORAGE = ROOT / "src" / "cnubot" / "storage"
CHUNKS_OUT = STORAGE / "_domain_chunks.json"
INDEX_OUT = STORAGE / "domain.bin"
REGISTRY = SRC / "cnubot" / "data" / "dept_registry.json"

# 주요 부서(학과 게시판 외 — 학교본부/도서관/입학/장학 등). 시드 보강.
EXTRA_SEEDS = [
    "plus.cnu.ac.kr",       # 학교본부
    "library.cnu.ac.kr",    # 도서관
    "ipsi.cnu.ac.kr",       # 입학
    "www.cnu.ac.kr",        # 대표
    "eng.cnu.ac.kr",        # 공과대학
    "ns.cnu.ac.kr",         # 자연과학대학
]

# 호스트 루트만으로는 BFS depth 안에 안 잡힐 수 있는 '딥 페이지'를 직접 시드.
# (셔틀버스·교통편 — 사용자 요청. 봇 위임 URL과 동일 페이지)
EXTRA_SEED_URLS = [
    "https://plus.cnu.ac.kr/html/kr/sub05/sub05_050403.html",   # 학교셔틀버스
    "https://plus.cnu.ac.kr/html/kr/sub01/sub01_01080301.html", # 찾아오시는길
    "https://plus.cnu.ac.kr/html/kr/sub01/sub01_01080302.html", # 교통편안내
]

_DOMAIN_SUFFIX = ".cnu.ac.kr"
_DROP_SUFFIXES = tuple(_NONHTML_EXT)  # .pdf/.hwp/.jpg/.zip ... 큐에 안 넣음


import re

# CNU CMS 공통 네비 보일러플레이트 (페이지마다 반복 → 검색 노이즈)
_BOILER = re.compile(
    r"(본문|주메뉴|서브메뉴|상단메뉴|하단메뉴|메뉴)\s*바로가기"
    r"|바로가기|더보기|MORE\b|페이지\s*처음으로|TOP\b")
_DATE = re.compile(r"20\d\d[.\-\s]*\d\d?[.\-\s]*\d\d?")


def _strip_boilerplate(text: str) -> str:
    return " ".join(_BOILER.sub(" ", text).split())


def _looks_listy(text: str) -> bool:
    """공지/게시판 리스트형(날짜가 촘촘) → 매일 바뀌는 시의성 콘텐츠. 정적 코퍼스에서 제외.
    날짜 토큰 밀도(1000자당 4건 이상)로 판별."""
    dates = len(_DATE.findall(text))
    return dates >= 4 and dates / max(len(text) / 1000, 1) >= 4


def in_domain(host: str) -> bool:
    host = (host or "").lower()
    return host == "cnu.ac.kr" or host.endswith(_DOMAIN_SUFFIX)


def normalize(url: str) -> str | None:
    """visited 비교용 정규화: scheme/host 소문자, query·fragment 제거, 끝 슬래시 정리.
    query 제거 = ?page=N / 캘린더 같은 무한 URL 함정 차단(정적 지식 수집엔 무해)."""
    try:
        p = urlparse(url)
    except Exception:
        return None
    if p.scheme not in ("http", "https"):
        return None
    host = p.netloc.lower()
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, host, path, "", "", ""))


def extract(url: str, timeout: float) -> tuple[str | None, str | None, list[str], str]:
    """페이지 1회 fetch → (본문텍스트|None, 제목|None, 도메인내 링크들, 결과태그).
    결과태그: 'ok'(본문 저장가능) | 'filtered'(fetch 성공이나 품질미달) | 'error'(fetch 실패).
    본문 추출 가드는 notice.fetch_page_text 로직을 미러(비HTML/로그인/빈셸/대형 거름).
    politeness 위해 페이지당 fetch는 1회만(링크+본문 동시 추출)."""
    try:
        r = httpx.get(url, headers=_HEAD, timeout=timeout, verify=False,
                      follow_redirects=True)
    except Exception:
        return None, None, [], "error"
    if r.status_code != 200:
        return None, None, [], "error"
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype.lower():
        return None, None, [], "error"
    if len(r.content) > 5_000_000:
        return None, None, [], "error"
    fp = str(r.url)
    if _LOGIN_RE.search(fp):  # 로그인으로 redirect → 본문 없음
        return None, None, [], "error"

    soup = BeautifulSoup(r.content, "lxml")

    # --- 링크 수집 (본문 거르기 전에: 네비 페이지도 링크 허브일 수 있음) ---
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        nxt = urljoin(fp, a["href"].strip())
        pu = urlparse(nxt)
        if not in_domain(pu.netloc):
            continue
        if pu.path.lower().rsplit("?", 1)[0].endswith(_DROP_SUFFIXES):
            continue
        links.append(nxt)

    # --- 제목 ---
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        title = soup.find("h1").get_text(" ", strip=True)

    # --- 본문 추출 (fetch_page_text 미러 + 품질 필터) ---
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "aside"]):
        tag.decompose()
    # 본문 컨테이너 우선. fr-view/content-box = 실제 글 본문(신뢰), 없으면 body 폴백.
    el = (soup.select_one("div.fr-view") or soup.select_one("div.b-content-box")
          or soup.select_one("#content") or soup.select_one("div.content")
          or soup.select_one("[role=main]") or soup.select_one("main"))
    is_article = el is not None  # 명시적 본문 컨테이너를 찾았는가
    if el is None:
        el = soup.body
    body = None
    if el:
        # 링크밀도: 앵커텍스트가 본문의 큰 비중이면 공지리스트/네비 허브 → 인덱싱 제외
        full = " ".join(el.get_text(" ").split())
        link_txt = sum(len(a.get_text(" ", strip=True)) for a in el.find_all("a"))
        link_density = link_txt / max(len(full), 1)
        txt = _strip_boilerplate(full)
        # 본문 컨테이너(is_article)면 링크밀도 기준 완화, body 폴백이면 엄격
        max_density = 0.55 if is_article else 0.30
        if len(txt) >= 80 and link_density <= max_density and not _looks_listy(txt):
            body = txt
    return body, title, links, ("ok" if body else "filtered")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """공백 경계 보존 슬라이딩 청크. size 문자 단위, overlap 만큼 겹침."""
    text = text.strip()
    if len(text) <= size:
        return [text]
    out, i, n = [], 0, len(text)
    while i < n:
        end = min(i + size, n)
        # 단어 중간 자르기 완화: 끝 근처 공백에서 절단
        if end < n:
            sp = text.rfind(" ", i + size - overlap, end)
            if sp > i:
                end = sp
        out.append(text[i:end].strip())
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return [c for c in out if c]


def seeds() -> list[str]:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    hosts = [d["host"] for d in reg if d.get("host")] + EXTRA_SEEDS
    seen, out = set(), []
    for h in hosts:
        if h not in seen:
            seen.add(h)
            out.append(f"https://{h}/")
    out.extend(EXTRA_SEED_URLS)  # 딥 페이지 직접 시드(셔틀·교통)
    return out


def crawl(max_pages: int, max_depth: int, delay: float,
          chunk_size: int, overlap: int, min_chars: int) -> dict:
    start = seeds()
    q: deque[tuple[str, int]] = deque((u, 0) for u in start)
    visited: set[str] = set()
    for u, _ in list(q):
        nu = normalize(u)
        if nu:
            visited.add(nu)
    chunks: list[dict] = []
    last_host_time: dict[str, float] = {}
    stats = {"visited": 0, "saved_pages": 0, "filtered": 0, "errors": 0,
             "too_short": 0, "504": 0, "queued_seeds": len(start)}
    t0 = time.time()

    while q and stats["visited"] < max_pages:
        url, depth = q.popleft()
        host = urlparse(url).netloc.lower()
        # 호스트별 politeness delay
        wait = delay - (time.time() - last_host_time.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        last_host_time[host] = time.time()

        body, title, links, tag = extract(url, timeout=8.0)
        stats["visited"] += 1
        if body is None:
            # fetch 실패(error) vs 품질필터(filtered) 구분. 둘 다 링크는 따라감(허브 가능)
            stats["errors" if tag == "error" else "filtered"] += 1
        elif len(body) < min_chars:
            stats["too_short"] += 1  # 본문 너무 짧음 → 저장 안 함, 링크는 확장
        else:
            stats["saved_pages"] += 1
            h = hashlib.sha1(url.encode()).hexdigest()[:12]
            for idx, ck in enumerate(chunk_text(body, chunk_size, overlap)):
                if len(ck.strip()) < min_chars:
                    continue
                chunks.append({
                    "doc_id": f"domain::{h}::{idx}",
                    "content": ck,
                    "title": title,
                    "source_url": url,
                    "file_type": "html",
                    "metadata": {"depth": depth, "host": host},
                })

        if depth < max_depth:
            for nxt in links:
                nu = normalize(nxt)
                if nu and nu not in visited:
                    visited.add(nu)
                    q.append((nxt, depth + 1))

        if stats["visited"] % 50 == 0:
            print(f"[crawl] 방문 {stats['visited']} | 저장페이지 {stats['saved_pages']} "
                  f"| 청크 {len(chunks)} | 큐 {len(q)} | {time.time()-t0:.0f}s",
                  flush=True)

    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["total_chunks"] = len(chunks)
    stats["queue_remaining"] = len(q)
    STORAGE.mkdir(parents=True, exist_ok=True)
    tmp = CHUNKS_OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, CHUNKS_OUT)
    print(f"[crawl] 완료 → {CHUNKS_OUT} ({len(chunks)} 청크)")
    print(f"[crawl] stats: {json.dumps(stats, ensure_ascii=False)}")
    return stats


def build(device: str) -> int:
    """torch+KURE 노드에서만. 별도 인덱스(domain.bin)로 빌드."""
    from cnubot.module1_indexer import build_vector_db  # lazy: faiss/torch
    from cnubot.schemas import IndexBuildConfig
    cfg = IndexBuildConfig(
        data_path=str(CHUNKS_OUT), save_path=str(INDEX_OUT), device=device)
    report = build_vector_db(cfg)
    print(report.model_dump_json(indent=2))
    return 0 if report.ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["crawl", "build", "all"], default="crawl")
    ap.add_argument("--max-pages", type=int,
                    default=int(os.environ.get("CRAWL_MAX_PAGES", "5000")))
    ap.add_argument("--depth", type=int,
                    default=int(os.environ.get("CRAWL_DEPTH", "3")))
    ap.add_argument("--delay", type=float,
                    default=float(os.environ.get("CRAWL_DELAY", "0.7")))
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=80)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--device", default="cuda:1")  # build 단계용
    args = ap.parse_args()

    if args.stage in ("crawl", "all"):
        print(f"=== 도메인 크롤 시작 (max_pages={args.max_pages}, depth={args.depth}, "
              f"delay={args.delay}s) ===")
        crawl(args.max_pages, args.depth, args.delay,
              args.chunk_size, args.overlap, args.min_chars)
    if args.stage in ("build", "all"):
        print(f"=== 벡터 빌드 (device={args.device}) ===")
        return build(args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
