"""공지 RAG (온디맨드): 질의 → 대상 학과 보드 라이브 fetch → 최신 공지 파싱.

학사 코퍼스(정적)에 없는 '최신/최근 공지'를 답하기 위해, 질의 시점에 학과 공지
게시판을 직접 긁는다(캐시 없음 = 항상 최신). 학과 해석은 dept_registry.json 사용.
보드는 서버렌더(그누보드류, articleNo/mode=view)라 httpx+bs4로 충분(Playwright 불필요).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from .schemas import NoticeItem

_PKG = Path(__file__).resolve().parent
_REGISTRY_PATH = _PKG / "data" / "dept_registry.json"
_DEFAULT_HOST = "computer.cnu.ac.kr"  # 학과 미지정 질의 기본값(연구 맥락: CS/AI)
_ART_RE = re.compile(r"(?:articleNo|wr_id|nttId|seq)=(\d+)")
_DATE_RE = re.compile(r"(?:20)?\d\d[.\-]\d\d?[.\-]\d\d?")
_HEAD = {"User-Agent": "Mozilla/5.0"}
_SUFFIX = ("교육과", "학부", "대학원", "대학", "전공", "과", "부")  # 후행 1개만 제거


def _nospace(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _dept_core(name: str) -> str:
    """학과명에서 후행 접미('과/학부/대학' 등) 1개만 제거 → 핵심어.
    '수학과'→'수학', '컴퓨터인공지능학부'→'컴퓨터인공지능', '물리학과'→'물리학'."""
    s = _nospace(name)
    for suf in _SUFFIX:
        if s.endswith(suf) and len(s) > len(suf) + 1:
            return s[: -len(suf)]
    return s


_PAGE_CACHE: dict[str, tuple[float, "str | None"]] = {}  # P8: url→(ts, text|None)
_PAGE_TTL = 600.0  # 초
_NONHTML_EXT = (".pdf", ".hwp", ".hwpx", ".doc", ".docx", ".xls", ".xlsx",
                ".ppt", ".pptx", ".zip", ".jpg", ".png", ".gif", ".mp4")
_LOGIN_RE = re.compile(r"login|signin|sso|auth|nidlogin", re.I)


def fetch_page_text(url: str, max_chars: int = 2000, timeout: float = 6.0) -> str | None:
    """임의 CNU 페이지의 본문 텍스트를 추출(adaptive 읽기용). 실패/부적합 시 None.
    가드: 비HTML·에러상태·로그인/홈 리다이렉트·빈셸·초대형·캐시(P1~P10)."""
    if not url or not url.startswith("http"):
        return None
    if url.lower().rsplit("?", 1)[0].endswith(_NONHTML_EXT):  # P4: 비HTML 링크
        return None
    import time as _t
    hit = _PAGE_CACHE.get(url)  # P8: 캐시(히트/미스 모두 캐싱 → 재fetch·재차단 방지)
    if hit and _t.time() - hit[0] < _PAGE_TTL:
        return hit[1]

    def _cache(v):
        _PAGE_CACHE[url] = (_t.time(), v)
        return v

    try:
        r = httpx.get(url, headers=_HEAD, timeout=timeout, verify=False,
                      follow_redirects=True)
    except Exception:
        return _cache(None)
    if r.status_code != 200:  # P5: 404/403/500 등 에러페이지 근거 금지
        return _cache(None)
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype.lower():  # P4: 비HTML 응답
        return _cache(None)
    if len(r.content) > 5_000_000:  # P10: 초대형 페이지
        return _cache(None)
    fp = str(r.url)
    if _LOGIN_RE.search(fp):  # P3: 로그인으로 redirect
        return _cache(None)
    # P7: 깊은 페이지를 요청했는데 홈(루트)으로 redirect → 페이지 유실로 간주
    try:
        from urllib.parse import urlparse
        if len(urlparse(url).path.strip("/")) > 1 and urlparse(fp).path.strip("/") in (
                "", "index.do", "main.do", "index.jsp", "html/kr"):
            return _cache(None)
    except Exception:
        pass
    # P2: bytes로 charset 자동 감지(EUC-KR 등 mojibake 방지)
    soup = BeautifulSoup(r.content, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "aside"]):  # P6
        tag.decompose()
    el = (soup.select_one("div.fr-view") or soup.select_one("div.b-content-box")
          or soup.select_one("#content") or soup.select_one("div.content")
          or soup.select_one("[role=main]") or soup.select_one("main") or soup.body)
    if not el:
        return _cache(None)
    txt = " ".join(el.get_text(" ").split())
    if len(txt) < 80:  # P1: 빈 셸/저신호(JS 미렌더)
        return _cache(None)
    # P3: 본문이 로그인 안내 위주면 거부
    if len(txt) < 500 and re.search(
            r"로그인.{0,6}(필요|하세요|해주)|아이디.{0,4}비밀번호|please\s+log\s*in|sign\s*in", txt, re.I):
        return _cache(None)
    return _cache(txt[:max_chars])


class NoticeService:
    def __init__(self, registry_path: str | Path = _REGISTRY_PATH,
                 default_host: str = _DEFAULT_HOST, timeout: float = 10.0):
        self.timeout = timeout
        self.default_host = default_host
        reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        self.depts = [d for d in reg if d.get("board")]  # 보드 있는 학과만

    def resolve_depts(self, query: str) -> list[dict]:
        """질의에 학과명 핵심어가 들어있으면 해당 학과(들, 긴 매칭 우선), 없으면 기본(CS/AI)."""
        qn = _nospace(query)
        hits = []
        for d in self.depts:
            core = _dept_core(d.get("name") or "")
            if len(core) >= 2 and core in qn:
                hits.append((len(core), d))
        if hits:
            hits.sort(key=lambda x: -x[0])  # 긴 핵심어 매칭 우선(예: '컴퓨터인공지능')
            return [d for _, d in hits[:2]]
        return [d for d in self.depts if d["host"] == self.default_host] or self.depts[:1]

    def _sibling_boards(self, board: str) -> list[str]:
        """학사공지(bachelor.do) ↔ 일반공지(notice.do) 형제 보드도 함께(사업단소식 등 커버)."""
        boards = [board]
        for a, b in [("bachelor.do", "notice.do"), ("notice.do", "bachelor.do")]:
            if board.endswith(a):
                boards.append(board[: -len(a)] + b)
        return boards

    def _fetch_board(self, url: str, dept: str, limit: int) -> list[NoticeItem]:
        import time as _t
        _t0 = _t.time()
        try:
            r = httpx.get(url, headers=_HEAD, timeout=self.timeout,
                          verify=False, follow_redirects=True)
            r.raise_for_status()
        except Exception as e:
            print(f"[notice-debug] _fetch_board 실패 ({_t.time()-_t0:.1f}s) "
                  f"url={url} → {type(e).__name__}: {e}")
            return []
        print(f"[notice-debug] _fetch_board OK ({_t.time()-_t0:.1f}s) "
              f"url={url} status={r.status_code} {len(r.text)}bytes")
        items: list[NoticeItem] = []
        for tr in BeautifulSoup(r.text, "lxml").find_all("tr"):
            a = tr.find("a", href=re.compile(r"articleNo|mode=view|wr_id|nttId|seq="))
            if not a:
                continue
            title = a.get_text(" ", strip=True)
            if len(title) < 4:
                continue
            href = str(httpx.URL(url).join(a.get("href", "")))
            row = tr.get_text(" ", strip=True)
            m_art = _ART_RE.search(href)
            m_date = _DATE_RE.search(row)
            items.append(NoticeItem(
                title=title[:120], url=href, dept=dept,
                posted=m_date.group(0) if m_date else None,
                article_no=int(m_art.group(1)) if m_art else None))
            if len(items) >= limit:
                break
        print(f"[notice-debug] _fetch_board 파싱: {len(items)}건 "
              f"(0건이면 게시판 HTML 구조 변경 의심)")
        return items

    def fetch_body(self, url: str, max_chars: int = 1400) -> str | None:
        """게시글 본문 추출(CNU CMS는 div.fr-view = Froala 본문). 실패 시 None."""
        try:
            r = httpx.get(url, headers=_HEAD, timeout=self.timeout,
                          verify=False, follow_redirects=True)
            soup = BeautifulSoup(r.text, "lxml")
        except Exception:
            return None
        el = soup.select_one("div.fr-view") or soup.select_one("div.b-content-box")
        if not el or len(el.get_text(strip=True)) < 30:
            return None
        return " ".join(el.get_text(" ").split())[:max_chars]

    @staticmethod
    def _tokens(s: str) -> list[str]:
        """매칭용 distinctive 토큰: 영숫자 4+ 또는 한글 3+."""
        return [t.lower() for t in re.findall(r"[A-Za-z0-9]{4,}|[가-힣]{3,}", s or "")]

    def best_title_match(self, query: str, items: list[NoticeItem]) -> NoticeItem | None:
        """질의의 distinctive 토큰이 제목에 들어가는 공지 1건(특정 공지 디테일 질문용)."""
        top = self.top_title_matches(query, items, limit=1)
        return top[0] if top else None

    def top_title_matches(self, query: str, items: list[NoticeItem],
                          limit: int = 3) -> list[NoticeItem]:
        """질의의 distinctive 토큰이 제목에 들어가는 공지를 점수 내림차순으로 상위 N건.
        같은 행사에 여러 공지(일정/장소/프로그램)가 있을 때 본문 통합용."""
        qtoks = set(self._tokens(query)) - {"공지", "소식", "알려줘", "뭐있어", "있어"}
        if not qtoks:
            return []
        scored = []
        for it in items:
            tl = it.title.lower()
            score = sum(1 for t in qtoks if t in tl)
            if score >= 1:
                scored.append((score, it))
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:limit]]

    def collect(self, query: str, per_board: int = 12) -> tuple[str, list[NoticeItem]]:
        """질의의 학과 보드(+형제)를 라이브 fetch → 최신순 병합."""
        depts = self.resolve_depts(query)
        label = " / ".join(d["name"] for d in depts) or "충남대"
        seen, items = set(), []
        for d in depts:
            for b in self._sibling_boards(d["board"]):
                for it in self._fetch_board(b, d["name"], per_board):
                    key = it.article_no or it.title
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(it)
        # 최신순: article_no 내림차순 (없으면 뒤로)
        items.sort(key=lambda x: x.article_no or -1, reverse=True)
        return label, items
