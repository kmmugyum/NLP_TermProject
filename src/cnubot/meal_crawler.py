"""학식 데이터 소스 크롤러 (Colab 배치가 돌릴 부분, 서버와 분리).

소스: mobileadmin.cnu.ac.kr/food/index.jsp — 서버 렌더 HTML(httpx+bs4, Playwright 불필요).
실DOM: table[0] = 구분(끼니, rowspan=2) × 대상(직원/학생) × 제1~4학생회관·생활과학대학.
       셀 = <li><h3 class=menu-tit03>정식(가격)</h3><p>요리<br/>요리…</p></li>.
주의: '메뉴운영내역' 셀이 rowspan=100(소스 버그)이라 열을 가림 → rowspan 그리드 전개 필수.
출력: schemas.MealCache (서버 CafeteriaRetriever.update_cache_directly 로 주입 가능).
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import httpx
from bs4 import BeautifulSoup

from .schemas import DailyMenu, MealCache, WeeklyMealCache

MEAL_URL = "https://mobileadmin.cnu.ac.kr/food/index.jsp"
_EN_PAREN = re.compile(r"\([A-Za-z][A-Za-z ]*\)")  # (pork included) — 영문주석만 제거(가격 보존)
_MEAL_TYPES = {"조식", "중식", "석식"}
_TARGETS = {"직원", "학생"}


def fetch_meal_html(url: str = MEAL_URL, timeout: float = 10.0,
                    target: date | None = None) -> str:
    """target 지정 시 해당 날짜(searchYmd='YYYY.MM.DD')의 식단 페이지. 미지정 시 오늘."""
    params = None
    if target is not None:
        params = {"searchYmd": target.strftime("%Y.%m.%d"), "searchView": "cafeteria",
                  "searchLang": "OCL04.10", "searchCafeteria": "OCL03.02"}
    r = httpx.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"},
                  timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.text


def _clean_dish(s: str) -> list[str]:
    s = _EN_PAREN.sub("", s).strip()
    return [p.strip() for p in re.split(r"[&/]", s) if p.strip()]  # 연두부&양념장 → 둘


def _parse_cell(td) -> list[list[str]]:
    """셀 → 코스별 menu_list 리스트. 운영안함/메뉴운영내역(li 없음) → []."""
    courses = []
    for li in td.find_all("li"):
        h3 = li.find("h3")
        course = h3.get_text(strip=True) if h3 else ""
        dishes: list[str] = []
        p = li.find("p")
        if p:
            for piece in p.get_text("\n").split("\n"):  # <br> → \n
                if piece.strip():
                    dishes.extend(_clean_dish(piece))
        menu_list = ([course] if course else []) + dishes
        if menu_list:
            courses.append(menu_list)
    return courses


def _build_grid(rows) -> dict[tuple[int, int], object]:
    """rowspan/colspan 전개 → (row, col) → 점유 td. 가려진 열 정확 매핑."""
    occ: dict[tuple[int, int], object] = {}
    for r, tr in enumerate(rows):
        c = 0
        for cell in tr.find_all(["th", "td"]):
            while (r, c) in occ:
                c += 1
            rs = int(cell.get("rowspan") or 1)
            cs = int(cell.get("colspan") or 1)
            for dr in range(rs):
                for dc in range(cs):
                    occ[(r + dr, c + dc)] = cell
            c += cs
    return occ


def parse_meal_html(html: str) -> MealCache:
    soup = BeautifulSoup(html, "lxml")

    term = soup.select_one("span.term") or soup.select_one("div.menu-top")
    m = re.search(r"(20\d\d)\.(\d\d)\.(\d\d)", (term.get_text() if term else "") or "")
    target = date(int(m[1]), int(m[2]), int(m[3])) if m else date.today()

    table = soup.find("table")
    rows = table.find_all("tr")
    # 헤더: [구분(colspan2), 제1, 제2, 제3, 제4, 생활과학] → 식당명은 [1:] (그리드 col 2~)
    places = [th.get_text(strip=True) for th in rows[0].find_all("th")][1:]

    occ = _build_grid(rows)
    menus: list[DailyMenu] = []
    for r in range(1, len(rows)):
        meal_c, tgt_c = occ.get((r, 0)), occ.get((r, 1))
        if meal_c is None or tgt_c is None:
            continue
        meal_t, tgt_t = meal_c.get_text(strip=True), tgt_c.get_text(strip=True)
        if meal_t not in _MEAL_TYPES or tgt_t not in _TARGETS:
            continue
        for i, place in enumerate(places):
            td = occ.get((r, 2 + i))
            if td is None:
                continue
            for menu_list in _parse_cell(td):
                menus.append(DailyMenu(
                    place=place, meal_type=meal_t, target=tgt_t, menu_list=menu_list))
    return MealCache(timestamp=datetime.now(), target_date=target, menus=menus)


def crawl_meal(url: str = MEAL_URL) -> MealCache:
    return parse_meal_html(fetch_meal_html(url))


def crawl_week(url: str = MEAL_URL, base: date | None = None) -> WeeklyMealCache:
    """현재 주 월~금 식단을 searchYmd로 순회 수집 → WeeklyMealCache.
    주말은 운영 안 함(빈 메뉴)이라 월~금만. 빈 날짜는 days에서 생략.

    Colab 등 한국 서버 접속이 느린 환경 대비: 개별 날짜 실패 시 silent skip 하되
    원인을 로그로 남겨(전부 실패 시 빈 days 반환의 진짜 이유 추적 가능) 진단을 돕는다."""
    base = base or date.today()
    week_start = base - timedelta(days=base.weekday())  # 이번 주 월요일
    days: dict[str, list[DailyMenu]] = {}
    fail_count = 0
    last_err: Exception | None = None
    for off in range(5):  # 월~금
        d = week_start + timedelta(days=off)
        try:
            mc = parse_meal_html(fetch_meal_html(url, target=d))
            if mc.menus:
                days[d.isoformat()] = mc.menus
        except Exception as e:  # 개별 날짜 실패는 건너뜀(부분 수집 허용)
            fail_count += 1
            last_err = e
            print(f"[meal] {d.isoformat()} 크롤 실패 → {type(e).__name__}: {e}")
    if fail_count == 5 and last_err is not None:
        # 5일 전부 실패 = 네트워크/접속 문제(Colab→mobileadmin.cnu.ac.kr 차단 등). 원인 명시.
        print(f"[meal] 주간 크롤 전체 실패({fail_count}/5). "
              f"원인 추정: 서버 접속 불가/타임아웃. 마지막 에러: {type(last_err).__name__}: {last_err}")
    return WeeklyMealCache(timestamp=datetime.now(), week_start=week_start, days=days)
