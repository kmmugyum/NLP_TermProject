# cnubot 패키지 진입점 — Colab Free 환경의 라이브 fetch 가속을 자동 활성화
# (persistent HTTP/2 client + 3-stage retry + Drive 디스크 캐시)
# 환경변수 CNU_NET_DISABLE=1 로 비활성 가능.
from . import _net  # noqa: F401  (side-effect: httpx.get/head monkey-patch)
