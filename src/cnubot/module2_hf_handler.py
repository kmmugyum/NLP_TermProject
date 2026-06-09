"""M2 라우터 LLM 경로 구현체 — M4와 동일한 Qwen 인스턴스 공유.

HFIntentLLM 은 자체 모델을 로드하지 않고, 주입된 backend(=M4 HFAnswerLLM)의
generate 를 재사용한다 → GPU0 에 7B 1개만 적재(M2/M4 공유, 메모리 경합 0).

regex JSON 가드: 모델이 JSON 앞뒤로 잡문을 붙여도 {...} 블록만 추출.
불완전 JSON(종결 } 유실)은 json.loads 에서 실패 → 예외 전파 → 라우터가 fallback.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from .module2_router import SYSTEM_PROMPT

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)  # greedy: 첫 { ~ 마지막 } (후행 잡문 차단)


class GenBackend(Protocol):
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str: ...


class HFIntentLLM:
    """router 가 기대하는 classify(query)->dict 계약 구현. backend 공유."""

    def __init__(self, backend: GenBackend, max_new_tokens: int = 256):
        # max_new_tokens 마진: refined_query+reason 이 잘려 } 유실되지 않도록 여유 확보.
        self.backend = backend
        self.max_new_tokens = max_new_tokens

    def classify(self, query: str) -> dict:
        prompt = f"{SYSTEM_PROMPT}\n\nQuery: {query}\n\nJSON:"
        raw = self.backend.generate(prompt, max_new_tokens=self.max_new_tokens)
        m = _JSON_RE.search(raw)
        if not m:
            raise ValueError(f"JSON 블록 추출 실패: {raw[:80]!r}")
        return json.loads(m.group(0))  # 불완전 JSON → JSONDecodeError 전파 (→ 라우터 fallback)
