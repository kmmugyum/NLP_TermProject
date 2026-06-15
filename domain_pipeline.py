#!/usr/bin/env python3
"""도메인 청크 자동 병합 파이프라인 (크롤 이후 단계) — build → dedup → A/B 게이트 → 병합·commit.

crawl_domain.py --stage crawl 이 만든 storage/_domain_chunks.json 을 받아:
  1) 현재 production 인덱스(academic_v2_bin.zip)를 펼쳐 baseline 로드
  2) dedup: 이미 인덱스에 있는 청크(source_url+content 해시) 제외 → 신규만
  3) baseline 에서 grounded 질문의 top1 청크 idx 기록
  4) 신규 청크 임베딩 후 add_chunks → merged top5 에서 baseline top1 이 빠지면 회귀로 간주
  5) 회귀 0 이면: merged 인덱스를 academic_v2.bin + meta 로 저장, zip 재패키징, git commit·push
     회귀 있으면: 중단(production zip 미변경) + 로그

봇 코드(추론/라우팅)는 일절 수정하지 않는다. 산출물(zip+meta)만 갱신 → 봇 재클론 시 반영.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/Disk1/cnu_env/hf_cache")
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
STORAGE = REPO / "src" / "cnubot" / "storage"
ZIP = REPO / "academic_v2_bin.zip"
META = STORAGE / "academic_v2.bin.meta.json"          # 추적되는 meta (zip 과 짝)
DOMAIN_CHUNKS = STORAGE / "_domain_chunks.json"
WORK_IDX = STORAGE / "academic_real.bin"               # 작업용(추출본, gitignore)
WORK_META = STORAGE / "academic_real.bin.meta.json"

# 회귀 게이트: 이 질문들의 baseline top1 청크가 merged top5 에서 빠지면 병합 중단.
GROUNDED = [
    "휴학은 최대 몇 학기까지 할 수 있어?",
    "복수전공이랑 부전공 차이가 뭐야?",
    "성적 장학금 받으려면 평점 몇 이상이어야 해?",
    "미적분학 몇 학년 과목이야?",
    "충남대 졸업 이수학점 최소 몇 학점?",
    "학사경고 연속이면 몇 번부터 유급/제적이에요?",
    "사회학과에서 컴공으로 전과하려면 성적요건이 어떻게 돼?",
    "대학원 석사 졸업하려면 수료학점 몇 학점이야?",
    "의과대학 졸업하려면 몇 학점 들어야 해?",
    "등록금 납부는 어떻게 해요?",
    "재수강은 몇 학점까지 가능해?",
    "졸업요건이 어떻게 돼?",
    "수강신청은 언제 해?",
    "복수전공 최소 몇 학점 이수해야 해?",
]


def _key(c: dict) -> str:
    return hashlib.md5(
        ((c.get("source_url") or "") + "" + (c.get("content") or "")).encode("utf-8")
    ).hexdigest()


def log(m): print(m, flush=True)


def main() -> int:
    if not DOMAIN_CHUNKS.is_file():
        log("[skip] _domain_chunks.json 없음 — 크롤 먼저 필요"); return 1
    if not ZIP.is_file() or not META.is_file():
        log("[err] production zip/meta 없음"); return 1

    # 1) production 인덱스 펼치기 (baseline)
    with zipfile.ZipFile(ZIP) as z:
        binname = next((n for n in z.namelist() if n.endswith("academic_v2.bin")),
                       z.namelist()[0])
        with z.open(binname) as src, open(WORK_IDX, "wb") as dst:
            shutil.copyfileobj(src, dst)
    shutil.copy2(META, WORK_META)

    from cnubot.module1_indexer import KUREEmbedder
    from cnubot.module3_retriever import AcademicRetriever
    import numpy as np

    emb = KUREEmbedder("nlpai-lab/KURE-v1", "cuda:0")
    r = AcademicRetriever(str(WORK_IDX), str(WORK_META), embedder=emb, top_k=5)
    base_n = r.index.ntotal
    log(f"[1] baseline {base_n} 청크")

    # 2) dedup — 이미 인덱스에 있는 청크 제외
    existing = {_key(m) for m in r.meta}
    dom = json.load(open(DOMAIN_CHUNKS))
    new = [c for c in dom if _key(c) not in existing]
    log(f"[2] 크롤 {len(dom)} → 신규 {len(new)} (중복 {len(dom)-len(new)} 제외)")
    if not new:
        log("[done] 신규 청크 0 — 병합 불필요, 종료"); return 0

    # 3) baseline top1 idx
    def top_idxs(q, k):
        qv = np.ascontiguousarray(emb.encode([q]), dtype=np.float32)
        _, idxs = r.index.search(qv, k)
        return [int(i) for i in idxs[0] if i >= 0]
    base_top1 = {q: (top_idxs(q, 1) or [-1])[0] for q in GROUNDED}

    # 4) 신규 병합(in-memory) + 게이트
    added = r.add_chunks(new)
    log(f"[3] add_chunks +{added} → {r.index.ntotal}")
    fails = [q for q in GROUNDED
             if base_top1[q] >= 0 and base_top1[q] not in top_idxs(q, 5)]
    if fails:
        log(f"[GATE FAIL] 회귀 {len(fails)}건 — 병합 중단(production 미변경):")
        for q in fails: log(f"    - {q}")
        return 2
    log(f"[4] GATE PASS — grounded {len(GROUNDED)}개 모두 top1 유지")

    # 5) persist + zip 재패키징 + commit
    import faiss
    faiss.write_index(r.index, str(STORAGE / "academic_v2.bin"))
    json.dump(r.meta, open(META, "w"), ensure_ascii=False)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(str(STORAGE / "academic_v2.bin"), "academic_v2.bin")
    log(f"[5] persist + zip 재패키징 ({ZIP.stat().st_size//1024//1024}MB), 총 {r.index.ntotal}청크")

    msg = (f"data(index): 도메인 신규 {added}청크 자동 병합 "
           f"({base_n}→{r.index.ntotal}) [pipeline, gate PASS]")
    try:
        subprocess.run(["git", "add", str(ZIP.relative_to(REPO)),
                        str(META.relative_to(REPO))], cwd=REPO, check=True)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
            log("[git] 변경 없음 — commit 생략"); return 0
        subprocess.run(["git", "commit", "-m", msg], cwd=REPO, check=True)
        subprocess.run(["git", "fetch", "origin", "--quiet"], cwd=REPO, check=True)
        subprocess.run(["git", "rebase", "origin/main"], cwd=REPO, check=True)
        subprocess.run(["git", "push"], cwd=REPO, check=True)
        log(f"[git] push 완료: {msg}")
    except subprocess.CalledProcessError as e:
        log(f"[git] 실패 → {e} (산출물은 디스크에 있음, 다음 실행 재시도)")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
