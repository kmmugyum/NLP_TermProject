# 핸드오프: 연구서버 → 로컬(GPU노드) — 학사 청크 v2 교체

> 연구서버(한국IP)가 학사 코퍼스 청킹 결함을 수정해 `_academic_chunks_v2.json`을 push함.
> 로컬은 **검증 → 백업 → 벡터 재빌드 → eval → 교체** 순으로 진행할 것.
> 연구서버-로컬 직접 연결 없음. 소통은 이 repo(GitHub) 경유.

## 배경 (연구서버가 측정·수정한 것)
기존 `_academic_chunks.json`(29,794 청크) 실측 결함 2가지 → v2에서 0%로 수정:
- **완전 중복 39.5%(11,782)** : 페이지단위 dedup만 있어 공통 nav/사이드바가 청크마다 복제됨(library 사이드바 217회 등). → 청크-레벨 content dedup + nav 링크줄 제거로 해결.
- **빈 파이프 표 9.5%(2,832)** : `| |||||||` 무한 반복으로 LLM이 표 못 읽음(졸업학점 환각 직접 원인 추정). → `\|(\s*\|)+` 압축 + 잡표행 드롭으로 해결.

| 지표 | BEFORE(prod) | AFTER(v2) |
|---|---|---|
| 청크 | 29,794 | 6,642 |
| 완전중복 | 39.5% | **0.0%** |
| 빈 파이프 표 | 9.5% | **0.0%** |
| <200자 | 512 | 0 |
| 평균/최대 길이 | 1054/2001 | 1029/2000 |

상세: `RECHUNK_V2_REPORT.md`. 청커: `rechunk_v2.py`.

## ⚠️ 한계(반드시 인지)
- v2는 **기존 raw 덤프(`*_out.jsonl`, May 24) 재청킹**이지 재크롤 아님. 정적 안내페이지라 staleness 무방.
- **학과 커버리지는 덤프 한계 그대로** — socio.cnu가 22% 편중, 고유URL 1,289(prod 3,838은 대부분 게시판/SNS/검색 잡URL). 전 학과 균등 아님.
- 청크 수 감소는 **정당**(중복·nav·게시판 제거)이나, 특정 학과 정적정보 누락 가능 → 아래 follow-up 참조.

## 로컬이 할 일 (순서대로)

### 1. 받기
```bash
git pull   # 또는 raw fetch
# 대상: src/cnubot/storage/_academic_chunks_v2.json
```

### 2. 독립 검증 (연구서버 수치 신뢰하지 말고 직접 재측정)
```python
import json, hashlib, re
v2 = json.load(open("src/cnubot/storage/_academic_chunks_v2.json"))
c = [d["content"] for d in v2]; n=len(c)
norm=lambda s: re.sub(r"\s+"," ",s).strip()
dup = n - len(set(hashlib.md5(norm(x).encode()).hexdigest() for x in c))
broken = sum(1 for x in c if re.search(r"\|(\s*\|){3,}", x))
print(f"청크 {n} | 중복 {dup} | 빈파이프 {broken}")  # 기대: 중복 0, 빈파이프 0
```

### 3. 환각 쿼리 대조 (핵심)
졸업이수학점 등 기존 환각 질문에 대해 v2 청크에서 관련 표가 깔끔히 검색되는지 확인.
기존 실패 케이스(`batch_eval_*` / anchor 질문)로 before/after 응답 비교.

### 4. 백업 (절대 덮어쓰기 전 필수)
```bash
cp src/cnubot/storage/_academic_chunks.json   _academic_chunks.json.bak_$(date +%Y%m%d)
cp src/cnubot/storage/academic_real.bin       academic_real.bin.bak_$(date +%Y%m%d)
cp src/cnubot/storage/academic_real.bin.meta.json academic_real.bin.meta.json.bak_$(date +%Y%m%d)
```

### 5. 벡터 재빌드 (KURE-v1, GPU노드)
`module1_indexer.build_vector_db()` 로 `_academic_chunks_v2.json` → 새 `.bin`.
data_path 를 v2 로 지정. cuda:1 정책 유지.

### 6. eval 후에만 교체
재빌드 인덱스로 bot eval(POPE 무관, 학사 QA 정확도) 통과 확인 후:
```bash
mv _academic_chunks_v2.json src/cnubot/storage/_academic_chunks.json
# .bin 도 교체
```
eval에서 졸업학점 환각이 안 고쳐지면 **교체하지 말고** 연구서버에 피드백(이 repo에 이슈/커밋).

## Follow-up (선택, 커버리지 보강이 필요하면)
연구서버에 "dept_registry.json 79개 학과 와이드 재크롤" 요청.
- `url_discover.discover()` + `collect_seeds()`, 공지/게시판 URL 제외, BFS depth 3, max 5000, delay 0.5~1.0s.
- 새 raw 덤프 → `rechunk_v2.py` 동일 적용 → 커버리지 균등화.
- 이건 **별건**(이번 결함수정과 분리). 현재 v2로 환각이 잡히면 불필요할 수도.
