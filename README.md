# 26-2 DSL Modeling — RecSys (생성형 추천)

Semantic ID 기반 생성형 추천(TIGER)이 **신규·비인기(콜드) 아이템을 추천하지 못하는 문제**를 다룹니다.
아이템 주소(SID)를 텍스트 대신 **그래프로 증강(G-SID)** 하면 나아진다는 가설을
Amazon Beauty 와 자체 구축한 구인구직 데이터셋, 두 도메인에서 검증합니다.

> 📌 **발표 준비라면 [`docs/00_발표_가이드.md`](docs/00_발표_가이드.md)**, 문서를 찾는다면 [`docs/README.md`](docs/README.md) 부터 보세요.
> 이 README 는 **지금 상태와 진입점**만 적습니다. 수치의 정본은 각 결과 문서입니다. 최종 갱신 2026-09-19.

---

## 1. 지금까지 확인된 것 (2026-09-19)

판정 규칙은 실행 전에 노션 계획 페이지에 고정했고, 아래는 그 규칙으로 자동 채점한 결과입니다.
원문은 `results/reports/`, 지표 전체 표는 `docs/RESULTS_요약_0920.md`.

| 주장 | 근거 | 판정 |
|---|---|---|
| **Beauty: 그래프 주소 > 텍스트 주소** | LOO 5시드 NDCG@10 +0.0044, 5/5 부호 일치, 노이즈 바닥 1.6배. COLD@50 +48% | ✅ 확정 |
| **세 트랙(LOO·W4·W5) 3시드 재현** | Temporal W4 +0.0024, W5 +0.0022, 각 3/3 부호 일치·바닥 초과 | ✅ 확정 |
| 튜닝식 β₀1.0 > 고전 블렌딩 α0.7 | 5시드 +0.0006, 사전 고정 바닥 0.0016 아래 | ❌ 구분 불가로 종결 |
| 다양성(APLT@50) 개선 | 5시드 평균차가 바닥 아래 | ⬜ 측정 불가 유지 |
| **구인구직 v3: 주지표 NDCG@10** | 6시드 짝지은 t 1.34 (기준 2.015). 시드 99 에서 부호 반전. recall@50·ndcg@50 은 유의 | ❌ 미확인 유지 |
| **구인구직 v4 (캐글 분포 재보정) 재현** | 6시드 짝지은 t: recall@10·ndcg@10·recall@50·ndcg@50 유의, NDCG@10 t 11.4. G1 무작위 그래프 대조 7/7 통과 | ✅ 재현 (사전등록 바닥 방식은 n=6 에서 recall@50 만 통과 — 둘 다 보고) |
| 이득이 학습 예산과 무관 (15k·45k) | 15k +0.0061, 45k +0.0054 (NDCG@10, 각 2/2 부호 일치). 30k 5시드 +0.0044 와 같은 방향 | ✅ 통과 (규칙 D) |
| 접두사 제약 디코딩 (Beauty, 3시드) | 무효 SID 1.3~1.7% → 0, 정확도 −0.1%, COLD@50 +3~4%. zero-shot 은 텍스트·그래프 모두 0 (정답 138개) | 균열 없음. 그래프 우위는 제약 ON 에서도 유지 |
| zero-shot 벽 | 원인은 SID 할당 커버리지. 2계층 접두사 확장으로 0 → 0.061 (전체 −15% 대가) | 원인 특정, 미해결 |

**실패한 대조 2건**(CRAB 코드북 재균형, 확산 다이얼)과 그 근거는 `docs/SHARE_GSID_결과요약.md` §8.
**인용하면 안 되는 낡은 수치**는 `docs/00_발표_가이드.md` 4절. 규칙: 1시드 수치는 인용하지 않습니다.

---

## 2. 세 갈래 작업과 정본 문서

| 갈래 | 무엇 | 정본 | 상태 |
|---|---|---|---|
| **실험** | G-SID 신뢰도 (시드 확장 P1·P3, Temporal P2, 예산 P4, 접두사 P5) | `docs/SHARE_GSID_결과요약.md` · `docs/RESULTS_실험표.md` · `results/reports/` | **P1~P5 전부 완료 (9/19)** |
| **데이터셋** | 구인구직 생성 데이터셋 v3 → v4 (캐글 분포 재보정, 게이트 G0~G4) | `data_gen/README.md` · `docs/DATASET_v4_파이프라인.md` · `docs/JOBS_결과정리.md` | v4 n=6 완료, G2·G4 미실시 |
| **서비스** | 학회 행사 네트워킹 웹앱 (이음 커넥트, 10월 말) | 노션 3번 페이지 | 설계 단계 |

세 갈래의 계획과 일정은 노션 **2차 발표 준비 계획** 페이지(팀 내부)에 있습니다. 2차 발표는 2026-09-24~30.

---

## 3. 어디서부터 읽나

| 나는… | 읽을 것 |
|---|---|
| 결과 수치가 필요하다 | [`docs/RESULTS_요약_0920.md`](docs/RESULTS_요약_0920.md) — 전 실험 지표·판정 한 장 |
| 발표 자료를 만든다 | [`docs/00_발표_가이드.md`](docs/00_발표_가이드.md) → `SHARE_GSID_결과요약.md` · `JOBS_결과정리.md` |
| 행사 서비스 백엔드를 만든다 | [`docs/BACKEND_설계계획.md`](docs/BACKEND_설계계획.md) · [`event_demo/README.md`](event_demo/README.md) |
| Beauty 수치를 확인한다 | [`docs/RESULTS_실험표.md`](docs/RESULTS_실험표.md) (실험표 ①~⑤) · [`results/README.md`](results/README.md) (JSON 색인) |
| 구인구직 데이터셋을 이해한다 | [`data_gen/README.md`](data_gen/README.md) → [`docs/PREREG_생성데이터셋.md`](docs/PREREG_생성데이터셋.md) → [`docs/DATASET_v4_파이프라인.md`](docs/DATASET_v4_파이프라인.md) |
| 서버에서 실험을 제출한다 | [`docs/GPU_서버_사용가이드.md`](docs/GPU_서버_사용가이드.md) 🔒 · [`docs/SERVER_공동사용.md`](docs/SERVER_공동사용.md) · 제출 스크립트는 `code/run_*.sh` 머리 주석 |
| 코드를 고친다 | [`code/README.md`](code/README.md) — 의존 관계 · 실행 순서 · **반복해서 당한 함정 8가지** |
| 그래프 SID 를 만든다 | [`docs/HANDOFF_graph_sid.md`](docs/HANDOFF_graph_sid.md) · [`csv_export/sid/gstar_noTau_k0_b10/README.md`](csv_export/sid/gstar_noTau_k0_b10/README.md) (승자 수식) |
| MaskGR · CRAB 을 돌린다 | [`docs/HANDOFF_MaskGR.md`](docs/HANDOFF_MaskGR.md) · [`crab/README.md`](crab/README.md) |
| 문서가 너무 많다 | [`docs/README.md`](docs/README.md) — 전 문서 분류(정본 / 배경 / 운영 / 기록) |

---

## 4. 파이프라인 한 장

```
Amazon Beauty · 구인구직(생성)
        │  전처리                 preprocessing/ · data_gen/
        ▼  Split A (LOO) · Split B (Temporal W3/W4/W5)
        │  텍스트 → 임베딩        GRID (flan-t5-xl, 2048차원)
        ▼
   ┌────┴────┐  주소 만들기
 텍스트 SID   G-SID (그래프 직교 주입, β₀=1.0)   ← 승자. 고전 블렌딩·CRAB 은 대조
   └────┬────┘  GRID RQ-KMeans, 코드북 256, 4단계 + 충돌 구분자
        ▼  학습                  TIGER (code/server/tiger_beauty.sh) · MaskGR (대조)
        ▼  덤프 → 채점           tiger_eval_dump.sh → code/tiger_to_eval.py → results/*.json
        ▼  판정                  code/verdict_*.py  (시드 평균 · 노이즈 바닥 · 짝지은 t · G1)
```

**노이즈 바닥**: 같은 레시피에서 시드만 바꾼 실행들의 최대 간격. 이보다 작은 차이는 주장하지 않습니다.

---

## 5. 저장소 구조

```
docs/              보고서·인수인계·계획 (→ docs/README.md 에 분류표)
code/              평가 하네스 · 제출/채점/판정 스크립트 (→ code/README.md)
code/server/       서버 sbatch 스크립트
results/           지표 JSON · 자동 보고서 (→ results/README.md)
data_gen/          구인구직 생성 데이터셋 v3·v4 (→ data_gen/README.md)
preprocessing/     Beauty 전처리 (+ 설계·수정 문서)
csv_export/        엑셀용 CSV · SID 자산 설명 (→ csv_export/README.md)
crab/              CRAB 코드북 재균형 (대조군)
patches/           GRID 패치 기록
GRID/  MaskGR/     업스트림 스냅샷 (snap-research, .git 제거) — GRID 에 팀 패치 2건
_archive/          안 쓰지만 지우지 않은 것 (→ _archive/README.md)

Beauty_split_A.pkl · Beauty_split_B.pkl   분할 파일 ★ 코드 기본값이라 옮기지 말 것
```

---

## 6. 실행 — 빠른 길

```bash
make help                                   # 로컬에서 할 수 있는 것 (표 재생성, 베이스라인, 채점)
python code/tiger_to_eval.py --pred <덤프.pkl> --sid <sid.pt> --label <이름>     # 덤프 → 지표
python code/tiger_to_eval.py ... --split Beauty_split_B.pkl --window W5         # Temporal

# 서버 (자세한 옵션은 각 스크립트 머리 주석)
bash code/run_gsid_week.sh --dry            # P1·P3 시드 확장
bash code/run_p2_temporal.sh --dry          # P2 Temporal
bash code/run_p4_p5.sh --dry                # P4 예산 · P5 접두사 덤프
bash code/run_jobs_v4.sh train --dry        # 구인구직 v4
```

의존성은 [`requirements.txt`](requirements.txt). `.pt`·덤프 pkl 은 torch 가 있어야 열립니다.

---

## 참고 논문

- **TIGER** — Rajput et al., NeurIPS 2023 ([2305.05065](https://arxiv.org/abs/2305.05065)) — 재현 베이스라인
- **GRID Handbook** — Ju et al., Snap Research 2025 ([2507.22224](https://arxiv.org/abs/2507.22224)) — 코드베이스
- **MaskGR** ([2511.23021](https://arxiv.org/abs/2511.23021)) — 마스크 확산 (대조)
- **CRAB** — Fan et al., 2026 ([2604.05113](https://arxiv.org/abs/2604.05113)) — 코드북 재균형 (실패한 대조)
- **Semantic IDs for Joint Generative Search and Recommendation** — Penha et al., RecSys 2025 ([2508.10478](https://arxiv.org/abs/2508.10478)) — 5시드 평균·짝지은 t 프로토콜의 근거
