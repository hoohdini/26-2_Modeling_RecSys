# 26-2 DSL Modeling — RecSys (생성형 추천)

Semantic ID 기반 생성형 추천에서 **신규·비인기(콜드) 아이템이 추천되지 않는 문제**를
다룹니다. 주소(Semantic ID)를 바꿔 보고, 모델을 바꿔 보고, 그 결과를 정확도만이 아니라
**다양성·롱테일 노출**까지 세 트랙으로 검증합니다.

---

## 30초 요약 — 이 프로젝트가 하는 일

```
Amazon Beauty 원본
        │
        ▼  ① 전처리                preprocessing/preprocess.py
   유저 22,363 · 상품 12,101 · 상호작용 198,502
        │
        ▼  ② 텍스트 → 벡터          GRID (flan-t5-xl 인코더)
   임베딩 12,101 × 2048
        │
        ▼  ③ 벡터 → 주소            GRID (RQ-KMeans, 코드북 256)
   SID  110-216-146-184  ← 상품 하나의 "주소". 앞자리가 큰 분류, 뒷자리가 세부
        │
        ├──────────────┬──────────────┬──────────────┐
     텍스트 주소     그래프 증강      CRAB 적용        결합        ← ④ 주소를 바꾼다
        └──────────────┴──────────────┴──────────────┘
        │
        ├───────────────────────────┐
     TIGER (자기회귀)          MaskGR (마스크 확산)                ← ⑤ 모델을 바꾼다
        └───────────────────────────┘
        │
        ▼  ⑥ 검증
   LOO 트랙  ·  Temporal 트랙  ·  다양성 트랙
```

**핵심 주장**: 마스크 확산 논문들과 CRAB은 **다양성 지표를 하나도 보고하지 않았습니다.**
우리 기여는 그 축을 재고, 추론 시점 파라미터 하나로 **정확도–롱테일 파레토 곡선**을 그린 것입니다.

---

## 실험 구조 — 주소 4종 × 모델 2종

| 주소 \ 모델 | TIGER (자기회귀) | MaskGR (마스크 확산) |
|---|---|---|
| **텍스트 주소** (베이스라인) | ① ✅ LOO·다양성 / 🔄 Temporal W5 | ④ ⬜ 학습 대기 (파이프라인 검증 완료) |
| **G-SID** (그래프 증강) | ③ ✅ LOO·다양성 / 🔄 Temporal W5 | ⑥ ⬜ |
| **CRAB 주소** (코드북 재균형) | ② ⬜ 주소 준비됨 | ⑤ ⬜ 주소 준비됨 |
| **CRAB + G-SID** | ⑧ ⬜ 주소 준비됨 | ⑦ ⬜ 주소 준비됨 |

> **왜 좌우를 다 채워야 하나**: ⑦(결합 주소 + 확산)이 좋게 나왔을 때
> *"주소를 합쳐서 좋아진 건가, 확산 덕분인가"* 를 답하려면 **같은 주소를 TIGER 에도
> 시켜봐야(⑧)** 합니다. 한쪽만 돌리면 분해가 안 됩니다.

**베이스라인 정의**: GRID 원본 + gradient clipping (스텝 수만 조정). 그 위에 그래프 증강과
CRAB 을 얹습니다.

### 주소 자산 — 지금 쓸 수 있는 SID 6종

②⑤⑦⑧ 은 **주소가 이미 다 만들어져 있고 학습만 남았습니다.**

| 행 | SID 이름 | 정체 | `WIDTH` |
|---|---|---|---|
| 텍스트 | `baseline/L4` (= `sid/L4`) | GRID 원본 + clipping | 256 |
| G-SID | `sid_T1_noTau_k0_b05` | **우리 튜닝식** (τ 없음 · κ 0 · β₀ 0.5) | 256 |
| G-SID | `gsid_a05_centered_nahye` | **고전 가중치 블렌딩** (α=0.5, 중심화) | 256 |
| CRAB | `crab_baseline` | 텍스트 + CRAB | **306** |
| CRAB+G-SID | `crab_T1_noTau_k0_b05` | 우리 튜닝식 + CRAB | **306** |
| CRAB+G-SID | `crab_gsid_a05_centered` | 고전 블렌딩 + CRAB | **306** |

> ⚠️ **G-SID 는 2종이 나란히 갑니다.** `sid_T1_...` 이 우리 방식이고
> `gsid_a05_centered_nahye` 가 비교용 고전 방식입니다. 이름만 보면 거꾸로 읽기 쉽습니다.
>
> ⚠️ **CRAB 주소는 `WIDTH=306` 을 넘겨야 합니다.** 토큰 분할로 코드 번호가 305 까지
> 늘어나서, 기본값 256 으로 돌리면 인덱스 범위를 벗어나 즉시 죽습니다. → 아래 함정 4번

---

## 검증 3트랙

| 트랙 | 무엇을 재나 | 분할 | 진입점 |
|---|---|---|---|
| **LOO** | 정확도 (Recall@5/@10, NDCG@10) | `Beauty_split_A.pkl` | `code/tiger_to_eval.py` |
| **Temporal** | 시간이 흐를 때 콜드 아이템을 잡나 | `Beauty_split_B.pkl` W3/W4/W5 | `code/tiger_to_eval.py --window W5` |
| **다양성** | 롱테일 노출·커버리지·도달가능성 | LOO/Temporal 덤프 재활용 | 같은 도구가 함께 출력 |

**컷오프는 전 지표 @10** 으로 통일했습니다(콜드 축만 @50). 다양성 대표축의 정식 명칭은
**APLT@10** (Average Percentage of Long Tail items)입니다.

> **다양성 트랙은 따로 돌리는 게 아닙니다.** `tiger_to_eval.py` 가 정확도와 다양성을
> 한 번에 뱉으므로, LOO·Temporal 덤프에서 자동으로 같이 나옵니다.

### 차이가 진짜인지 판정하는 자 — 노이즈 바닥

같은 레시피에 **시드만 바꾼** 두 실행의 간격입니다. **이보다 작은 차이는 주장할 수 없습니다.**

```
NDCG@10  0.0016        APLT@10  0.0176
```

> ⚠️ "유의하다(p<0.05)"와 "노이즈 바닥보다 크다"는 **다른 질문**입니다. 시드만 바꾼
> 실행도 "유의"하게 나옵니다. 반드시 이 자를 같이 대십시오. → `docs/TRACK_C_보고서.md` 6절

---

## 지금 상태

| 단계 | 상태 |
|---|---|
| 전처리 (Split A) | ✅ |
| 임베딩 · SID (텍스트, 3단계/4단계) | ✅ |
| TIGER 베이스라인 (①) | ✅ |
| 다양성 평가 하네스 + 파레토 곡선 인프라 | ✅ |
| G-SID 주소 | ✅ **확정 2종** — 우리 튜닝식 `T1` · 고전 블렌딩 `a05_centered` |
| CRAB 주소 | ✅ **3종 생성 완료** — 텍스트 / 튜닝식+CRAB / 고전+CRAB |
| MaskGR 파이프라인 | ✅ **서버 배포 + 환경 구축 + GPU 스모크 통과** |
| Temporal 분할 (Split B) | ✅ **재전처리 완료** — W3/W4/W5 전부 학습 가능 |
| Temporal 학습 (TIGER) | 🔄 **W5 진행 중** — 텍스트·`T1` 학습 완료(조기종료), `a05_centered` 대기 |

**Temporal W5 학습 실측** (2026-08-24, `partition1` RTX 6000 Ada · 30k 예산에서 조기종료)

| 조건 | 종료 step | 최저 val/loss | 소요 |
|---|---|---|---|
| `tg_text_W5` (텍스트) | 15,999 | 12.3521 @ 6,999 | 약 1시간 50분 |
| `tg_T1_W5` (우리 튜닝식) | 15,999 | 12.0995 @ 6,999 | 약 1시간 45분 |

> 두 조건 모두 **step 6,999 에서 검증 손실 최저**를 찍고 이후 개선이 없어 patience 8 로
> 조기 종료했습니다. 30k 를 다 돌 필요가 없다는 뜻이라, 남은 윈도우 예산을 그만큼 줄여
> 잡아도 됩니다.

> ⚠️ **Temporal 은 지금 시드 1회입니다.** 노이즈 바닥(아래)보다 작은 차이는 주장할 수
> 없는데, LOO 실측에서 `COLD_recall@50` 의 시드 간 차이가 **0.0044** 로 두 G-SID 사이
> 격차(0.0026)보다 큽니다. 콜드 축 결론을 내려면 시드 2회가 필요합니다.

**TIGER 베이스라인 (①, 4단계 SID · 전수 22,363명)**

| 모델 | Recall@10 | NDCG@10 | APLT@10 |
|---|---|---|---|
| EASE_R (고전 최강) | 0.0509 | 0.0265 | 0.0567 |
| **TIGER `clip_L4`** | **0.0692** | **0.0384** | **0.0497** |
| TIGER 논문 (Beauty) | 약 0.0648 | 약 0.0384 | — |

> **정확도는 이기는데 롱테일 노출은 오히려 낮습니다** (0.0497 vs 0.0567).
> 우리가 풀겠다고 한 문제가 실제로 존재한다는 첫 수치입니다. → `docs/TRACK_C_보고서.md`

---

## 어디서부터 읽나 — 역할별

| 나는… | 읽을 것 |
|---|---|
| **처음 온 사람** | 이 README → [`docs/TEAM_RESULTS.md`](docs/TEAM_RESULTS.md) |
| **텍스트 SID 3트랙을 채운다** | [`docs/RUNBOOK_텍스트SID_3트랙.md`](docs/RUNBOOK_텍스트SID_3트랙.md) — 남은 GPU 작업 7회 |
| **다른 SID 로 실험한다** | [`docs/HANDOFF_트랙C_다양성평가.md`](docs/HANDOFF_트랙C_다양성평가.md) — 내 실행에 지표 붙이는 법 |
| **MaskGR 을 돌린다** | [`docs/HANDOFF_MaskGR.md`](docs/HANDOFF_MaskGR.md) |
| **그래프 SID 담당** | [`docs/HANDOFF_graph_sid.md`](docs/HANDOFF_graph_sid.md) |
| **CRAB 주소 담당** | [`crab/README.md`](crab/README.md) — 재구현 근거·설정·미해결 과제 |
| **전처리 담당** | [`docs/REQUEST_전처리_splitB.md`](docs/REQUEST_전처리_splitB.md) |
| **다양성·롱테일 결과가 궁금** | [`docs/TRACK_C_보고서.md`](docs/TRACK_C_보고서.md) · [비교표](docs/TRACK_C_비교표.md) |
| **재전처리로 뭐가 바뀌었나** | [`docs/SPLIT_재전처리_영향.md`](docs/SPLIT_재전처리_영향.md) — 실측 대조 |
| **TIGER 재현 경위** | [`docs/TIGER_BASELINE_REPORT.md`](docs/TIGER_BASELINE_REPORT.md) |
| **서버에 들어간다** | [`docs/SERVER_공동사용.md`](docs/SERVER_공동사용.md) · [`docs/GPU_서버_사용가이드.md`](docs/GPU_서버_사용가이드.md) 🔒 |
| **코드를 고친다** | [`code/README.md`](code/README.md) — 의존 관계와 실행 순서 |

---

## 저장소 구조

```
Data/              Amazon 2014 Beauty 원본 (meta, reviews 5-core)
preprocessing/     전처리 코드 + 설계 문서
Beauty_split_A.pkl LOO 분할 ★ 코드 기본값이라 옮기지 말 것
Beauty_split_B.pkl Temporal 분할 — W3/W4/W5 (누적 train)
Beauty_related_separate.pkl  관계종류 보존 그래프 (G-SID 가중치 실험용)
                   셋 다 preprocessing/repreprocess.py 가 만드는 것과 같은 파일

embeddings/        flan-t5-xl 아이템 임베딩 (12,101 × 2048)
sid/               Semantic ID
                     L3/ L4/                텍스트 (베이스라인)
                     gsid_*/                그래프 증강
                     crab_*/                CRAB 적용분 ★ WIDTH=306
Tokenization/      그래프 SID 생성·비교 스크립트
crab/              CRAB 코드북 재균형 (crab_sid.py + README) — 후처리라 GPU 불필요

GRID/              snap-research/GRID 스냅샷 (팀 패치 2건 반영) — 임베딩·SID·TIGER
MaskGR/            snap-research/MaskGR 스냅샷 — 마스크 확산

code/              평가 하네스·변환·분석 (→ code/README.md)
code/server/       서버 실행 스크립트 (sbatch)

results/           지표 JSON (→ results/README.md)
tiger_runs/        실행별 학습 로그 (metrics.csv)
csv_export/        엑셀에서 여는 CSV — 발표자료용 (→ csv_export/README.md)
docs/              리포트와 인수인계 문서
_archive/          지금은 안 쓰는 파일 (지운 게 아님 → _archive/README.md)
```

### 핵심 산출물 불러오기

```python
import torch
emb = torch.load("embeddings/beauty_A/merged_predictions_tensor.pt")  # (12101, 2048) 행=item_id
sid = torch.load("sid/L4/sid_tensor.pt").T                            # (12101, 5)    행=item_id
#   sid[i, :4] = 코드 4자리 / sid[i, 4] = 충돌 구분자(0부터)
#   코드 범위는 SID 마다 다릅니다 — 텍스트·G-SID 0~255, CRAB 주소 0~305
```

| SID 설정 | 코드만으로 유일 | 구분자 필요 | 아이템당 토큰 |
|---|---|---|---|
| 3단계 × 256 | 87.97% | 1,456 (12.03%) | 4 |
| **4단계 × 256** (채택) | **93.69%** | **763 (6.31%)** | 5 |

---

## 실행 — 빠른 길

환경과 의존성은 [`requirements.txt`](requirements.txt) 참고. 자주 쓰는 명령은 `Makefile` 에 있습니다.

```bash
make help              # 무엇을 할 수 있는지
make table             # 지금 결과로 비교표 다시 만들기 (GPU 불필요)
make baselines         # 고전 베이스라인 재계산
```

전체 흐름과 서버 명령은 각 핸드오프 문서에 있습니다.

---

## ⚠️ 반복해서 당한 함정 (읽고 시작하세요)

1. **`src.inference`(`predict_step`) 산출물로 평가하지 마십시오.** 정답을 가리지 않아
   Recall@10 이 0.0660 → 0.0961(**+46%**)로 부풀려집니다. 올바른 경로는
   `code/server/tiger_eval_dump.sh` 입니다. 관련 스크립트는 `_archive/` 로 옮겼습니다.
2. **`src.train train=False` 로 평가하지 마십시오.** `ckpt_path` 를 설정에서 읽지 않아
   **랜덤 가중치로 평가**하고 경고 한 줄만 남깁니다. `src/eval_dump.py` 를 쓰십시오.
3. **TFRecord 는 split 당 1파일**이라 `num_workers=0` + `timeout=0` +
   `persistent_workers=false` 를 **세트로** 줘야 합니다. 안 그러면 워커가 굶어 죽습니다.
4. **`num_hierarchies` = SID 텐서의 행 수**(4단계 → 5), `vocab_size` = NH × `WIDTH`.
   `WIDTH` 는 **SID 마다 다릅니다** — 텍스트·G-SID 는 256(VOCAB 1280), **CRAB 주소는
   306**(VOCAB 1530). CRAB 은 과인기 토큰을 쪼개면서 코드 번호를 305 까지 늘리므로,
   256 으로 돌리면 임베딩 테이블 범위를 벗어나 즉시 죽습니다.
5. **검증 표본을 셔플하지 않으면** 이력이 긴 유저만 뽑혀 검증 지표가 낙관적으로 나옵니다.
   전수 검증(`LIMIT_VAL=1.0`)이 26초면 끝나니 아낄 이유가 없습니다.
6. **MaskGR 에서는 `+trainer.…` 가 아니라 `++trainer.…` 를 쓰십시오.**
   `configs/experiment/discrete_diffusion_train.yaml` 이 `gradient_clip_val` 과
   `limit_val_batches` 를 **이미 정의**하고 있어서, GRID 에서 하던 대로 `+`(추가)를 주면
   *"Could not append to config"* 로 죽습니다. GRID 설정에는 이 키들이 없어 `+` 가 맞았습니다.
7. **`limit_val_batches` 에 분수를 주지 마십시오.** 데이터셋이 `IterableDataset` 라
   lightning 이 `1.0` 이거나 정수만 받습니다. `0.05` 같은 값은
   `MisconfigurationException` 입니다.
8. **MaskGR 은 실패해도 3번 재시도합니다**(`src/utils/restart_job.py`). 설정 오류처럼
   재시도해도 안 고쳐지는 실패에서 GPU 슬롯을 그만큼 더 잡아먹습니다. 새 설정은
   **CPU 스모크로 먼저** 걸러내는 편이 쌉니다 — `trainer=cpu` + `++trainer.precision=32`
   로 학습 루프 진입까지는 GPU 없이 검증됩니다. (`beam_search_generation` 이
   `device='cuda'` 하드코딩이라 검증 단계부터는 GPU 가 필요합니다.)

---

## 업스트림 스냅샷

`GRID/` 와 `MaskGR/` 는 업스트림을 복사해 넣은 것입니다(`.git` 제거).

| 폴더 | 업스트림 | 기준 커밋 |
|---|---|---|
| `GRID/` | https://github.com/snap-research/GRID | `2fe3475` (main) |
| `MaskGR/` | https://github.com/snap-research/MaskGR | `b2d44f2` (main) |

> `GRID/` 에는 팀 패치 2건이 반영돼 있습니다:
> 1. `src/utils/inference_utils.py` — 단일 GPU 에서 `torch.distributed.barrier()` 로
>    병합이 항상 실패하던 버그
> 2. `src/utils/tensor_utils.py` — SID 충돌 구분자를 0부터 시작하도록 변경
>
> **`MaskGR/` 은 아직 무패치입니다.** 신규 파일만 얹어 씁니다 — `docs/HANDOFF_MaskGR.md` 참고.

---

## 참고 논문

- **TIGER** — 재현 베이스라인 (SID 패러다임)
- **GRID Handbook** ([2507.22224](https://arxiv.org/abs/2507.22224)) — 코드베이스·설계 근거
- **MaskGR** ([2511.23021](https://arxiv.org/abs/2511.23021)) — 마스크 확산 생성형 추천
- **Can GR Reach Cold Items?** — 콜드스타트 진단·절대시간 분할 프로토콜
- **CRAB** ([2604.05113](https://arxiv.org/abs/2604.05113)) — 코드북 재균형
