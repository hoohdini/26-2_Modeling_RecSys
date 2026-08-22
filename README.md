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
| **텍스트 주소** (베이스라인) | ① ✅ **완료** | ④ ⬜ 다음 |
| **G-SID** (그래프 증강) | ③ 🔄 진행 중 | ⑥ ⬜ |
| **CRAB 주소** (코드북 재균형) | ② ⬜ | ⑤ ⬜ |
| **CRAB + G-SID** | ⑧ ⬜ | ⑦ ⬜ |

> **왜 좌우를 다 채워야 하나**: ⑦(결합 주소 + 확산)이 좋게 나왔을 때
> *"주소를 합쳐서 좋아진 건가, 확산 덕분인가"* 를 답하려면 **같은 주소를 TIGER 에도
> 시켜봐야(⑧)** 합니다. 한쪽만 돌리면 분해가 안 됩니다.

**베이스라인 정의**: GRID 원본 + gradient clipping (스텝 수만 조정). 그 위에 그래프 증강과
CRAB 을 얹습니다.

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
| G-SID 주소 (③) | 🔄 진행 중 |
| MaskGR 파이프라인 | ✅ 준비 완료 — 실행 대기 |
| CRAB 주소 | 🔄 파이프라인 정리 중 |
| Temporal 분할 (Split B) | ✅ **재전처리 완료** — W3/W4/W5 전부 학습 가능 |

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
| **다른 SID 로 실험한다** | [`docs/HANDOFF_트랙C_다양성평가.md`](docs/HANDOFF_트랙C_다양성평가.md) — 내 실행에 지표 붙이는 법 |
| **MaskGR 을 돌린다** | [`docs/HANDOFF_MaskGR.md`](docs/HANDOFF_MaskGR.md) |
| **그래프 SID 담당** | [`docs/HANDOFF_graph_sid.md`](docs/HANDOFF_graph_sid.md) |
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
sid/               Semantic ID — L3/L4(텍스트), gsid_a01/gsid_a03(그래프)
Tokenization/      그래프 SID 생성·비교 스크립트

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
#   sid[i, :4] = 코드 4자리(0~255) / sid[i, 4] = 충돌 구분자(0부터)
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
4. **`num_hierarchies` = SID 텐서의 행 수**(4단계 → 5), `vocab_size` = NH × 256.
5. **검증 표본을 셔플하지 않으면** 이력이 긴 유저만 뽑혀 검증 지표가 낙관적으로 나옵니다.
   전수 검증(`LIMIT_VAL=1.0`)이 26초면 끝나니 아낄 이유가 없습니다.

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
