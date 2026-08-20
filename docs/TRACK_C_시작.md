# 트랙 C — 다양성 평가 트랙 (새 세션 시작 문서)

> 이 문서를 읽고 트랙 C 작업을 시작하세요.
> 작성: 2026-08-21 · 이전 세션(토큰화·임베딩·SID·TIGER)에서 인수인계

---

## 0. 30초 요약

지금 TIGER 평가는 **Recall@5/10, NDCG@5/10만** 봅니다. 즉 "정확한가"만 재고
**"신규·비인기 상품을 잘 추천하는가"는 못 재고 있습니다.** 그게 이 프로젝트의 핵심 주장인데도요.

**좋은 소식: 다양성 지표는 이미 전부 구현돼 있습니다** (`code/evaluate.py`).
고전 베이스라인(Random/MostPopular/ItemKNN/EASE_R)에는 이미 적용돼 있고,
**TIGER 출력만 이 하네스에 연결하면 됩니다.**

트랙 C의 일은 "지표를 새로 만드는 것"이 아니라 **"TIGER 추천 결과를 뽑아 기존 하네스에 넣는 것"** 입니다.

---

## 1. 프로젝트 맥락 (모르면 헤맵니다)

생성형 추천(TIGER/MaskGR)에서 **신규·비인기(콜드) 아이템 추천 문제**를
도달가능성 / 도달확률 / 편향 세 축으로 분해해 개선하고,
**정확도–롱테일 파레토 곡선**으로 검증하는 프로젝트입니다.

실험 구조는 4×2 그리드입니다.

| SID 방식 | TIGER (AR) | MaskGR (확산+다이얼) |
|---|---|---|
| 텍스트 SID | ① **완료** | ④ 첫 파레토 곡선 |
| CRAB SID | ② | ⑤ |
| G-SID (그래프) | ③ 진행 중 | ⑥ 최종 제안 |
| CRAB + G-SID | ⑧ | ⑦ |

> 팀 브리핑에 따르면 **파레토 곡선이 발표의 주인공**입니다.
> 마스크 확산 논문 3편과 CRAB이 **다양성 지표를 하나도 보고하지 않았기 때문에**,
> 우리가 그 곡선을 그리면 비교 대상이 없다는 것이 1순위 기여입니다.
> 그런데 지금 그 곡선을 그릴 데이터가 없습니다. 그게 트랙 C 가 푸는 문제입니다.

전체 배경은 `팀브리핑.md`, 지금까지 결과는 `TOKENIZE_EMBED_REPORT.md` 참고.

---

## 2. 이미 있는 것 — `code/evaluate.py`

### 인터페이스가 아주 단순합니다

```python
class Evaluator:
    def __init__(self, train_counts, n_items, tail_frac=0.5): ...
    def evaluate(self, preds, targets, K=(10, 20, 50), per_user=False):
        """preds: {user: [item,...] 내림차순}, targets: {user: set(item)}"""
```

**`preds` 와 `targets` 두 dict 만 만들면 끝입니다.**

### 이미 계산되는 지표

| 분류 | 지표 |
|---|---|
| 정확도 | `recall@k`, `hr@k`, `ndcg@k` |
| **다양성** | `aplt@k` (롱테일 비율), `coverage@k`, `exposure_gini@k`, `tail_exposure@k` |
| **콜드** | `bucket_recall@k` (head/mid/low/few-shot/zero-shot), `COLD_recall@k` |
| **파레토** | `pareto_table(points, acc="ndcg@20", div="tail_exposure@20")` — 지배 여부까지 계산 |

k = 10, 20, 50 을 기본으로 냅니다.

### 고전 베이스라인은 이미 이 지표로 측정돼 있습니다

`results/baseline_Beauty_loo.json` — Random / MostPopular / ItemKNN / EASE_R
전부 위 지표가 채워져 있습니다. **TIGER 결과만 같은 형식으로 만들면 바로 비교표가 나옵니다.**

---

## 3. 해야 할 일 — TIGER 추천 결과 뽑기

### ★ 핵심 난관 (이전 세션에서 확인함, 반드시 읽을 것)

GRID 의 `src.inference`(`predict_step`)로는 **평가용 추천을 뽑을 수 없습니다.**

```python
# src/models/modules/semantic_id/tiger_generation_model.py
def predict_step(self, batch: SequentialModelInputData):      # 라벨 인자가 없음
    generated_sids, _ = self.model_step(batch)

def eval_step(self, batch: Tuple[SequentialModelInputData, SequentialModuleLabelData], ...):
    ...                                                        # 라벨을 받아 정답을 가림
```

`predict_step` 은 **정답을 가리지 않습니다.** 전체 시퀀스(정답 포함)를 넣고 "그 다음"을
생성하는 실서비스용 경로입니다. 이걸로 평가하면 정답을 입력에 넣은 셈이 되어 부풀려집니다.

`labels` 오버라이드를 넣어도, `masking_token` 을 학습과 같은 -1 로 맞춰도
**예측이 md5 단위로 동일**했습니다 — 라벨이 아예 소비되지 않습니다.

### 그래서 선택지는 셋

| 방법 | 내용 | 난이도 |
|---|---|---|
| **A** | `eval_step` 에 예측 덤프 훅 추가 (callback 또는 코드 수정) | 중 · **권장** |
| B | 테스트 루프를 직접 작성 (마스킹 → generate → 저장) | 중상 |
| C | `predict_step` 에 마스킹을 넣도록 수정 | 중 (원본 동작 변경이라 주의) |

A 를 권합니다. `eval_step` 안에서 `generated_ids` 와 `label_data` 를 이미 다 갖고 있으므로,
거기서 user_id 와 함께 파일로 떨구면 됩니다.

### ★ top_k 를 반드시 늘려야 합니다

현재 `top_k_for_generation: 10` 입니다. 그런데 하네스는 **@50 까지** 계산하고,
파레토 곡선 기본값도 `ndcg@20` / `tail_exposure@20` 입니다.

**최소 50개는 생성해야 합니다.** 빔서치 폭이 늘어 느려지니 시간 여유를 두세요.

```
model.top_k_for_generation=50
```

### SID → item_id 역매핑

TIGER 는 SID 를 생성하므로 아이템 번호로 되돌려야 합니다.

```python
import torch
sid = torch.load("sid/L4/sid_tensor.pt").t()          # (12101, 5), 행=item_id
sid2item = {tuple(int(x) for x in sid[i]): i for i in range(sid.shape[0])}
```

- 생성된 SID 중 **실제 아이템에 매핑 안 되는 것이 0.03% 정도** 나옵니다(무효 SID).
  → 추천 목록에서 빼거나 별도 집계하세요. **버리는 개수를 반드시 로깅**할 것.
- 마지막 자리는 충돌 구분자입니다. 5자리 전부 맞아야 같은 아이템입니다.

---

## 4. 트랙 C 가 만들어야 할 것

### 4-1. 최소 목표 (여기까지만 해도 발표 가능)

**텍스트 SID × TIGER 의 다양성 지표**를 뽑아 고전 베이스라인과 같은 표에 넣기.

```
results/tiger_clip_L4_loo.json   ← baseline_Beauty_loo.json 과 같은 형식
```

이것만 있으면 *"TIGER 가 정확도는 EASE_R 보다 31% 높은데, 롱테일 노출은 어떤가?"* 에
답할 수 있습니다. **이 질문에 아직 아무도 답을 못 하고 있습니다.**

### 4-2. 본 목표 — 트랙 C 평가 프로토콜 설계

기존 두 트랙과 나란히 놓을 세 번째 트랙을 정의해야 합니다.

| 트랙 | 분할 | 무엇을 보나 | 상태 |
|---|---|---|---|
| A | Leave-One-Out | 표준 정확도 | ✅ 사용 중 |
| B | 절대시간 분할 | 시간에 따른 콜드스타트 | ⚠️ 프로토콜 미정 (아래 6절) |
| **C** | **?** | **다양성·롱테일 노출** | **← 설계 대상** |

설계할 때 답해야 할 것:

1. **어떤 분할 위에서 잴 것인가?** A(LOO) 위에서 다양성만 추가로 볼 것인가,
   아니면 별도 분할이 필요한가? (LOO 재사용이 가장 싸고, 기존 베이스라인과 바로 비교됨)
2. **롱테일 경계를 어디로?** 현재 `tail_frac=0.5` (하위 50%). 논문 관행과 맞는지 확인 필요
3. **어떤 지표를 대표로?** 파레토 곡선의 두 축을 무엇으로 할지
   (기본값 `ndcg@20` × `tail_exposure@20`, 하지만 `coverage` 나 `gini` 도 후보)
4. **다이얼이 없는 AR 모델에서 곡선을 어떻게 그릴 것인가?** ← **가장 중요**

### 4-3. ★ 4번 항목이 진짜 어려운 지점입니다

파레토 곡선은 "정확도를 조금 포기하면 다양성을 얼마나 얻는가"의 지도입니다.
곡선을 그리려면 **조절 가능한 손잡이(다이얼)** 가 있어야 하는데,
TIGER 같은 AR 모델은 빔서치라 다이얼이 없습니다. 점 하나만 찍힙니다.

가능한 접근:

- **생성 온도(temperature) / top-p 샘플링** — 빔서치 대신 샘플링으로 바꾸고 온도를 돌린다
- **인기도 페널티** — 생성 확률에서 아이템 빈도를 빼는 방식(re-ranking)
- **MaskGR 쪽에서만 곡선을 그리고 TIGER 는 점 하나로 둔다** ← 팀브리핑의 원래 설계

팀브리핑에는 *"오른쪽 사서(MaskGR)에 다이얼을 붙이는 건 50~100줄"* 이라고 돼 있습니다.
**즉 곡선의 주무대는 ④번 칸(MaskGR)이고, 트랙 C 는 그 평가 인프라를 미리 깔아두는 일**로
보는 게 맞을 수 있습니다. 팀과 범위를 먼저 합의하세요.

---

## 5. 필요한 파일 위치

```
code/evaluate.py                      평가 하네스 (다양성 지표 전부 구현됨)
results/baseline_Beauty_loo.json      고전 베이스라인 결과 (형식 참고용)
Beauty_split_A.pkl                    LOO 분할 (uid/iid/item_text/loo_train/val/test)
sid/L4/sid_tensor.pt                  텍스트 SID (5, 12101) — 전치 필요
embeddings/beauty_A/...tensor.pt      임베딩 12,101 × 2048
code/server/tiger_beauty.sh           TIGER 학습 스크립트
tiger_runs/<TAG>/metrics.csv          학습별 지표
docs/TIGER_BASELINE_REPORT.md         TIGER 결과·함정 상세
```

체크포인트는 서버에만 있습니다:
`/data1/dsl05/recsys/tiger_out/<TAG>/checkpoints/` (각 160MB)

**현재 최고 모델**: `clip_L4` (Recall@5 0.0444 / Recall@10 0.0660)
학습 레시피는 warmup 1,500 + cosine + clip 1.0, 30,000스텝.

---

## 6. 알고 있어야 할 상태

### Split B (트랙 B)는 아직 못 씁니다

콜드스타트 실험용인데 train 이 전체의 0.04~2.62% 뿐이라 학습이 불가능합니다.
버그가 아니라 논문 프로토콜(균등 달력 5등분)이 의도대로 구현된 결과입니다.
변경 요청서가 나가 있습니다 → `REQUEST_전처리_splitB.md`

**트랙 C 를 LOO 위에서 설계하면 이 블로커를 피할 수 있습니다.**

### 검증은 반드시 전수로

이전 세션에서 검증에 3,200명 표본을 썼다가 **성능이 3.2배 부풀려진 신호**를 읽었습니다.
전수 검증 비용은 전체의 6% 뿐입니다. `LIMIT_VAL=1.0` 을 쓰세요.

### 서버 함정

- 공유 저장소: 마스터는 `/data1/dsl05`, **계산 노드는 `/mnt/data1/dsl05`**
- `conda activate` 가 계산 노드에서 깨짐 → python 절대경로 호출
- `cryptography==42.0.8` 고정 (glibc 버전 차이)
- 자세한 건 `SERVER_공동사용.md`, `GPU_서버_사용가이드.md`

### GPU

- 계정 전체 **동시 2장** (누적 총량 제한은 없음)
- 팀원과 공유하는 단일 계정이므로 작업 이름에 본인 이름 넣기

---

## 7. 첫 세션에서 하면 좋을 순서

1. `code/evaluate.py` 와 `results/baseline_Beauty_loo.json` 을 읽고 형식 파악
2. `eval_step` 에서 예측을 덤프하는 방법 설계 (3절 A안)
3. `top_k_for_generation=50` 으로 `clip_L4` 체크포인트를 재평가
4. `results/tiger_clip_L4_loo.json` 생성 → 고전 베이스라인과 한 표에 놓기
5. 그 결과를 보고 트랙 C 프로토콜을 설계 (4-2절)

1~4 만 해도 **"TIGER 의 롱테일 노출은 얼마인가"** 라는, 지금 아무도 모르는 답이 나옵니다.
