# 그래프 기반 SID 담당자에게 — 인수인계

작성 2026-08-19 · 담당: 토큰화·임베딩·SID 파트

지금까지 **텍스트만 쓰는 baseline SID**(TIGER 방식)를 끝까지 돌려놨습니다.
그래프를 넣은 방식을 설계하실 때 필요한 입력물, 비교 기준, 그리고 **이미 밟은 지뢰**를 정리합니다.

---

## 1. 바로 쓸 수 있는 입력물

### 1-1. 아이템 임베딩 (당신 방법의 출발점이 될 것)

```
embeddings/beauty_A/merged_predictions_tensor.pt     # 12,101 x 2048 float32 (95MB)
```

```python
import torch
emb = torch.load("embeddings/beauty_A/merged_predictions_tensor.pt", map_location="cpu")
# emb.shape == (12101, 2048)
# ★ 행 인덱스 = item_id (0~12100). 별도 매핑 불필요.
```

| 항목 | 값 |
|---|---|
| 모델 | `google/flan-t5-xl` **인코더만** (1.22B 파라미터) |
| 풀링 | 토큰 평균(mean pooling), `attention_mask` 반영 |
| 입력 텍스트 | `Beauty_split_A.pkl` 의 `item_text` (제목 + 브랜드 + 설명 + 카테고리) |
| 토크나이저 | `max_length=128`, `truncation=True`, `padding=max_length` |
| 정밀도 | fp32 |
| 검증 | id 0~12,100 전부 존재 · 0벡터 없음 · 재실행 시 비트 단위 동일 |

> `item_id` 체계는 `Beauty_split_A.pkl` / `Beauty_split_B.pkl` / `Beauty_related_separate.pkl`
> 세 파일이 **전부 동일**합니다. ASIN 매핑은 `csv_export/dataset/A_items.csv` 참고.

### 1-2. 그래프 데이터

```
csv_export/dataset/relations_edges_long.csv     # src_item_id, dst_item_id, relation
csv_export/dataset/relations_summary.csv
```

원본은 `Beauty_related_separate.pkl` 의 `item_graph_detailed`:
`{src_item_id: {dst_item_id: {'also_bought', 'also_viewed', ...}}}` 형태입니다.

| 관계 | 간선 수 | 비중 |
|---|---|---|
| also_bought | 239,345 | 59.29% |
| also_viewed | 155,350 | 38.48% |
| bought_together | 9,021 | 2.23% |
| **관계 라벨 합계** | **403,716** | — |
| **중복 제거한 고유 방향 간선** | **301,103** | — |

> ⚠️ **한 간선이 여러 관계를 동시에 가질 수 있습니다.** 그래서 라벨 합계(403,716)와
> 고유 간선 수(301,103)가 다릅니다. 논문/발표에 숫자를 쓸 때 어느 쪽인지 명시하세요.
> 멀티 엣지 그래프로 다루려면 relation 별로 분리하고, 단순 그래프면 dedup 하세요.

또 하나 중요한 점: **그래프에 등장하는 아이템은 11,903개**로, 전체 12,101개보다 198개 적습니다.
즉 **고립 노드가 198개** 있습니다. 이 아이템들에 대해 그래프 신호가 아예 없으므로,
설계 시 fallback(텍스트 임베딩만 사용 등)이 필요합니다.

### 1-3. Baseline SID (당신 방법과 비교할 대상)

```
sid/L3/sid_tensor.pt     # (4, 12101)  ← 3단계
sid/L4/sid_tensor.pt     # (5, 12101)  ← 4단계
csv_export/sid/*.csv     # 사람이 읽을 수 있는 형태
```

```python
t = torch.load("sid/L4/sid_tensor.pt", map_location="cpu")
sid = t.T                    # ★ 반드시 전치! (레벨+1, 아이템) 으로 저장돼 있음
# sid.shape == (12101, 5), 행 인덱스 = item_id
# sid[i, :4] = 코드북 코드 4자리 (각 0~255)
# sid[i, 4]  = 충돌 구분자 (같은 코드를 공유하는 아이템들의 0부터 시작하는 일련번호)
```

---

## 2. Baseline 성능 — 당신 방법이 이겨야 할 숫자

| 항목 | 3단계 × 256 | 4단계 × 256 |
|---|---|---|
| 코드만으로 유일한 아이템 비율 | 87.97% | **93.69%** |
| 고유 코드 조합 수 | 10,645 | 11,338 |
| 충돌 그룹에 속한 아이템 | 2,347 (19.40%) | 1,248 (10.31%) |
| **구분자(마지막 자리)가 필요한 아이템** | **1,456 (12.03%)** | **763 (6.31%)** |
| 마지막 자리 최댓값 | 32 | 17 |
| 최대 충돌 그룹 크기 | 33 | 18 |
| 레벨별 사용된 코드 수 | 256/256/256 | 256/256/256/256 |
| 학습 MSE (epoch) | 0.000515 | 0.000511 |

- 모든 레벨에서 **256개 코드를 전부 사용**했습니다 (죽은 코드 없음).
- 재현: `code/server/sid_beauty.sh` (sbatch 한 번으로 3·4단계 모두 생성, 약 7분)
- 검증 스크립트는 `code/export_sid_csv.py` 가 같은 지표를 뽑아 줍니다.

**비교할 때 같은 지표를 쓰세요.** 특히 "충돌 비율"보다 **"구분자가 필요한 아이템 비율"과
"마지막 자리 최댓값"**이 실질적으로 중요합니다 (아래 3-2 참고).

---

## 3. ★ 우리가 측정한 것 — 그래프 방식의 근거가 될 내용

여기가 당신 파트에 가장 직접적으로 관련된 부분입니다.

### 3-1. 충돌의 원인은 "코드북 부족"이 아니라 "임베딩이 거의 같아서"

| 확인 항목 | 결과 |
|---|---|
| 임베딩이 비트 단위로 동일한 아이템 | 8개 (0.07%) |
| `item_text` 가 완전히 동일한 아이템 | 9개 (0.07%) |
| 충돌 그룹 내부 최대 코사인 유사도 (중앙값) | **0.974** |
| 내부 최대 코사인 ≥ 0.99 인 그룹 (3단계) | 249 / 891 (27.9%) |

충돌 그룹을 열어보면 **같은 제품의 색상·향 변형**들입니다:

- 3단계 최대 그룹(코드 `36-78-170`, 33개): Harmony Gelish UV 젤 폴리시 색상 변형
- 4단계 최대 그룹(코드 `37-177-177-18`, 18개): REVLON Colorburst Lip Butter 색상 변형
  (Creamsicle / Sugar Plum / Fig Jam / Candy Apple / Sweet Tart ...)

설명문이 통째로 같고 색상명 하나만 다르기 때문에, flan-t5-xl 평균 풀링으로는 구분이 안 됩니다.

> **그래프가 기여할 수 있는 지점이 여기입니다.** 이 변형들은 텍스트로는 거의 동일하지만
> also_bought / also_viewed 패턴은 서로 다를 수 있습니다. 반대로 "정말 같이 팔리는 세트"라면
> 그래프도 이들을 묶을 것이고, 그럼 충돌을 줄이는 게 아니라 **묶는 게 맞다**는 근거가 됩니다.
> 어느 쪽인지는 실제로 재보셔야 합니다.

### 3-2. 임베딩 공간이 심하게 쏠려 있습니다 (anisotropy)

| 측정 | 값 |
|---|---|
| 무작위 두 아이템의 코사인 유사도 평균 | **0.698** (표준편차 0.083) |
| 1레벨 양자화 후 남은 잔차 에너지 | 16.2% |
| 3레벨 후 | 12.8% |
| 6레벨 후 | 10.4% |

**1레벨에서 에너지의 83.8%가 제거되고, 그 뒤로는 거의 안 줄어듭니다.** 잔차가 잘게 쪼개지지
않는다는 뜻이고, 이래서 층을 쌓아도 수익이 급격히 체감합니다 (아래 3-3).

> 그래프 신호를 섞으면 임베딩이 더 고르게 퍼질 가능성이 있습니다. 그렇다면
> **"충돌 감소"보다 "잔차 에너지 감소 곡선"과 "코사인 유사도 분포"가 더 설득력 있는 지표**입니다.
> 이 두 수치는 위 표와 직접 비교할 수 있게 만들어 뒀습니다.

### 3-3. 층을 늘리는 것만으로는 한계가 있습니다 (로컬 시뮬레이션)

| 설정 | 코드만 유일 | 구분자 필요 | 아이템당 토큰 | 20개 히스토리 길이 |
|---|---|---|---|---|
| 3단계 × 256 | 84.9% | 15.1% | 4 | 80 토큰 |
| 4단계 × 256 | 93.2% | 6.8% | 5 | 100 토큰 |
| 5단계 × 256 | 95.6% | 4.4% | 6 | 120 토큰 |
| 6단계 × 256 | 96.5% | 3.5% | 7 | 140 토큰 |
| 3단계 × 1024 | 94.1% | 5.9% | **4** | **80 토큰** |

> 시뮬레이션은 GRID 실제 구현(kmeans++ 초기화 등)과 달라 절대값이 조금 낮게 나옵니다
> (실제 3단계는 87.97%). **경향 비교용**으로만 보세요.

3→4단계는 +8.3%p지만 4→5는 +2.4%p, 5→6은 +0.9%p로 수익이 급감하고, 6단계까지 가도
5.8%는 여전히 충돌합니다. 반면 시퀀스 길이는 계속 늘어납니다(어텐션 O(n²)).
**즉 "층을 더 쌓는다"는 방향은 막다른 길에 가깝고, 임베딩 자체를 개선하는 쪽이 지렛대입니다.**

---

## 4. 파이프라인에 끼워 넣는 법 — 가장 쉬운 경로

당신 방법이 **(12101 × D) 텐서 하나**를 만들어 내기만 하면, 그 뒤는 기존 파이프라인을
그대로 재사용할 수 있습니다.

```bash
# 1) 당신의 그래프 반영 임베딩을 저장 (행 인덱스 = item_id)
torch.save(my_graph_embedding, "my_embedding.pt")     # shape (12101, D)

# 2) sid_beauty.sh 에서 EMB / DIM 두 줄만 바꾸면 끝
#    EMB=.../my_embedding.pt
#    DIM=D
sbatch sid_beauty.sh
```

`code/server/sid_beauty.sh` 가 3단계·4단계 학습+부여를 한 번에 돌려주고,
`code/export_sid_csv.py` 가 위 2절과 **똑같은 지표**로 CSV를 뽑아 줍니다.
이렇게 하면 baseline과 완전히 동일한 조건에서 비교됩니다.

만약 SID 생성 알고리즘 자체를 바꾸실 거라면(예: 그래프 제약이 들어간 클러스터링),
`GRID/src/modules/clustering/residual_quantization.py` 와
`GRID/src/models/modules/clustering/mini_batch_kmeans.py` 를 보세요.
`configs/experiment/rqvae_train_flat.yaml`, `rvq_train_flat.yaml` 도 준비돼 있습니다.

---

## 5. ★ 이미 밟은 지뢰 — 같은 데서 시간 쓰지 마세요

### 5-1. GRID 코드 버그 2개 (레포에 패치 반영 완료)

| 파일 | 문제 | 우리 수정 |
|---|---|---|
| `GRID/src/utils/inference_utils.py` | `on_predict_end` 가 `if trainer.global_rank != None:` 로 검사 → rank 0에서도 **항상 참**이라, 단일 GPU 실행이면 프로세스 그룹 없이 `torch.distributed.barrier()` 를 불러 죽음. **추론은 다 끝나고 병합 직전에** 터져서 `.pt` 가 안 만들어짐 | `torch.distributed.is_available() and is_initialized()` 로 3곳 수정 |
| `GRID/src/utils/tensor_utils.py` | `deduplicate_rows_in_tensor` 가 `arange(1, N+1)` → 충돌 그룹이 1..N, 비충돌만 0 (docstring 과도 불일치) | `arange(0, N)` 로 수정. **마지막 자리 = 0부터 시작하는 일련번호** (팀 합의 규칙) |

> 이 저장소의 `GRID/` 스냅샷에는 **두 패치가 이미 반영**돼 있습니다. 업스트림을 새로 클론하면
> 다시 겪게 되니 주의하세요.

### 5-2. 설정 함정

1. **`num_workers=0` 필수** — `items/` TFRecord가 파일 1개라, 워커 2개 이상이면 하나가 빈
   이터레이터를 받아 `RuntimeError: generator raised StopIteration` 으로 죽습니다.
   `num_workers=0` + `timeout=0` + `persistent_workers=false` 를 **세트로** 주세요.
2. **레벨당 학습 스텝** — GRID는 `steps_per_layer = trainer.max_steps // n_layers` 로 나눠 씁니다.
   `max_steps` 를 기본값 30으로 두면 3단계는 레벨당 10스텝, 4단계는 7스텝이 되어 **조건이 달라집니다.**
   우리는 `max_steps = 30 × 레벨수` 로 줘서 양쪽 다 레벨당 30스텝으로 맞췄습니다.
   당신 방법과 비교하실 때도 이걸 맞추세요.
3. **`.project-root`** — GRID 루트에 빈 파일이 없으면 실행 자체가 안 됩니다.
4. **TFRecord는 GZIP 압축 필수**, 폴더명은 `training` / `evaluation` / `testing`.

### 5-3. 서버(학과 GPU) 관련

- 공유 저장소 마운트 경로가 **마스터 노드는 `/data1/<계정>`, 계산 노드는 `/mnt/data1/<계정>`** 로
  다릅니다. sbatch 스크립트는 반드시 `/mnt/data1` 기준으로 쓰세요. `#SBATCH --output` 을 틀리게 쓰면
  **로그 파일조차 없이 즉시 FAILED** 납니다.
- 마스터는 Ubuntu 22.04(glibc 2.35), 계산 노드는 20.04(glibc 2.31)입니다. pip가 마스터 기준으로
  최신 휠을 받아서 계산 노드에서 `GLIBC_2.33 not found` 가 납니다 → `cryptography==42.0.8` 고정.
- conda가 `/data1` prefix로 설치돼 있어 계산 노드에서 `conda activate` 가 깨집니다.
  **python 절대경로로 직접 호출**하세요.
- 접속 정보·계정은 팀 내부 문서 `GPU_서버_사용가이드.md` 에 있습니다
  (**계정 정보 때문에 깃허브에는 안 올립니다** — 팀 내부 채널로 받으세요).
- 참고 성능: RTX 6000 Ada 48GB 1장 기준 임베딩 추출 3분 50초, SID 3·4단계 합쳐 6분 59초.
- ⚠️ **`dsl05` 는 팀 공용 계정입니다.** SID 를 만들 때 `TAG` 를 지정하지 않으면 baseline 과
  충돌합니다. 규칙은 [`SERVER_공동사용.md`](SERVER_공동사용.md) 참고:
  ```bash
  sbatch --export=ALL,TAG=gsid_본인이름,EMB=<내 임베딩.pt>,DIM=<D> \n         --job-name=sid_본인이름 ~/recsys/sid_beauty.sh
  ```

---

## 6. 알고 계셔야 할 데이터 이슈 2개 (전처리 담당자 확인 대기 중)

1. **`Beauty_split_B.pkl` 의 시간 분할이 설계와 다릅니다.** train이 전체의 0.04~2.62%뿐이고
   W3는 아이템 12,101개 중 12,032개가 `unseen` 입니다. **콜드스타트 실험에 아직 못 씁니다.**
   그래프 SID의 콜드스타트 효과를 보시려면 이게 고쳐져야 합니다.
2. **`item_text` 가 실행마다 달라지는 재현성 버그.** `flatten_categories()` 가 set을 정렬 없이
   join해서 카테고리 단어 순서가 매번 바뀝니다 (Split A와 B가 11,909개 아이템에서 다름).
   **이번 임베딩은 Split A 텍스트 기준**입니다. 텍스트를 다시 만들어 임베딩을 새로 뽑으면
   결과가 달라지니, 비교하실 거면 반드시 `embeddings/beauty_A/merged_predictions_tensor.pt` 를
   그대로 쓰세요.

---

## 7. 질문 있으면

- 전체 경위·수치는 `docs/TOKENIZE_EMBED_SID_REPORT.md`
- CSV 파일 설명은 `csv_export/README.md`
- 재현 스크립트는 `code/server/`
