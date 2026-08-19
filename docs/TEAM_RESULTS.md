# 팀 공유 — 토큰화 · 임베딩 · SID 결과 (2026-08-19)

담당 파트가 끝났습니다. **무엇이 나왔고, 어디에 있고, 각자 뭘 하면 되는지**만 정리했습니다.
자세한 경위는 `docs/TOKENIZE_EMBED_SID_REPORT.md`, 그래프 SID 담당자는 `docs/HANDOFF_graph_sid.md`.

---

## 1. 한 줄 요약

Amazon Beauty 12,101개 아이템에 대해 **flan-t5-xl 임베딩(2048차원)** 을 뽑고,
**RQ-KMeans로 Semantic ID를 3단계·4단계 두 가지** 만들었습니다. 전부 검증 통과했고
학과 GPU 서버에서 재현 가능합니다.

```
item 0  "WAWO 15 Color Professionl Makeup Eyeshadow..."
   3단계 SID → 16-106-240-0
   4단계 SID → 199-23-73-216-0
```

## 2. 결과물 위치

| 무엇 | 경로 | 크기 |
|---|---|---|
| 아이템 임베딩 | `embeddings/beauty_A/merged_predictions_tensor.pt` | 12,101 × 2048, 95MB |
| SID 3단계 | `sid/L3/sid_tensor.pt` | (4, 12101) |
| SID 4단계 | `sid/L4/sid_tensor.pt` | (5, 12101) |
| **엑셀에서 열리는 CSV 전부** | `csv_export/` | 84MB |
| 재현 스크립트 | `code/server/` | — |
| 상세 리포트 | `docs/TOKENIZE_EMBED_SID_REPORT.md` | — |

> `.pt` 는 `torch.load()` 로 읽습니다. **SID 텐서는 (레벨+1, 아이템) 형태라 `.T` 로 전치**해야
> 행 인덱스가 `item_id` 가 됩니다. 임베딩은 전치 없이 바로 행=item_id 입니다.
>
> 용량 때문에 뺀 것: 임베딩 전체 CSV(235MB), `merged_predictions.pkl`(214MB).
> 둘 다 `.pt` 와 내용이 같습니다. 필요하면 말씀하세요.

## 3. 숫자 (발표자료용)

### 임베딩

| 항목 | 값 |
|---|---|
| 모델 | google/flan-t5-xl 인코더 (1.22B 파라미터), 평균 풀링 |
| 차원 | 2048 |
| 아이템 | 12,101개 전부 (결측·0벡터 없음) |
| 소요 시간 | RTX 6000 Ada 1장으로 **3분 50초** (맥북 CPU 추정 6.5시간) |
| 재현성 | 재실행 결과가 **비트 단위로 동일** |

**품질 근거 (최근접 이웃)** — `csv_export/embeddings/item_nearest_neighbors_top5.csv`

| 기준 상품 | 1위 이웃 | 코사인 |
|---|---|---|
| Biosilk Silk Therapy Serum | Biosilk Silk Therapy Hair Treatment | 0.939 |
| Conair 조명 화장거울 | Revlon 조명 회전거울 | 0.917 |
| Garnier Moisture Rescue 로션 | Garnier Moisture Rescue 젤크림 | 0.917 |
| Boots No7 페이스마스크 | Boots No7 인텐스 세럼 | 0.888 |

### SID

| 항목 | 3단계 × 256 | 4단계 × 256 |
|---|---|---|
| 코드만으로 유일한 비율 | 87.97% | **93.69%** |
| 구분자(마지막 자리)가 필요한 아이템 | 1,456 (12.03%) | **763 (6.31%)** |
| 마지막 자리 최댓값 | 32 | 17 |
| 최대 충돌 그룹 | 33개 | 18개 |
| 아이템당 토큰 수 | 4 | 5 |
| 레벨별 사용된 코드 | 256/256/256 | 256/256/256/256 |
| 전체 SID 유일성 | 12,101/12,101 ✓ | 12,101/12,101 ✓ |

- 모든 레벨에서 **256개 코드를 전부 사용**했습니다 (죽은 코드 없음 = 코드북 활용 양호).
- 충돌 처리는 **마지막 자리에 0부터 시작하는 일련번호**를 붙이는 방식(TIGER 방식)입니다.

### 충돌은 고장이 아닙니다 (발표에서 물어볼 만한 부분)

충돌 그룹을 열어보면 **같은 제품의 색상 변형**입니다 — 오히려 의미를 잘 잡았다는 근거입니다.

- 4단계 최대 그룹(`37-177-177-18`, 18개): REVLON Colorburst Lip Butter
  Creamsicle / Sugar Plum / Fig Jam / Candy Apple / Sweet Tart ...

## 4. ★ 결정이 필요한 것 — 3단계냐 4단계냐

TIGER 학습에 들어가려면 하나를 골라야 합니다.

**추천: 4단계.** 이유는 마지막 자리 토큰이 **모델이 예측할 근거가 없는 정보**이기 때문입니다.
"Sweet Tart"가 4번인 건 순전히 item_id 순서 탓이라 모델은 사실상 0 이외를 못 맞춥니다.
그래서 구분자가 필요한 아이템이 적고 최댓값이 작을수록 유리합니다
(12.03% / 최댓값 32 → **6.31% / 최댓값 17**).

대신 시퀀스가 아이템당 1토큰 길어집니다 (20개 히스토리 기준 80 → 100 토큰).

> 5·6단계는 권하지 않습니다. 4→5는 +2.4%p, 5→6은 +0.9%p로 수익이 급감하는 반면
> 시퀀스 길이 비용은 계속 늘어납니다 (어텐션 O(n²)).
> 둘 다 만들어 놨으니 **TIGER 초기 실험에서 양쪽 다 돌려보고 정해도 됩니다.**

## 5. ★ 막혀 있는 것 — 전처리 담당자 확인 필요

### ① `Beauty_split_B.pkl` 시간 분할이 설계와 다릅니다 (우선순위 높음)

설계는 "80/85/90/95% 분위수 컷 → 약 8:2"인데 실측은 이렇습니다:

| 윈도우 | train | val | test | train 비중(전체 대비) |
|---|---|---|---|---|
| W3 | 75건 | 502 | 4,738 | 0.04% |
| W4 | 533건 | 4,738 | 20,872 | 0.27% |
| W5 | 5,209건 | 20,872 | **172,315** | 2.62% |

W3는 아이템 12,101개 중 **12,032개가 `unseen`** 이라 학습이 사실상 불가능합니다.
분위수를 리뷰 건수가 아니라 **고유 타임스탬프 기준으로 계산**했거나 부등호가 반대인 것으로 추정합니다.

**이대로는 콜드스타트 실험(공식 결과 B분할)에 쓸 수 없습니다.** 이게 우리 프로젝트의 핵심
주제라서 가장 먼저 풀려야 합니다. `csv_export/dataset/B_windows_summary.csv` 를 열면 바로 보입니다.

### ② `item_text` 가 실행마다 달라집니다 (재현성 버그)

`flatten_categories()` 가 set을 정렬 없이 join해서 카테고리 단어 순서가 매번 바뀝니다.
Split A와 B가 **11,909개 아이템에서 텍스트가 다릅니다** (단어 집합은 같고 순서만 다름).

수정 제안: `' '.join(sorted(flat))` 로 순서 고정.

> 이번 임베딩은 **Split A 텍스트 기준**입니다. 텍스트를 고쳐서 다시 뽑으면 임베딩도 SID도
> 달라지니, 고칠 거면 지금 고치고 재추출하는 게 낫습니다 (재추출 4분, SID 7분이라 부담 없음).

## 6. 각자 할 일

| 담당 | 할 일 |
|---|---|
| **전처리** | 위 ①②. 특히 Split B 분할 로직 확인 — 이게 콜드스타트 실험의 전제 |
| **그래프 SID** | `docs/HANDOFF_graph_sid.md` 참고. 임베딩 `.pt` 를 입력으로 쓰고, 결과는 같은 지표로 비교 |
| **TIGER 학습** | `sid/L3` 또는 `sid/L4` 를 입력으로. `GRID/configs/experiment/tiger_train_flat.yaml` |
| **발표자료** | `csv_export/` 의 요약 CSV들. 아래 7절 참고 |

## 7. 발표자료에 바로 쓸 CSV

| 파일 | 쓸모 |
|---|---|
| `csv_export/dataset/dataset_summary.csv` | 데이터셋 규모 한 장 요약 |
| `csv_export/sid/sid_summary.csv` | 3단계 vs 4단계 비교표 |
| `csv_export/sid/sid_L4_collisions.csv` | 충돌 처리 설명용 (립버터 18색이 그대로 보임) |
| `csv_export/embeddings/item_nearest_neighbors_top5.csv` | 임베딩 품질 근거 |
| `csv_export/embeddings/item_embeddings_pca2d.csv` | 산점도 (엑셀 분산형 차트로 바로) |
| `csv_export/dataset/relations_summary.csv` | 관계 3종 구성비 |
| `csv_export/dataset/B_windows_summary.csv` | 5절 ① 문제를 보여주는 표 |

인코딩은 전부 UTF-8 BOM이라 엑셀에서 더블클릭해도 안 깨집니다.

> 숫자 인용 시 주의: `relations_edges_long.csv` 의 403,716행은 **관계 라벨 기준**입니다.
> 한 간선이 also_bought와 also_viewed를 동시에 가질 수 있어, 중복 없앤 고유 방향 간선은
> **301,103개**입니다. 어느 쪽인지 명시하세요.

## 8. 인프라 정리 (누구나 쓸 수 있게 해뒀습니다)

학과 GPU 서버(dsl05)에 환경 구축이 끝나 있습니다.

> ⚠️ **`dsl05` 는 팀 전체가 같이 쓰는 단일 계정입니다.** 팀원이 접속하면 제가 만든 데이터에서
> 그대로 이어서 작업하게 되지만, **서로의 결과를 지울 수도 있습니다.**
> 서버에 들어오기 전에 반드시 [`docs/SERVER_공동사용.md`](SERVER_공동사용.md) 를 읽어주세요.

- conda 환경(python 3.10 + torch 2.6.0+cu124), flan-t5-xl 캐시 11GB, GRID 코드, TFRecord 전부 업로드
- `sbatch ~/recsys/embed_beauty.sh` (임베딩 4분) / `sbatch ~/recsys/sid_beauty.sh` (SID 7분)
- 사용 가능 GPU: `partition1` (hpc-stat1, **RTX 6000 Ada 48GB 5장**), `jobs` (hpc, 2장)
- 접속 방법·함정은 팀 내부 문서 `GPU_서버_사용가이드.md`
  (**계정 정보가 있어 깃허브에는 안 올립니다** — 팀 채널로 받으세요)

GRID 코드 버그 2개도 잡아서 이 레포의 `GRID/` 스냅샷에 반영해 뒀습니다
(단일 GPU에서 병합이 항상 실패하던 버그, 충돌 번호 0-base 규칙). 자세한 건 리포트 참고.
