# CSV 내보내기 (2026-08-19)

전처리 pkl 3종과 flan-t5-xl 임베딩을 **엑셀에서 바로 열 수 있는 CSV**로 변환한 것입니다.
발표자료용 표·그래프를 여기서 바로 만들 수 있게 구성했습니다.

- 인코딩은 전부 **UTF-8 with BOM** — 엑셀에서 더블클릭해도 한글·특수문자가 안 깨집니다.
- 생성 스크립트: `code/export_csv.py` (다시 만들려면 이것만 실행)
- 파일 목록·크기는 `00_파일목록.csv` 에도 표로 정리돼 있습니다.

---

## 발표자료에 바로 쓸 만한 것 (★)

| 파일 | 쓸모 |
|---|---|
| `dataset/dataset_summary.csv` | **데이터셋 규모 한 장 요약** — 유저/아이템/상호작용 수 |
| `dataset/B_windows_summary.csv` | **Split B 분할 이상을 보여주는 표** (아래 주의 참고) |
| `dataset/relations_summary.csv` | 관계 3종(also_bought/also_viewed/bought_together) 구성비 |
| `embeddings/item_nearest_neighbors_top5.csv` | **임베딩이 잘 뽑혔다는 근거** — 최근접 이웃 예시표 |
| `embeddings/item_embeddings_pca2d.csv` | **산점도용 2D 좌표** — 엑셀 분산형 차트로 바로 그림 |
| `embeddings/pca_explained_variance.csv` | 몇 차원이면 충분한지 근거 (상위 50개로 전체 분산의 67.2% 설명) |

---

## 1. 데이터셋 (`dataset/`)

| 파일 | 행 수 | 내용 |
|---|---|---|
| `A_items.csv` | 12,101 | 아이템 마스터. `item_id`(0~12100) · ASIN · 판매순위 · **GRID에 넣은 `item_text`** |
| `A_users.csv` | 22,363 | 유저별 train 시퀀스(공백 구분 문자열) + val/test 정답 아이템 |
| `A_interactions_long.csv` | 183,348 | Split A 상호작용 롱포맷 (`user_id, position, item_id, split`). 피벗테이블용 |
| `B_windows_summary.csv` | 3 | W3/W4/W5 윈도우별 규모·콜드등급 분포 |
| `B_train_sequences_long.csv` | 5,817 | Split B 윈도우별 train 시퀀스 |
| `B_targets_long.csv` | 224,037 | Split B 윈도우별 val/test 정답 |
| `B_cold_tiers.csv` | 36,303 | 윈도우 × 아이템별 콜드 등급과 등장 횟수 |
| `relations_edges_long.csv` | 403,716 | 관계 간선 (`src, dst, relation`) |
| `relations_summary.csv` | 4 | 관계 종류별 간선 수 |
| `dataset_summary.csv` | 15 | 전체 통계 요약 |

`item_id` 는 세 pkl과 임베딩에서 **전부 동일한 번호 체계**입니다. `A_items.csv` 의 행 순서가
곧 임베딩 파일의 행 순서(= `item_id`)이므로 VLOOKUP 없이 그대로 붙일 수 있습니다.

### ⚠️ Split B 를 발표에 쓸 때 주의

`B_windows_summary.csv` 를 보면 train 비중이 윈도우 안에서도 1.4~2.6%, 전체 대비로는
0.04~2.62% 밖에 안 됩니다.

이건 **버그가 아니라 프로토콜 특성**입니다. `preprocess.py` 가 전체 기간(2002-06-12 ~
2014-07-23)을 **균등한 달력 기간으로 5등분**하는 "Can GR Reach Cold Items?" 논문 방식을
쓰는데, 리뷰가 2013년에 몰려 있어서(2002년은 4건) 초기 윈도우가 텅 비게 됩니다.
(🔄 처음엔 "분위수 계산 버그"로 적었으나 정정합니다 — 분위수 계산 자체가 없습니다.)

다만 W3는 아이템 12,101개 중 12,032개가 `unseen` 이라 학습 자체가 불가능하므로,
**프로토콜을 그대로 쓸지 결정되기 전까지는 콜드스타트 공식 결과로 쓰지 마세요.**
(자세한 내용은 `../TOKENIZE_EMBED_REPORT.md` 1절)

또 `relations_edges_long.csv` 의 행 수(403,716)는 **관계 라벨 기준**입니다. 한 간선이
also_bought와 also_viewed를 동시에 가질 수 있어서, 중복을 없앤 고유 방향 간선은 301,103개입니다.
발표에 숫자를 쓸 때 어느 쪽인지 명시하세요.

## 2. 임베딩 (`embeddings/`)

원본은 `../embeddings/beauty_A/merged_predictions_tensor.pt` (12,101 × 2048, float32).

| 파일 | 크기 | 내용 |
|---|---|---|
| `item_embeddings_preview.csv` | 0.1MB | 앞 300개 아이템 × 앞 16차원 + 상품명. **가볍게 눈으로 확인용** |
| `item_embedding_stats.csv` | 0.8MB | 아이템별 노름·평균·표준편차·최소/최대 |
| `item_embeddings_pca2d.csv` | 1.6MB | PCA 2차원 좌표 (산점도용) |
| `item_embeddings_pca50.csv` | 5.3MB | PCA 50차원 축약본 |
| `pca_explained_variance.csv` | - | 주성분별 설명 분산 |
| `item_nearest_neighbors_top5.csv` | 13.6MB | 아이템별 코사인 유사도 top-5 이웃 |
| `item_embeddings_full_2048d.csv` | **235MB** | 원본 전체 12,101 × 2048 |

> `item_embeddings_full_2048d.csv` 는 235MB · 2,050열입니다. 엑셀에서 열리기는 하지만
> (열 한도 16,384 이내) 매우 느립니다. 실제 분석은 pandas나 R로 읽거나,
> 원본 `.pt` 를 그대로 쓰는 걸 권합니다. 눈으로 볼 목적이면 `preview` 나 `pca50` 을 쓰세요.
> 값이 원본 `.pt` 와 일치하는지는 검증했습니다 (dim_0 / dim_2047 대조).

### 임베딩 품질 근거

`item_nearest_neighbors_top5.csv` 를 열어 아무 아이템이나 골라보면 같은 브랜드·같은 카테고리
상품이 상위에 옵니다. 예시:

| 기준 상품 | 1위 이웃 | 코사인 |
|---|---|---|
| Biosilk Silk Therapy Serum | Biosilk Silk Therapy Hair Treatment | 0.939 |
| Conair 조명 화장거울 | Revlon 조명 회전거울 | 0.917 |
| Garnier Moisture Rescue 로션 | Garnier Moisture Rescue 젤크림 | 0.917 |
| Boots No7 페이스마스크 | Boots No7 인텐스 세럼 | 0.888 |

## 3. SID / Semantic ID (`sid/`)

임베딩을 RQ-KMeans 로 양자화해 만든 **아이템별 이산 토큰 튜플**입니다.
레벨당 코드북 크기 256, 3단계와 4단계 두 가지를 만들었습니다.
마지막 자리는 **같은 코드를 공유하는 아이템들의 0부터 시작하는 일련번호**(충돌 구분용)입니다.

```
item 0  "WAWO 15 Color Professionl Makeup Eyeshadow..."
   3단계 → 16-106-240-0
   4단계 → 199-23-73-216-0
```

| 파일 | 행 수 | 내용 |
|---|---|---|
| `sid_summary.csv` | 2 | **3단계 vs 4단계 비교 요약** (발표자료용) |
| `sid_L3_items.csv` | 12,101 | 아이템별 3단계 SID (`code_1~3`, `dedup_last`, `sid`, 충돌 여부, 상품명) |
| `sid_L4_items.csv` | 12,101 | 아이템별 4단계 SID |
| `sid_L3_collisions.csv` | 2,347 | 3단계에서 충돌한 아이템만, 그룹 크기 큰 순 |
| `sid_L4_collisions.csv` | 1,248 | 4단계에서 충돌한 아이템만 |
| `sid_L3_code_usage.csv` | 768 | 레벨별 코드(0~255)에 몇 개 아이템이 배정됐는지 |
| `sid_L4_code_usage.csv` | 1,024 | 위와 동일 (4단계) |

### 3단계 vs 4단계

| 항목 | 3단계 | 4단계 |
|---|---|---|
| 코드만으로 유일한 비율 | 87.97% | **93.69%** |
| 충돌 아이템 | 2,347 (19.40%) | 1,248 (10.31%) |
| 최대 충돌 그룹 | 33개 | 18개 |
| 아이템당 토큰 수 | 4 | 5 |
| 학습 MSE | 0.000515 | 0.000511 |

4단계가 충돌을 절반으로 줄이지만 재구성 오차는 거의 같고, 시퀀스가 아이템당 1토큰씩 길어집니다.
두 레벨 수 모두 **256개 코드를 전부 사용**했습니다 (죽은 코드 없음).

### 충돌이 "잘못된 것"은 아닙니다

충돌은 임베딩이 거의 같은 아이템들이 같은 코드를 받은 것이고, 실제로 열어보면
**같은 제품의 색상·향 변형**들입니다. 발표에서 이 예시를 쓰면 SID가 의미를 잘 잡았다는 근거가 됩니다.

- 3단계 최대 그룹 (`36-78-170`, 33개): Harmony Gelish UV 젤 폴리시 색상 변형들
- 4단계 최대 그룹 (`37-177-177-18`, 18개): REVLON Colorburst Lip Butter 색상 변형들
  (Creamsicle / Sugar Plum / Fig Jam / Candy Apple / Sweet Tart ...)

`sid_L*_collisions.csv` 를 `sid_prefix` 로 정렬해 보면 이런 그룹들이 그대로 보입니다.
