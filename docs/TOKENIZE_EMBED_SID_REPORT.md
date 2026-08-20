# 토큰화·임베딩·SID 작업 리포트 (2026-08-19, 최종)

작업: 팀원 전처리 pkl 검증 → GRID 입력 TFRecord 생성(토큰화) → flan-t5-xl 임베딩 추출
→ **RQ-KMeans 로 아이템별 Semantic ID(SID) 생성**
상태: **완료.** 전부 학과 GPU 서버(dsl05, RTX 6000 Ada)에서 수행.

최종 산출물
- 임베딩: `embeddings/beauty_A/merged_predictions_tensor.pt` — **12,101 × 2048 (float32, 95MB)**
- SID: `sid/L3/sid_tensor.pt`, `sid/L4/sid_tensor.pt` — **3단계 / 4단계, 레벨당 코드북 256**

---

## 1. 전처리 데이터셋 검증

받은 파일 3개 (프로젝트 루트):

| 파일 | 내용 | 검증 결과 |
|---|---|---|
| `Beauty_split_A.pkl` | Leave-one-out 분할 | 유저 22,363 ✓ / 상품 12,101 ✓ / train seq 최대 20(max_seq_len=20 ✓) |
| `Beauty_split_B.pkl` | 절대시간 분할 (W3/W4/W5) | ⚠️ 아래 문제 ① |
| `Beauty_related_separate.pkl` | 관계 3종 분리 그래프 | 아이템 11,903 / 방향 간선 301,103 (also_bought 239,345 · also_viewed 155,350 · bought_together 9,021). 합집합이 split A/B의 `item_graph`와 정확히 일치 ✓ |

- 세 파일의 `iid` 매핑 동일 ✓
- Split A 상호작용 합계 183,348 (= 198,502 − max_seq_len=20 절단분 15,154. 의도된 동작으로 보임)

### ⚠️ 발견한 문제 — 전처리 담당자 확인 필요

**① Split B의 학습 데이터가 너무 적어 이대로는 실험 불가.**

> 🔄 **2026-08-19 정정.** 처음에 "분위수 계산이 잘못된 버그"로 보고했으나 **틀렸습니다.**
> `preprocess.py:148 split_chronological_windows()` 를 확인하니 **분위수 계산 자체가 없습니다.**
> 전체 기간을 **균등한 달력 기간으로 5등분**하는 "Can GR Reach Cold Items?" 논문 방식이고,
> 코드에 `초기 윈도우는 데이터가 희박할 수 있음` 이라는 주석까지 있습니다. **의도된 구현입니다.**

실제 컷 날짜 (전체 기간 2002-06-12 ~ 2014-07-23, 12.11년을 2.42년씩):

| 윈도우 | 기간 | 리뷰 수 | 비중 |
|---|---|---|---|
| W1 | 2002-06-12 ~ 2004-11-12 | 75 | 0.04% |
| W2 | 2004-11-12 ~ 2007-04-16 | 502 | 0.25% |
| W3 | 2007-04-16 ~ 2009-09-17 | 4,738 | 2.39% |
| W4 | 2009-09-17 ~ 2012-02-19 | 20,872 | 10.51% |
| W5 | 2012-02-19 ~ 2014-07-23 | 172,315 | 86.81% |

여기서 만든 3개 분할(직전 윈도우=val, 그 이전 누적=train):

| 라벨 | train | val | test |
|---|---|---|---|
| W3 | 75건 (전체의 0.04%) | 502 | 4,738 |
| W4 | 533건 (0.27%) | 4,738 | 20,872 |
| W5 | 5,209건 (2.62%) | 20,872 | **172,315 (87%)** |

리뷰가 2013년에 몰려 있어(2002년은 4건) 균등 시간 분할을 하면 쏠림이 그대로 반영됩니다.

**그래서 남는 쟁점은 "버그 수정"이 아니라 "프로토콜 선택"입니다.** W3는 train이 75건이라
학습 자체가 불가능하고, W5조차 train 5,209건 대 test 172,315건입니다. 논문 방식을 그대로
따를지, 아니면 데이터량 기준 컷으로 바꿀지 **전처리 담당자와 논의가 필요합니다.**
(참고: 리뷰 건수 분위수로 잘랐다면 컷이 2012-08-16 / 2013-04-05 / 2013-09-28 / 2014-03-07 로
전부 마지막 2년에 몰립니다.)

**② Split A와 B의 `item_text`가 11,909/12,101개 아이템에서 서로 다름.**
단어 집합은 동일하고 카테고리 단어 순서만 다름 → `flatten_categories()`가 set을 정렬 없이
join해서 실행마다 순서가 바뀌는 재현성 버그. 임베딩이 텍스트 순서에 민감하므로
`' '.join(sorted(flat))` 로 고정 권장. **이번 임베딩은 Split A의 텍스트를 기준으로 진행.**

## 2. GRID 입력 TFRecord 생성 (완료)

- 변환 스크립트: `code/to_grid_v2.py` (신규 — 팀원 pkl 스키마용 어댑터. 기존 `to_grid.py`는 구버전 pkl용이라 미수정)
- 입력: `Beauty_split_A.pkl` (LOO) / 출력: `grid_data/beauty_A`
  - `items/` 12,101건 (id int64, text bytes)
  - `training/` 22,363건 = loo_train (마지막이 라벨)
  - `evaluation/` 22,363건 = loo_train + [val]
  - `testing/` 22,363건 = loo_train + [val, test]
  - 전부 GZIP TFRecord, 폴더명은 GRID config 규칙(`training/evaluation/testing`) 준수
- 되읽기 검증: 4개 폴더 모두 건수·스키마 일치 ✓

## 3. 임베딩 추출 — 완료 (학과 GPU 서버)

맥북(Intel CPU, GPU 없음)에서는 전체 6.5시간 예상이라 중단했고, 윈도우 데스크탑 이관을 준비하던 중
학교 VPN이 연결되어 **학과 GPU 서버(dsl05)에서 실행**했습니다.

| 항목 | 값 |
|---|---|
| 노드 / GPU | `hpc-stat1` (`partition1`), **NVIDIA RTX 6000 Ada 48GB** 1장 |
| 모델 | `google/flan-t5-xl` 인코더 (1.22B 파라미터), fp32 |
| 입력 | `grid_data/beauty_A/items` (12,101 아이템, Split A의 `item_text`) |
| 배치 | `batch_size_per_device=64`, `num_workers=0` |
| 소요 시간 | **3분 50초** (190 배치, 약 0.90 it/s) |
| Slurm Job | 116247(추론 성공·병합 실패) → 116315(패치 후 완주, `COMPLETED`) |

### 산출물

```
embeddings/beauty_A/
├── merged_predictions_tensor.pt   # 12,101 × 2048 float32 (95MB) ← 최종 산출물
├── merged_predictions.pkl         # 원본 리스트 형식 (214MB)
└── run_116315.log                 # 실행 로그
```

`.pt`는 **행 인덱스 = item_id** 인 텐서입니다 (id 0~12,100 전부 존재, 결측·0벡터 없음).

### 검증 결과

- 형태 12,101 × 2048 / item_id 0~12,100 **전부 고유**, 누락 없음
- 전(全) 0 벡터 행 **0개**, 임베딩 노름 평균 1.356
- **의미 검증(최근접 이웃)**: 무작위 상품 4건의 top-3 이웃이 모두 같은 브랜드·같은 카테고리로 나옴
  - Garnier 로션 → Garnier 젤크림 / Garnier 안티링클 크림 (cos 0.92, 0.91)
  - Conair 조명 거울 → Revlon 조명 거울 / Conair 양면 조명 거울 (cos 0.92, 0.91)
  - Biosilk 실크테라피 세럼 → Biosilk 실크테라피 트리트먼트 (cos 0.94)
  - Boots No7 페이스마스크 → Boots No7 세럼 2종 (cos 0.89, 0.89)
- **재현성 확인**: 코드 패치 후 재실행한 결과가 첫 실행과 **비트 단위로 동일** (`torch.equal` True, max abs diff 0.0)

### 실행 중 해결한 문제

1. **Slurm 로그도 없이 즉시 FAILED** — 공유 저장소가 마스터에서는 `/data1/dsl05`, 계산 노드에서는
   `/mnt/data1/dsl05` 로 마운트 경로가 다름. `#SBATCH --output` 이 계산 노드에 없는 경로라 로그 파일
   생성부터 실패. 스크립트 전체를 `/mnt/data1` 기준으로 수정 + `--chdir` 지정.
2. **`GLIBC_2.33 not found` (cryptography)** — 마스터는 Ubuntu 22.04(glibc 2.35), 계산 노드는
   20.04(glibc 2.31). pip가 마스터 기준 최신 휠을 받아서 계산 노드에서 로드 실패.
   `cryptography==42.0.8` 로 고정해 해결.
3. **`conda activate` 불가** — conda가 `/data1` prefix로 설치돼 계산 노드에서 경로가 깨짐.
   python 절대경로(`/mnt/data1/.../envs/grid/bin/python`) 직접 호출로 우회.
4. **`RuntimeError: generator raised StopIteration`** — `items/` TFRecord가 파일 1개인데
   `num_workers=2` 라 워커 하나가 빈 이터레이터를 받음. `num_workers=0` + `timeout=0` +
   `persistent_workers=false` 로 해결.
5. **★ GRID 코드 버그 — 단일 GPU 실행에서 병합 단계가 항상 실패**
   `src/utils/inference_utils.py` 의 `on_predict_end` 가 `if trainer.global_rank != None:` 로
   분기하는데, rank 0에서도 이 조건은 **항상 참**이라 프로세스 그룹 없이 `torch.distributed.barrier()`
   를 호출해 `ValueError: Default process group has not been initialized` 로 죽습니다.
   추론 자체는 끝나고 pkl 조각 190개는 남지만 최종 `.pt` 가 안 만들어집니다.
   → `if torch.distributed.is_available() and torch.distributed.is_initialized():` 로 3곳 수정.
   패치는 `desktop_handoff/GRID` 와 서버 양쪽에 반영했고, 패치 후 job 116315가 정상 완주했습니다.
   (첫 실행분은 `code/server/merge_preds.py` 로 수동 병합했고, 두 결과는 동일함을 확인)

## 4. SID(Semantic ID) 생성 — 완료

임베딩을 RQ-KMeans 로 양자화해 아이템마다 이산 토큰 튜플(SID)을 부여했습니다.

| 설정 | 값 |
|---|---|
| 방식 | Residual K-means (`configs/experiment/rkmeans_*_flat.yaml`) |
| 코드북 레벨 | **3단계 / 4단계 두 가지** |
| 레벨당 코드북 크기 | **256** |
| 충돌 처리 | 마지막 자리에 **0부터 시작하는 일련번호** 추가 |
| 레벨당 학습 스텝 | 30 (아래 주의 참고) |
| Slurm Job | 116566, `COMPLETED`, **6분 59초** (두 설정 학습+부여 전부 포함) |

SID 형식 예시 (같은 아이템의 3단계 / 4단계 결과):

```
item 0  "WAWO 15 Color Professionl Makeup Eyeshadow..."
   3단계 → 16-106-240-0        (코드 3자리 + 충돌 구분 1자리)
   4단계 → 199-23-73-216-0     (코드 4자리 + 충돌 구분 1자리)
```

### 결과 비교

| 항목 | 3단계 | 4단계 |
|---|---|---|
| 표현 가능한 조합 | 256³ = 16,777,216 | 256⁴ = 4,294,967,296 |
| 실제 고유 코드 조합 | 10,645 | 11,338 |
| **코드만으로 유일한 비율** | 87.97% | **93.69%** |
| 충돌에 걸린 아이템 | 2,347개 (19.40%) | 1,248개 (10.31%) |
| 최대 충돌 그룹 크기 | 33 | 18 |
| 마지막 자리 최댓값 | 32 | 17 |
| 레벨별 사용된 코드 수 | 256/256/256 | 256/256/256/256 |
| 전체 SID 유일성 | 12,101/12,101 ✓ | 12,101/12,101 ✓ |
| 학습 MSE (epoch) | 0.000515 | 0.000511 |

- **모든 레벨에서 256개 코드를 전부 사용**했습니다 (죽은 코드 없음 = 코드북 활용도 양호).
- 4단계가 충돌을 절반으로 줄이지만(19.4% → 10.3%), 재구성 오차(MSE)는 거의 같습니다.
  시퀀스 길이가 아이템당 4토큰 → 5토큰으로 늘어나는 비용과 맞바꾸는 셈입니다.

### 충돌 그룹이 의미적으로 타당한지 확인

충돌 = 임베딩이 거의 같은 아이템들이므로, **같은 제품의 색상·향 변형**이 묶여야 정상입니다.
실제로 그렇게 나왔습니다:

- 3단계 최대 그룹(코드 `36-78-170`, 33개): Harmony Gelish UV 젤 폴리시 **색상 변형들**
- 4단계 최대 그룹(코드 `37-177-177-18`, 18개): REVLON Colorburst Lip Butter **색상 변형들**
  (Creamsicle / Sugar Plum / Fig Jam / Candy Apple / Sweet Tart ...)

### 검증

- 코드 값 범위 0~255 ✓ / 마지막 자리 0부터 시작 ✓
- 비충돌 아이템의 마지막 자리는 전부 0 ✓
- 코드+마지막자리를 합친 전체 SID가 12,101개 **모두 유일** ✓

### 주의 — 우리가 GRID 기본 동작에서 바꾼 것 2가지

1. **충돌 번호를 0부터 시작하도록 수정.** GRID 원본 `deduplicate_rows_in_tensor` 는
   `arange(1, N+1)` 이라 충돌 그룹이 1..N 을 받고 비충돌 아이템만 0을 받습니다(코드 자체도
   docstring 과 불일치). 우리 규칙("마지막 자리 = 같은 코드를 공유하는 아이템들의 0부터 시작하는
   일련번호", TIGER 논문 방식)에 맞춰 `arange(0, N)` 으로 고쳤습니다.
   `src/utils/tensor_utils.py` 에 주석과 함께 반영돼 있습니다.
2. **레벨당 학습 스텝을 고정.** GRID 는 `steps_per_layer = trainer.max_steps // n_layers` 로
   나눠 쓰기 때문에, 기본값 `max_steps=30` 을 그대로 두면 **3단계는 레벨당 10스텝, 4단계는 7스텝**이
   되어 두 설정의 조건이 달라집니다. `max_steps = 30 × 레벨수` 로 주어 양쪽 다 레벨당 30스텝
   (아이템 12,101개 · 배치 2048 기준 약 5 에폭)으로 맞췄습니다.

### 산출물

```
sid/
├── L3/sid_tensor.pt      # (4, 12101)  ← GRID 후처리가 transpose 해서 저장 (레벨+1, 아이템)
├── L3/train_csv/         # 학습 지표
├── L4/sid_tensor.pt      # (5, 12101)
├── L4/train_csv/
└── run_116566.log
```

> `.pt` 는 **(레벨+1) × 아이템** 형태입니다. 아이템 기준으로 쓰려면 `t.T` 로 전치하세요.
> 행 인덱스(전치 후)는 `item_id` 와 동일합니다.

재현: `code/server/sid_beauty.sh` (sbatch 한 번으로 3단계·4단계 모두 생성)

## 5. 재현 방법

서버 스크립트 일체는 `code/server/` 에 있습니다.

| 파일 | 역할 |
|---|---|
| `server_setup.sh` | 마스터 노드에서 miniconda + GRID 의존성 환경 구축 |
| `prefetch_model.sh` | flan-t5-xl(11GB) 을 `$HOME/hf_home` 에 사전 다운로드 |
| `embed_beauty.sh` | 임베딩 추출 sbatch 스크립트 (검증된 최종본) |
| `sid_beauty.sh` | **SID 생성 sbatch 스크립트** — 3단계·4단계를 한 번에 학습+부여 |
| `merge_preds.py` | 병합이 실패했을 때 pkl 조각을 수동 병합 (패치 후엔 불필요) |
| `verify_embed.py` | 산출물 형태·최근접 이웃 검증 |

CSV 변환 스크립트는 `code/export_csv.py`(데이터셋·임베딩), `code/export_sid_csv.py`(SID) 입니다.

```bash
ssh -p 37220 dsl05@165.132.80.36     # 데스크탑에 SSH 키 등록 완료 — 비밀번호 불필요
sbatch ~/recsys/embed_beauty.sh      # 임베딩 (약 4분)
sbatch ~/recsys/sid_beauty.sh        # SID 3단계+4단계 (약 7분)
```

상세 환경·함정은 `GPU_서버_사용가이드.md` 에 실측 기준으로 갱신해 두었습니다.

## 5-1. 부수 산출물 / 현재 환경

- **학과 서버 dsl05**: miniconda(`envs/grid`, py3.10 + torch 2.6.0+cu124), flan-t5-xl 캐시 11GB,
  GRID 코드·TFRecord 업로드 완료. 데스크탑 SSH 키 등록 완료. **초기 비밀번호는 팀 합의대로 아직 변경 안 함.**
- **윈도우 데스크탑 로컬 폴백**: `desktop_handoff/condaenv` (Python 3.11 + GPU torch).
  이 PC의 파이썬이 3.13뿐이라 `tensorflow-cpu==2.18.0` 이 안 깔려서 conda로 3.11 환경을 따로 만들었습니다.
  RTX 4060 Ti(8GB)로도 돌릴 수 있게 준비돼 있으나, 서버가 훨씬 빨라 사용하지 않았습니다.
- 맥 로컬 실행 환경: SSD의 `gridwork.sparseimage`
- 팀 깃허브 레포(hoohdini/26-2_Modeling_RecSys)에 GRID/MaskGR 스냅샷·전처리 코드·README 푸시됨
  (`GPU_서버_사용가이드.md` 는 계정 정보 포함이라 미업로드)

## 6. CSV 내보내기 (발표자료용)

전처리 데이터셋과 임베딩을 엑셀에서 바로 열 수 있게 CSV로 변환해 `csv_export/` 에 두었습니다.
(생성 스크립트 `code/export_csv.py`, 설명은 `csv_export/README.md`, 인코딩 UTF-8 BOM)

- `csv_export/dataset/` — 아이템 마스터 12,101 / 유저 22,363 / Split A 상호작용 183,348 /
  Split B 윈도우·타깃·콜드등급 / 관계 간선 403,716 / 전체 요약
- `csv_export/embeddings/` — 미리보기 · 아이템별 통계 · PCA 2D·50D · 최근접 이웃 top-5 ·
  전체 원본 12,101 × 2048 (235MB)
- `csv_export/sid/` — 아이템별 SID(3단계·4단계) · 충돌 그룹 목록 · 레벨별 코드 사용 분포 ·
  두 설정 비교 요약 (`sid_summary.csv`)

`item_id` 체계가 pkl·임베딩·CSV 전부 동일하므로 `A_items.csv` 행 순서에 임베딩을 그대로 붙일 수 있습니다.
전체 CSV는 원본 `.pt` 와 값이 일치함을 대조 확인했습니다.

발표에 바로 쓸 만한 것: `dataset_summary.csv`(규모 요약), `B_windows_summary.csv`(분할 이상 근거),
`relations_summary.csv`(관계 구성비), `item_nearest_neighbors_top5.csv`(임베딩 품질 근거),
`item_embeddings_pca2d.csv`(산점도), `pca_explained_variance.csv`(상위 50개 주성분이 전체 분산의 67.2%).

## 7. 다음 단계 (아직 안 함)

1. **전처리 담당자 확인 필요** — 1절의 문제 ①(Split B 시간 분할)·②(`item_text` 순서 비결정성).
   특히 ①은 콜드스타트 실험을 못 돌리는 수준이라 우선 처리 필요.
2. **TIGER 학습** — 이번 SID 를 입력으로 생성 모델 학습
   (`configs/experiment/tiger_train_flat.yaml` / `tiger_inference_flat.yaml`).
   3단계 SID 와 4단계 SID 중 어느 쪽을 쓸지 결정 필요 (충돌률 vs 시퀀스 길이 트레이드오프).
3. RQ-VAE 방식 SID 도 뽑아 RQ-KMeans 와 비교할지 결정 (`rqvae_train_flat.yaml` 준비돼 있음)
4. Split B 수정본이 오면 동일 파이프라인으로 재실행
   (`to_grid_v2.py` → `embed_beauty.sh` → `sid_beauty.sh`)
