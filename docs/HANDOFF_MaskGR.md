# MaskGR 파이프라인

> 실험표 오른쪽 열(④⑤⑥⑦) · 관련: `docs/TRACK_C_보고서.md`, `docs/HANDOFF_트랙C_다양성평가.md`
> 결론부터: **SID 하나를 받아 세 트랙 지표까지 한 줄로 갑니다.** `./run_maskgr.sh <SID이름> all`

## 이 파이프라인이 서 있는 자리

```
베이스라인   GRID 원본 + gradient clipping (스텝 수만 조정)  →  텍스트 SID
                                    │
              ┌─────────────────────┼─────────────────────┐
         그래프 증강              CRAB 적용            결합
              └─────────────────────┼─────────────────────┘
                                    ▼
                        SID 텐서 (.pt) — 어느 것이든 동일 규격
                                    ▼
                  ┌─────────────────┴─────────────────┐
              TIGER (GRID)                      MaskGR (확산)   ← 이 문서
                  └─────────────────┬─────────────────┘
                                    ▼
                   검증 3트랙:  LOO · Temporal · 다양성
```

**SID 종류가 무엇이든 파이프라인은 같습니다.** `SID=` 만 바뀝니다.

---

## 0. 왜 갈아끼우기만 하면 되는가

MaskGR 저장소를 뜯어보니 **GRID와 규약이 그대로 겹칩니다.**

| 항목 | GRID(TIGER) | MaskGR | |
|---|---|---|---|
| SID 입력 | `semantic_id_path` | `sid_data_path` | 같은 `.pt` 텐서, 키 이름만 다름 |
| 데이터 폴더 | `training/ evaluation/ testing/` | 동일 | **`grid_data/beauty_A` 재사용** |
| 계층 수 | `num_hierarchies=5` | 동일 | 4단계 SID + 충돌 구분자 |
| 시퀀스 길이 | 120 | 120 | |
| 평가 진입점 | `eval_step` | `eval_step` | **시그니처까지 동일** |
| 생성 결과 전달 | `evaluator(marginal_probs, generated_ids, labels)` | 동일 | |
| 생성 결과 모양 | `(B, C, H)` | 동일 | |

그래서 **트랙 C 하네스(`tiger_to_eval.py` 이하)가 수정 없이 MaskGR 덤프를 읽습니다.**

---

## 1. 준비해 둔 파일

```
code/run_maskgr.sh                러너 — SID 하나로 학습·덤프를 트랙별로 제출
code/to_grid_b.py                 Split B 윈도우 → TFRecord (Temporal 트랙 입력)
code/server/maskgr_dumper.py      eval_step / evaluator 를 런타임에 감싸는 덤프 콜백
code/server/maskgr_eval_dump.py   학습 없이 test 만 도는 엔트리포인트
code/server/maskgr_train.sh       학습 sbatch (함정 전부 눌러 둠)
code/server/maskgr_eval_dump.sh   덤프 sbatch (확산 다이얼 노출)
```

기존 하네스에 추가한 것 (하위 호환 유지 — 기존 명령은 그대로 동작합니다):

```
evaluate.py        load_split_b(pkl, window)   Temporal 윈도우 로더
tiger_to_eval.py   --window <라벨>              주면 Split B 로 읽고 콜드 등급별 recall 도 출력
```

**MaskGR 기존 파일은 한 줄도 고치지 않습니다.** 신규 파일만 얹고, 테스트 시작 시점에
두 지점을 감쌌다가 끝나면 원상복구합니다 — GRID에서 쓴 방식 그대로입니다.

### 서버 배치 (1회만)

```bash
scp -P 37220 code/server/maskgr_eval_dump.py dsl05@165.132.80.36:/data1/dsl05/recsys/MaskGR/src/
ssh -p 37220 dsl05@165.132.80.36 "mkdir -p /data1/dsl05/recsys/MaskGR/src/callbacks && \
  touch /data1/dsl05/recsys/MaskGR/src/callbacks/__init__.py"
scp -P 37220 code/server/maskgr_dumper.py \
  dsl05@165.132.80.36:/data1/dsl05/recsys/MaskGR/src/callbacks/
scp -P 37220 code/server/maskgr_train.sh code/server/maskgr_eval_dump.sh \
  dsl05@165.132.80.36:/data1/dsl05/recsys/
```

conda 환경은 별도로 하나 필요합니다 (GRID는 torch 2.6, MaskGR requirements는 별도):

```bash
conda create -n maskgr python=3.10 -y && conda activate maskgr
pip install -r MaskGR/requirements.txt
```

---

## 2. 실행 — 러너 한 줄

```bash
./run_maskgr.sh text                # ④ 베이스라인 (텍스트 SID, LOO)
./run_maskgr.sh gsid_a01            # ⑥ 그래프 SID
./run_maskgr.sh crab                # ⑤ CRAB 주소
./run_maskgr.sh gsid_a01 all        # LOO + Temporal 전부
```

러너가 하는 일: **학습 → 검증덤프(CAND=10) → 본덤프(CAND=200)** 를 `--dependency` 로
묶어 제출합니다. 기다리지 않고 큐에 넣기만 하며, GPU 동시 2장 제한은 슬럼이 처리합니다.
덤프가 끝난 뒤 실행할 로컬 명령을 러너가 마지막에 출력합니다.

### 세 트랙이 어떻게 갈라지는가

| 트랙 | 데이터 | 무엇이 다른가 |
|---|---|---|
| **LOO** | `grid_data/beauty_A` | 유저당 정답 1개. 지금까지의 모든 수치가 이 트랙 |
| **Temporal** | `grid_data/beauty_B/<윈도우>` | **윈도우 하나가 데이터셋 하나.** 유저당 정답 여러 개 |
| **다양성** | 위 두 덤프를 재활용 | 별도 학습 없음. 같은 덤프에 APLT/Coverage/도달가능성을 얹습니다 |

다양성 트랙은 **따로 돌리는 게 아닙니다.** `tiger_to_eval.py` 가 정확도와 다양성 지표를
한 번에 뱉으므로 LOO·Temporal 덤프에서 자동으로 같이 나옵니다.

### Temporal 트랙 준비 (윈도우 → TFRecord)

```bash
python3 code/to_grid_b.py Beauty_split_B.pkl grid_data/beauty_B          # 전 윈도우
python3 code/to_grid_b.py Beauty_split_B.pkl grid_data/beauty_B -w W5    # 하나만
```

윈도우 하나가 독립 데이터셋(`training/ evaluation/ testing/`)이 되므로,
`DATA=grid_data/beauty_B/W5` 로 주면 학습·덤프가 그대로 돕니다.

> **Split B 재전처리 완료** (2026-08-22). 리뷰 건수 분위수로 위치를 잡고 날짜값으로
> 스냅하는 방식이라 W3/W4/W5 가 전부 학습 가능한 규모입니다.
>
> | 윈도우 | 학습 유저 | 채점 유저 | 정답 | train 상호작용 |
> |---|---|---|---|---|
> | W3 | 9,455 | 11,358 | 39,728 | 39,563 |
> | W4 | 14,589 | 11,489 | 39,619 | 79,342 |
> | W5 | 18,110 | 10,234 | 39,813 | 119,070 |
>
> `to_grid_b.py` 와 `evaluate.load_split_b` 는 **신·구 스키마를 모두 받습니다.**
> (신: `windows` 가 dict + `train_seq` + `cold_tiers` 최상위 / 구: list + `train_user_seq`)

> ⚠️ **Temporal 은 유저당 정답이 여러 개**입니다 (지금 W5 기준 테스트 유저 21,786명 중
> 21,397명이 2개 이상). "마지막 아이템이 라벨" 규약이 안 맞아서, testing 시퀀스에는
> 정답 하나를 **마스크 자리를 만들기 위한 placeholder** 로만 붙이고 채점은 split 의
> 정답 집합 전체로 합니다. 그래서 **모델 내부 지표(val/recall@5)는 우리 지표와
> 다릅니다** — Temporal 에서는 내부 지표를 믿지 마십시오.

### 로컬 지표 (덤프가 끝난 뒤)

```bash
# LOO + 다양성 (한 번에 나옵니다)
python code/tiger_to_eval.py \
  --pred tiger_runs/mg_gsid_a01_A/eval_dump_c200_t0.01_s5/test_predictions_rank0.pkl \
  --sid sid/gsid_a01/L4/sid_tensor.pt --label mg_gsid_a01

# Temporal (윈도우별) — --window 를 주면 Split B 로 읽습니다
python code/tiger_to_eval.py \
  --pred tiger_runs/mg_gsid_a01_B_W5/eval_dump_c200_t0.01_s5/test_predictions_rank0.pkl \
  --sid sid/gsid_a01/L4/sid_tensor.pt \
  --split Beauty_split_B.pkl --window W5 --label mg_gsid_a01

# 한 표로 (TIGER 열과 나란히)
python code/compare_table.py results/baseline_splitA_loo.json \
       results/tiger_clip_L4_loo.json results/tiger_mg_gsid_a01_loo.json --k 10
```

Temporal 결과에는 전처리 담당이 정의한 **콜드 등급별 recall**
(`unseen` / `very_rare` / `rare` / `normal`)이 함께 찍힙니다. 우리 버킷(학습 등장 횟수)과는
다른 정의라 섞지 않고 따로 보고합니다.

---

## 3. ★ 순서 — 텍스트 SID(④)를 먼저

**그래프 SID나 CRAB 주소를 기다리는 동안 텍스트 SID로 ④번 칸을 끝내 두십시오.**
④가 있어야 ⑤·⑥이 나왔을 때 **"확산 덕분인가 주소 덕분인가"** 를 분해할 수 있습니다.

```bash
sbatch --export=ALL,TAG=mg_text_A,SID=<텍스트 SID>,SMOKE=1 maskgr_train.sh   # 먼저 스모크
./run_maskgr.sh text                                                          # 본 실행
```

같은 이유로 **주소를 바꿀 때는 TIGER 쪽(왼쪽 열)도 같은 주소로 돌려야** 합니다.
한쪽만 돌리면 주소 효과와 모델 효과가 섞입니다.

---

## 4. 확산 다이얼 — 트랙 C가 기다린 "생성 자체의 손잡이"

TIGER 곡선은 **생성 후 재정렬**이라 후보 풀 밖으로 못 나갑니다(보고서 4-3절).
MaskGR은 생성 분포 자체를 바꿀 수 있고, `diffusion_config` 에 손잡이가 여럿 있습니다.

| 손잡이 | 기본값 | 성격 | 스윕 제안 |
|---|---|---|---|
| **`unmasking_temperature`** | **0.01** | 사실상 argmax. 키우면 생성이 다양해짐 | **1순위 다이얼**: 0.01 / 0.1 / 0.3 / 0.5 / 1.0 / 2.0 |
| `num_steps` | `num_hierarchies` | 확산 스텝 수 | 2순위: NH / 2·NH / 4·NH |
| `unmasking_type` | `top-prob` | 어느 칸부터 확정할지 | 이산 변형: `random` 과 대조 |
| `noise_schedule` | `uniform` | 마스킹 일정 | 이산 변형: `edm`, `last-token-ar` |
| `inference_type` | `beam-search-generation` | `constrained-` 는 접두사 제약 | **GRID에서 버그로 못 켠 그것 — 여기선 될 수 있음** |
| `num_candidates` | 10 | 빔 폭 (TIGER의 `top_k_for_generation`) | **실험 간 고정.** 곡선용은 200 |

**곡선 그리는 법**: `TEMP` 만 훑으면서 매번 덤프 → `tiger_to_eval.py` → `compare_table.py`.
덤프 폴더 이름에 설정이 박히고(`eval_dump_c200_t0.3_s5`), `dump_meta.json` 에도 남습니다.

> ⚠️ **TIGER 곡선과 MaskGR 곡선을 같은 그림에 겹칠 때 성격을 표기할 것.**
> TIGER는 재정렬 참조선, MaskGR은 생성 다이얼입니다. 보고서 7절 Q4의 결정이 이것입니다.
> 그리고 MaskGR 덤프의 `scores` 는 빔 누적 점수라 TIGER의 `marginal_probs` 와
> **같은 물건이 아닙니다** — 재정렬 다이얼을 MaskGR에 그대로 걸면 안 됩니다.

---

## 5. 함정 (GRID에서 겪은 것이 그대로 재현됩니다)

1. **`num_workers=8` / `timeout=60` / `persistent_workers=true` 가 MaskGR 기본값입니다.**
   TFRecord가 split당 1파일이라 워커가 굶어 죽습니다. 스크립트에서 전부 0/false로 눌러 뒀습니다.
2. **`src/inference.py` 가 저장소에 없습니다.** Makefile의 `make inference` 는 그 파일을
   부르는데 실제로는 존재하지 않습니다. 그래서 `maskgr_eval_dump.py` 를 따로 뒀습니다.
3. **기본 `batch_size` 가 2048** 입니다. 우리 규모엔 과해서 256으로 낮춰 잡았습니다.
4. **기본 로거가 wandb** 입니다. 오프라인 서버에서 죽으므로 `logger=csv` + `WANDB_MODE=offline`.
5. `train.py` 의 test 블록은 GRID보다 낫습니다(빈 문자열이면 `cfg.ckpt_path` 폴백).
   그래도 체크포인트 콜백이 없으면 랜덤 가중치로 갑니다 — 엔트리포인트에서 명시적으로 막았습니다.

---

## 6. 검증 순서 (덤프가 옳은지부터)

1. `CAND=10` 덤프 → 학습 로그의 `val/recall@5` 와 대조. 일치하면 덤프가 평가 경로를 안 바꾼 것.
2. 정답 SID 대조 — 덤프의 `label_sid` 를 `Beauty_split_A.pkl` 의 `loo_test` 와 전건 비교.
   (Temporal 은 placeholder 라서 이 대조를 건너뜁니다 — 도구가 알아서 건너뛰고 이유를 찍습니다.)
3. `tiger_to_eval.py` 자체 계산값이 MaskGR 내부 지표와 맞는지.

세 겹 다 TIGER에서 쓴 방법 그대로입니다 (보고서 2절).

---

## 7. 아직 확인 못 한 것

- [ ] MaskGR requirements 와 서버 CUDA 조합 — 실제 설치 전엔 모릅니다
- [ ] `constrained-beam-search-generation` 이 실제로 도는지 (GRID의 같은 기능은 버그로 사망)
- [ ] MaskGR 학습이 30,000스텝에서 수렴하는지 — TIGER 예산을 그대로 가져왔을 뿐입니다
- [ ] `projection` / `project_generated_ids` 가 무엇으로 켜지는지 — 무효 SID를 줄일 수 있는 경로로 보입니다
- [x] ~~Temporal 윈도우 규모~~ — 재전처리로 해결. W3 9,455 / W4 14,589 / W5 18,110 유저
- [ ] CRAB 주소가 올라오면 `./run_maskgr.sh crab` 로 같은 경로를 태우면 됩니다 —
      파이프라인 변경 없음
