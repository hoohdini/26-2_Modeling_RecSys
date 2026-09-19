# 런북 — 텍스트 SID로 3트랙 완성하기

> 목표: `sid/L4/sid_tensor.pt`(텍스트 SID) 하나로 **LOO · Temporal · 다양성** 세 트랙을
> TIGER와 MaskGR 양쪽에서 채우는 것. 실험표 ①과 ④에 해당합니다.
> 진행 상황은 아래 표에 ✅/⬜ 로 표시돼 있습니다.

---

## 지금 어디까지 됐나

| | LOO | Temporal (W3/W4/W5) | 다양성 |
|---|---|---|---|
| **고전 베이스라인** | ✅ | ✅ **완료** | ✅ (두 트랙에 포함) |
| **TIGER** (①) | ✅ | ⬜ **GPU 필요 — 재학습** | ✅ LOO분 완료 / ⬜ Temporal분 |
| **MaskGR** (④) | ⬜ **GPU 필요 — 학습** | ⬜ **GPU 필요 — 학습** | ⬜ |

다양성 트랙은 별도 실행이 아닙니다 — `tiger_to_eval.py` 가 정확도와 함께 뱉으므로
LOO/Temporal 덤프가 생기는 순간 같이 나옵니다.

---

## ★ 왜 재학습이 필요한가

**MaskGR** 은 마스크 확산 트랜스포머라 TIGER 체크포인트를 재활용할 수 없습니다. 새로 학습합니다.

**TIGER 도 Temporal 에서는 재학습이 필요합니다.** 윈도우마다 train 이 따로이기 때문입니다.
지금 있는 `clip_L4` 는 LOO(=전 기간) 로 학습됐으므로, 그 모델로 W3 테스트를 채점하면
**미래 데이터로 과거를 맞히는 셈**이라 지표가 통째로 무의미해집니다.

```
W3 스냅샷:  train=C1        val=C2   test=C3
W4 스냅샷:  train=C1+C2     val=C3   test=C4
W5 스냅샷:  train=C1+C2+C3  val=C4   test=C5
```

즉 **윈도우당 모델 하나**입니다. 텍스트 SID 기준으로 TIGER 3개 + MaskGR 3개(Temporal)
\+ MaskGR 1개(LOO) = **학습 7회**가 남았습니다.

---

## 1. 이미 끝난 것 — Temporal 고전 베이스라인

GPU 없이 CPU 로 계산했습니다 (`results/baseline_splitB_W*.json`).

| 윈도우 | 채점 유저 | 제외된 유저(학습이력 없음) | EASE_R R@20 | EASE_R N@20 | EASE_R TailExp@20 |
|---|---|---|---|---|---|
| W3 | 4,917 | **6,441 (56.7%)** | 0.0081 | 0.0048 | 0.0533 |
| W4 | 6,868 | 4,621 (40.2%) | 0.0150 | 0.0083 | 0.0898 |
| W5 | 7,264 | 2,970 (29.0%) | 0.0145 | 0.0085 | 0.0989 |

> ⚠️ **테스트 유저의 29~57% 가 그 윈도우에 학습 이력이 없습니다.** 협업필터 계열은 애초에
> 예측을 못 하므로 채점에서 제외했고, 몇 명을 뺐는지 `_source.n_user_dropped_no_history` 에
> 남겼습니다. **트랙 간에 유저 수가 다르므로 LOO 수치와 직접 비교하면 안 됩니다.**

> Temporal 의 정확도가 LOO 보다 훨씬 낮은 것은 정상입니다 (EASE_R R@20: LOO 0.0755 → W5 0.0145).
> 미래를 맞히는 과제라 원래 어렵습니다. **비교는 같은 트랙 안에서만** 하십시오.

재현:

```bash
for W in W3 W4 W5; do
  python code/baselines_a.py --split Beauty_split_B.pkl --window $W
done
```

---

## 2. Temporal 입력 만들기 (로컬, tensorflow 필요)

```bash
python code/to_grid_b.py Beauty_split_B.pkl grid_data/beauty_B
```

윈도우 하나가 독립 데이터셋(`training/ evaluation/ testing/`)이 됩니다.

```
grid_data/beauty_B/W3   training 7,089 · evaluation 5,716 · testing 4,917
grid_data/beauty_B/W4   training 12,452 · evaluation 7,837 · testing 6,868
grid_data/beauty_B/W5   training 16,676 · evaluation 8,774 · testing 7,264
```

행 수가 유저 수보다 적은 것은 정상입니다 — 아이템이 1개뿐인 시퀀스는 버립니다
(히스토리 없이 다음을 맞히라는 건 다른 과제입니다).

서버로 올립니다:

```bash
scp -rP 37220 grid_data/beauty_B dsl05@165.132.80.36:/data1/dsl05/recsys/grid_data/
```

---

## 3. TIGER × Temporal (GPU) — 윈도우당 학습 1회

`tiger_beauty.sh` 와 `tiger_eval_dump.sh` 가 이제 `DATA` 를 받습니다.
**학습과 덤프에 반드시 같은 `DATA` 를 주십시오.**

```bash
SID=/data1/dsl05/recsys/sid_out/baseline/L4/infer/pickle/merged_predictions_tensor.pt

for W in W3 W4 W5; do
  D=/data1/dsl05/recsys/grid_data/beauty_B/$W
  # ① 학습
  sbatch --export=ALL,TAG=tg_text_B_$W,SID=$SID,DATA=$D,CLIP=1.0 tiger_beauty.sh
  # ② 덤프 (학습이 끝난 뒤)
  sbatch --export=ALL,TAG=tg_text_B_$W,SID=$SID,DATA=$D,TOPK=200,BATCH=32 tiger_eval_dump.sh
done
```

---

## 4. MaskGR (GPU) — LOO 1회 + Temporal 3회

먼저 배치(1회만) 후 스모크 테스트를 거칩니다. 절차는 `docs/HANDOFF_MaskGR.md` 1절.

```bash
# 스모크 — 50스텝만 돌려 죽지 않는지 확인
sbatch --export=ALL,TAG=mg_text_smoke,SID=$SID,SMOKE=1 maskgr_train.sh

# ④ LOO
./run_maskgr.sh text

# Temporal 3윈도우
./run_maskgr.sh text temporal          # WINDOWS="W3 W4 W5" 로 조절
```

---

## 5. 덤프 → 지표 (로컬, GPU 불필요)

```bash
SIDP=sid/L4/sid_tensor.pt

# LOO + 다양성
python code/tiger_to_eval.py --pred tiger_runs/mg_text_A/eval_dump_c200_t0.01_s5/test_predictions_rank0.pkl \
                             --sid $SIDP --label mg_text

# Temporal + 다양성 (윈도우별)
for W in W3 W4 W5; do
  python code/tiger_to_eval.py \
    --pred tiger_runs/tg_text_B_$W/eval_dump_k200/test_predictions_rank0.pkl \
    --sid $SIDP --split Beauty_split_B.pkl --window $W --label tg_text
  python code/tiger_to_eval.py \
    --pred tiger_runs/mg_text_B_$W/eval_dump_c200_t0.01_s5/test_predictions_rank0.pkl \
    --sid $SIDP --split Beauty_split_B.pkl --window $W --label mg_text
done

# 표
python code/compare_table.py results/baseline_splitA_loo.json \
       results/tiger_clip_L4_loo.json results/tiger_mg_text_loo.json --k 10
python code/compare_table.py results/baseline_splitB_W5.json \
       results/tiger_tg_text_W5_temporal.json results/tiger_mg_text_W5_temporal.json --k 10
```

Temporal 결과에는 전처리 담당이 정의한 **콜드 등급별 recall**
(`unseen` / `very_rare` / `rare` / `normal`)이 함께 찍힙니다.

---

## 6. 검증 순서 — 숫자를 믿기 전에

TIGER 에서 쓴 3겹을 그대로 씁니다 (`docs/TRACK_C_보고서.md` 2절).

1. **얕은 덤프로 학습 로그와 대조** — `TOPK=10`(TIGER) / `CAND=10`(MaskGR) 로 한 번 돌려
   학습 로그의 `val/recall@5` 와 맞는지. 맞으면 덤프가 평가 경로를 안 바꾼 것.
2. **정답 SID 대조** — LOO 에서는 도구가 자동으로 전건 대조합니다.
   Temporal 은 placeholder 라 건너뜁니다(도구가 이유를 찍습니다).
3. **노이즈 바닥** — 시드를 바꿔 최소 2회 돌리고, 차이가
   **NDCG@10 0.0016 · APLT@10 0.0176** 보다 큰지 확인.

> ⚠️ **Temporal 에서는 모델 내부 지표(val/recall@5)를 믿지 마십시오.** testing 시퀀스에
> 붙인 정답은 마스크 자리를 만들기 위한 placeholder 하나뿐이라, 내부 지표는 그 하나만
> 정답으로 칩니다. 우리 지표는 정답 집합 전체로 채점합니다.

---

## 남은 GPU 작업 요약

| 작업 | 횟수 | 비고 |
|---|---|---|
| TIGER × Temporal | 3 (W3/W4/W5) | 윈도우당 train 이 달라 재학습 필수 |
| MaskGR × LOO | 1 | ④번 칸 |
| MaskGR × Temporal | 3 | |
| **합계** | **7회 학습 + 덤프** | GPU 동시 2장 제한이라 슬럼이 큐잉 |

각 학습 뒤 덤프 2회(검증용 얕은 것 + 본 덤프)가 붙습니다.
