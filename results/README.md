# results/ — 지표 JSON 색인

**전부 재현 가능합니다.** 각 파일은 아래 명령이 만든 것이고, 같은 입력이면 같은 값이 나옵니다.
컷오프는 보고용으로 **@10** 을 씁니다(콜드 축만 @50). JSON 에는 @10/@20/@50 이 모두 들어 있습니다.

---

## 무엇이 무엇을 만들었나

| 파일 | 무엇 | 만든 명령 |
|---|---|---|
| `baseline_splitA_loo.json` | 고전 베이스라인 4종 (Random / MostPopular / ItemKNN / EASE_R) | `python code/baselines_a.py` |
| `tiger_clip_L4_loo.json` | **TIGER 베이스라인 (실험표 ①)** | `code/tiger_to_eval.py --pred <k50덤프> --label clip_L4` |
| `tiger_clip_L4_k10_loo.json` | 위와 같은 모델, 빔 폭 10 — **덤프 경로 검증용** | 같은 도구, `k10` 덤프 |
| `tiger_clip_L4_excl_seen_loo.json` | 이미 본 상품을 제외한 변형 (고전 베이스라인과 규칙 맞춤) | `--exclude-seen` |
| `pareto_clip_L4_k20.json` | 재정렬 다이얼 곡선 — **후보 50개 풀**, 12지점 | `code/pareto_dial.py --pred <k50> --k 20` |
| `pareto_clip_L4_k200_k20.json` | 재정렬 다이얼 곡선 — **후보 200개 풀**, 9지점 ★ 본 곡선 | `pareto_dial.py --pred <k200> --label clip_L4_k200 --k 20 --lams 0,0.2,0.5,1,2,3,5,8,12` |
| `pareto_clip_L4_exposure_k20.json` | 노출 인지 다이얼 곡선, 7지점 | `pareto_dial.py ... --mode exposure --lams 0,0.5,1,2,4,8,16` |
| `pareto_compare.json` | 체크포인트 4개 곡선 비교 + **노이즈 바닥** | `code/pareto_compare.py --ref ... --run ... --noise-pair "clip_L4,seed7" --k 10` |
| `reachability_clip_L4*.json` | 도달가능성 — 후보에 아예 안 오르는 상품 (체크포인트 4종) | `code/reachability.py --pred <k200> --label <태그>` |
| `significance_tiger.json` | TIGER vs 베이스라인 paired bootstrap | `code/significance.py --pred <k50> --k 10` |
| `tail_sensitivity.json` | 롱테일 경계(`tail_frac`) 민감도 | `python code/tail_sensitivity.py` |

전체 재현 명령은 `docs/TRACK_C_보고서.md` 9절에 있습니다.

---

## 파일 형태 두 가지

**① 지표 묶음** — `{라벨: {지표...}}`. `compare_table.py` 가 이 형식을 읽습니다.

```json
{ "clip_L4": { "recall@10": 0.0692, "ndcg@10": 0.0384, "tail_exposure@10": 0.0497,
               "coverage@10": 0.3887, "COLD_recall@50": 0.0429,
               "_source": { "pred_file": "...", "track": "loo", "window": null } } }
```

**② 곡선/분석** — `pareto_*`, `reachability_*`, `significance_*`, `tail_sensitivity`.
각자 고유 스키마이고, `pareto_*` 의 `points` 안에는 ①과 같은 지표 묶음이 들어 있어
`compare_table.py` 가 그대로 읽습니다.

---

## 지표 이름

| 키 | 뜻 |
|---|---|
| `recall@k` / `ndcg@k` / `hr@k` | 정확도 |
| **`tail_exposure@k`** | **다양성 대표축.** 발표에는 **APLT@k** 로 씁니다 |
| `aplt@k` | `tail_exposure@k` 와 **같은 값**입니다. 둘을 나란히 보고하지 마십시오 |
| `coverage@k` | 전체 12,101개 중 한 번이라도 추천된 상품 비율 |
| `exposure_gini@k` | 노출 집중도 (낮을수록 고르게) |
| `COLD_recall@50` | 학습 등장 5회 이하 상품이 정답일 때의 Recall@50 |
| `bucket_recall@50` | 인기 버킷별 recall (zero-shot / few-shot / low / mid / head) |
| `tier_recall@k` | **Temporal 트랙 전용.** 전처리 담당이 정의한 콜드 등급별 recall |

`_source` 에 어떤 덤프·SID·분할로 만든 값인지 들어 있습니다. **결과를 인용할 때 이걸 먼저
확인하십시오** — 특히 `track`(loo/temporal)과 `exclude_seen`.

---

## 주의

- `_INVALID` 키가 있는 파일은 **정답 누출이 있는 예측**으로 만든 것입니다. 정확도 지표를
  발표에 쓰면 안 됩니다. (지금은 그런 파일이 없습니다.)
- `truncated_at` 이 요구 컷오프보다 작으면 그 컷오프 지표는 **과소평가**된 값입니다.
  예: 유저당 생성 50개인데 `@50` 을 보면 사실상 전수라 다이얼에 반응하지 않습니다.
- 결과 파일은 **덮어쓰기**로 생성됩니다. 스윕할 때는 `--out` 으로 이름을 나누십시오.

---

## 2차 발표 실험 트랙 (2026-09-15 ~ 09-19 · 서버 자동 채점본)

전부 서버 `tiger_to_eval.py` 가 만든 ① 형식 JSON 이고, 판정표 원문은 `reports/` 에 있다. 제출 스크립트와 판정 규칙은 `code/run_*.sh` · `code/verdict_*.py` 머리 주석, 노션 1-1 페이지 참고.

| 폴더 | 내용 | 판정 스크립트 | 보고서 |
|---|---|---|---|
| `gsid_week/` | Beauty LOO 텍스트 · 튜닝식 β₀1.0 · 고전 α0.7 × 시드 42·7·13·3·21 = 15 (P1, n=5) | `verdict_gsid_week.py` | `reports/GSID_WEEK_보고.txt` |
| `jobs_v3_seeds/` | 구인구직 v3 추가 시드 3·21·99 × text/gsid = 6 (P3, 기존 3시드와 합쳐 n=6) | `verdict_v3.py` (JOBS_VER=v3 SEEDS=42,7,13,3,21,99) | 같은 파일 |
| `jobs_v4/` | 구인구직 v4 text/gsid × 6시드 + rewired × 3시드 = 15 | `verdict_v3.py` (JOBS_VER=v4) | `reports/V4_보고.txt` |
| `p2_temporal/` | Beauty Temporal W4·W5 × text/b10 × 시드 42·7·13 = 12 (P2, 30k 레시피 통일. 4셀은 기존 tg_* 덤프 재채점) | `verdict_p2_temporal.py` | `reports/P2_보고.txt` |

요지: P1 그래프>텍스트 확정(규칙 A 튜닝식>고전은 바닥 아래로 종결). P3 v3 n=6 은 시드 99 부호반전으로 주지표 미확인. v4 n=6 은 짝지은 t 로 정확도 4지표 유의·G1 통과, 사전등록 바닥 방식은 n=6 에서 recall@50 만 통과. P2 규칙 C 통과(세 트랙 3시드 재현). P4·P5 는 9/19 저녁 예정.

---

## `experiment_8cells/` — 실험표 8칸 지표 원본 (2026-08-25)

`docs/RESULTS_실험표.md` 의 모든 표가 이 27개 JSON 에서 나왔습니다.
학습 21회 · 덤프 21회 · 실패 0건. **전부 시드 42 단일**입니다.

### 파일 이름 규칙

```
<모델>_<주소>_<트랙>.json

  모델    tg = TIGER          mg = MaskGR
  주소    text = 텍스트        T1 = G-SID 튜닝식      a05c = G-SID 고전
          crab_text = CRAB    crab_T1 / crab_a05c = CRAB + G-SID
  트랙    A = LOO (Split A)   W3/W4/W5 = Temporal (Split B)
          MaskGR 의 Temporal 은 과거 명명을 따라 _B_W4 / _B_W5
```

### 실험표 대응

| 칸 | 주소 | TIGER | MaskGR |
|---|---|---|---|
| ①④ | 텍스트 | `tg_text_A.json` | `mg_text_A.json` |
| ③⑥ | G-SID 튜닝식 | `tg_T1_A.json` | `mg_T1_A.json` |
| ③⑥ | G-SID 고전 | `tg_a05c_A.json` | `mg_a05c_A.json` |
| ②⑤ | CRAB | `tg_crab_text_A.json` | `mg_crab_text_A.json` |
| ⑧⑦ | CRAB + G 튜닝식 | `tg_crab_T1_A.json` | `mg_crab_T1_A.json` |
| ⑧⑦ | CRAB + G 고전 | `tg_crab_a05c_A.json` | `mg_crab_a05c_A.json` |

Temporal 은 `tg_{text,T1,a05c}_{W3,W4,W5}.json` (TIGER, 9개) 와
`mg_{text,T1,a05c}_B_{W4,W5}.json` (MaskGR, 6개).

> **MaskGR 에 W3 이 없는 것은 사전 배제입니다.** 테스트 유저 11,358명 중 6,441명(56.7%)이
> 그 윈도우에 학습 이력이 없어 채점 유저가 4,917명뿐이고, TIGER 로 재 본 결과 조건 간
> 최대 차이가 전부 노이즈 바닥 아래였습니다. TIGER × W3 수치는 그대로 보존돼 있습니다.

### 각 파일 안에

`{라벨: {지표..., "_source": {...}}}` 형식입니다. `compare_table.py` 가 그대로 읽습니다.

지표는 @10/@20/@50 이 모두 들어 있고, 보고용 컷오프는 **@10**(콜드축만 @50)입니다.
다양성 축은 별도 파일이 아니라 같은 JSON 안에 있습니다 — `aplt@10` · `coverage@10` ·
`exposure_gini@10` · `tail_exposure@10`.

`_source` 블록에 **추적·검증에 필요한 것이 다 있습니다**:

```json
"_source": {
  "pred_file": "...",  "sid_file": "...",  "split_file": "...",  "window": "W5",
  "n_invalid_sid": 0,  "invalid_sid_rate": 0.0,      ← SID 조회 실패율. 1.0 이면 좌표계 불일치
  "label_match_rate": 1.0,                            ← 정답 SID 전건 대조 (LOO 만)
  "n_slot_per_user": 200.0, "truncated_at": 200.0
}
```

### 재현

덤프 pkl 은 용량 때문에 레포에 없습니다(`.gitignore` 의 `*.pkl`). 서버
`~/recsys/{tiger_out,maskgr_out}/<태그>/eval_dump*/test_predictions_rank0.pkl` 에 있고,
체크포인트가 남아 있어 `tiger_eval_dump.sh` / `maskgr_eval_dump.sh` 로 다시 뜰 수 있습니다
(TIGER 1~2분, MaskGR 22분~1시간).

```bash
# TIGER 덤프는 그대로
python code/tiger_to_eval.py --pred <덤프.pkl> --sid <sid.pt> \
       --split Beauty_split_A.pkl --label <라벨> --out results/<라벨>.json
# Temporal 은 --window W5 추가 (Split B 로 읽는다)

# ★ MaskGR 덤프는 반드시 좌표계를 먼저 되돌린다 — 안 하면 전 지표가 0 이다
python code/maskgr_to_tiger.py <덤프.pkl> <변환본.pkl>                            # 일반 주소
python code/maskgr_to_tiger.py <덤프.pkl> <변환본.pkl> --stride 307 --width 306   # CRAB 주소
```

자세한 내용과 판정(무엇을 주장할 수 있고 없는지)은 `docs/RESULTS_실험표.md` 참고.

---

## `gsid_param_sweep/` — β₀ 스윕 + 다중시드 검증 원본 (2026-08-28)

`docs/RESULTS_실험표.md` 실험표 ⑤의 원본입니다. 전부 **재전처리(8/22) 이후의
신규 Split A/B** 로 채점 (COLD 분모 6,197 — 구 실험표 ①~④의 6,597 과 다름).
채점 하네스: 서버 `work/eval/tiger_to_eval.py`, 덤프는 전부 TOPK=200.

### 파일 명명

- `ns_<조건>_s<시드>.json` — LOO (Split A)
- `nsB_<조건>_<윈도우>[_s7].json` — Temporal (Split B, W4/W5). 시드 표기 없으면 42.

| 조건 코드 | 의미 | SID |
|---|---|---|
| `base` / `text` | baseline 텍스트 SID | `sid_out/baseline` |
| `T1` | 튜닝식 β₀0.5 (τ없음·κ0) | `sid_out/sid_T1_noTau_k0_b05` |
| `b07` | 튜닝식 β₀0.7 | `sid_out/sid_noTau_k0_b07` |
| `b10` | **튜닝식 β₀1.0 — 승자** | `sid_out/sid_noTau_k0_b10` |
| `a05c` | 고전 블렌딩 α0.5 (중심화) | `sid_out/gsid_a05_centered_nahye` |
| `a07` | 고전 블렌딩 α0.7 (중심화) | `sid_out/sid_blend_centered_a07` |

주의: 기존 조건(base/T1/a05c 시드42·7 등)도 **기존 덤프를 신규 스플릿으로 재채점**한
것이라, `results/experiment_8cells/` 의 같은 조건 수치와 다릅니다 (특히 COLD·APLT).
W5 는 tgfix 레시피(8k 스텝) 기준으로 통일했습니다.
