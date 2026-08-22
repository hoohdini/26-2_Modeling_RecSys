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
