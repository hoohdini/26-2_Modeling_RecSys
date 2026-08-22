# 재전처리(2026-08-22)가 기존 결과에 미치는 영향

> `preprocessing/repreprocess.py` 산출물로 갈아끼웠을 때 무엇이 바뀌고 무엇이 안 바뀌는지
> **실측**한 기록입니다. 원인과 설계 근거는 `preprocessing/전처리_수정문서.md`.

---

## 결론 세 줄

1. **정확도 지표는 한 자리도 안 바뀝니다.** `loo_test` 가 22,363명 전원 동일하기 때문입니다.
2. **다양성 지표만 미세하게 움직입니다.** 롱테일 집합의 기준인 학습 등장 횟수가 달라져서인데,
   이동폭이 **노이즈 바닥의 7%** 수준이라 트랙 C 결론은 그대로 섭니다.
3. **`item_text` 는 99.1% 의 아이템에서 바뀌었습니다.** 카테고리 단어 *순서*만 바뀐 것이지만
   임베딩은 순서에 민감하므로, **엄밀히 하려면 임베딩·SID 재추출이 필요합니다.**

---

## 1. 실측 — 같은 덤프를 구/신 Split A 로 각각 채점

모델은 그대로 두고(`clip_L4`, 옛 데이터로 학습된 것) 채점 기준만 바꿔 본 것입니다.
즉 **지표 정의가 얼마나 흔들리는지**만 재는 실험이고, 재학습 효과는 포함돼 있지 않습니다.

| 지표 | 구 Split A | 신 Split A | 차이 |
|---|---|---|---|
| Recall@10 | 0.0692 | 0.0692 | **0** |
| NDCG@10 | 0.0384 | 0.0384 | **0** |
| Coverage@10 | 0.3887 | 0.3887 | **0** |
| Gini@10 | 0.9117 | 0.9117 | **0** |
| **APLT@10** | 0.0497 | 0.0485 | **−0.0012** |
| COLD-R@50 | 0.0429 | 0.0416 | −0.0013 (대상 6,597 → 6,197) |

**APLT 이동폭 0.0012 는 노이즈 바닥(APLT@10 0.0176)의 7% 입니다.** 시드만 바꿔도
그보다 14배 크게 움직입니다. 지금까지의 트랙 C 서술은 유효합니다.

재현:

```bash
python code/tiger_to_eval.py \
  --pred tiger_runs/clip_L4/eval_dump_k50/test_predictions_rank0.pkl \
  --split Beauty_split_A.pkl --label clip_L4
```

---

## 2. 무엇이 바뀌었나 — 대조 결과

| 항목 | 결과 |
|---|---|
| `uid` / `iid` 매핑 | **완전히 동일** ✔ — 그래서 기존 SID 텐서가 그대로 유효합니다 |
| `loo_test` | **22,363명 전원 동일** ✔ — 정확도가 안 바뀌는 이유 |
| `loo_train` | 935명(4.2%)에서 다름. 옛 pkl 은 `max_seq_len=20` 으로 잘려 있었고 신 pkl 은 안 자릅니다 (최대 길이 20 → 202, 총 상호작용 138,622 → 153,776) |
| 유저 키 자료형 | `reviewerID` 문자열 → **정수 `user_id`** |
| `max_seq_len` 키 | 삭제 (대신 `user_seq` 추가) |
| `item_text` | **11,993 / 12,101 (99.1%)** 에서 다름 — 카테고리 단어 순서만 |

`item_text` 예시 (내용은 같고 순서만):

```
구:  ... Concealer: 15 Color Concealer. Makeup Beauty Concealers & Neutralizers Face
신:  ... Concealer: 15 Color Concealer. Beauty Concealers & Neutralizers Face Makeup
```

---

## 3. ★ 남은 일 — 임베딩·SID 재추출

`item_text` 가 바뀌었으므로 **지금 있는 임베딩과 SID 는 옛 텍스트로 만든 것**입니다.
엄밀히 하려면 다시 뽑아야 합니다.

| 단계 | 비용 | 지금 상태 |
|---|---|---|
| 임베딩 재추출 (`code/server/embed_beauty.sh`) | 약 4분 | 옛 텍스트 기준 |
| SID 재생성 (`code/server/sid_beauty.sh`) | 약 7분 | 옛 텍스트 기준 |
| TIGER 재학습 | 수 시간 | 옛 SID 로 학습됨 |
| 지표 재계산 | 즉시 | — |

**앞의 두 단계는 11분이라 미룰 이유가 없습니다.** 비싼 것은 재학습이고, 그건 최종 결과를
내기 전에 한 번 하면 됩니다.

> ⚠️ **재추출하면 SID 가 바뀌므로 `sid/L4/sid_tensor.pt` 도 바뀝니다.** 그러면
> 지금 있는 `results/*.json` 은 다른 SID 위에서 잰 값이 됩니다. 섞어서 비교하지 마십시오.
> 재추출본은 `sid/L4_v2/` 처럼 따로 두고 어느 쪽인지 라벨에 남기는 편이 안전합니다.

---

## 4. 옛 Split A 는 지웠는가

아닙니다. `_archive/data/Beauty_split_A_v1_maxseq20.pkl` 로 옮겨 뒀습니다.
지금 커밋된 `results/*.json` 은 전부 그 파일로 만든 값이라, 기존 수치를 재현하려면 이걸
써야 합니다.

```bash
python code/tiger_to_eval.py --pred <덤프> \
  --split _archive/data/Beauty_split_A_v1_maxseq20.pkl --label clip_L4_v1
```

---

## 5. 코드 호환

`evaluate.py` / `tiger_to_eval.py` / `to_grid_v2.py` / `to_grid_b.py` 는
**신·구 스키마를 모두 받습니다.** 자료형과 구조를 보고 자동으로 분기하므로
옛 pkl 로 만든 명령도 그대로 돕니다.

| 바뀐 것 | 대응 |
|---|---|
| 유저 키 str → int | `evaluate.split_key_type()` 이 판별, `remap_users(..., key_type)` 이 분기 |
| `windows` list → dict | `to_grid_b.iter_windows()` 가 둘 다 순회 |
| `train_user_seq` → `train_seq` | 두 키를 모두 찾아봄 |
| `cold_tiers` 가 윈도우 안 → 최상위 | 두 위치를 모두 찾아봄 |
