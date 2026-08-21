# 인수인계 — 내 실행에 다양성·롱테일 지표 붙이기 (트랙 C 하네스)

> 대상: 그래프 SID(`gsid_a01`, `gsid_a03`) 등 **다른 SID로 TIGER를 돌린 사람**
> 상세 근거: `docs/TRACK_C_보고서.md` · 결과 표: `docs/TRACK_C_비교표.md`
> **전 지표 컷오프는 @10** — 정확도 파트(Recall@5/@10)와 맞춘 값입니다. 콜드 축만 @50.

정확도(Recall@5/@10)만으로는 "생성형 추천이 비인기 상품에 강하다"는 우리 주장을 검증할 수
없습니다. 이 하네스는 **같은 추천 결과에 다양성·롱테일 노출·콜드 지표를 얹어 줍니다.**
SID만 바꿔서 그대로 재사용할 수 있게 만들어 뒀습니다. 코드 수정은 필요 없습니다.

---

## 0. 3분 요약 — 명령 세 줄

```bash
# ① 서버: 정답이 가려진 추천 결과를 덤프 (약 1~4분)
sbatch --export=ALL,TAG=<내TAG>,SID=<내 sid_tensor.pt 절대경로>,NH=<계층수>,TOPK=200,BATCH=32 \
       tiger_eval_dump.sh

# ② 로컬: 덤프를 받아 지표 계산
scp -P 37220 "dsl05@165.132.80.36:/data1/dsl05/recsys/tiger_out/<내TAG>/eval_dump_k200/*.pkl" \
    tiger_runs/<내TAG>/eval_dump_k200/
python code/tiger_to_eval.py --pred tiger_runs/<내TAG>/eval_dump_k200/test_predictions_rank0.pkl \
                             --sid sid/<내SID>/L4/sid_tensor.pt \
                             --label <내TAG> --out results/tiger_<내TAG>_loo.json

# ③ 텍스트 SID·고전 베이스라인과 한 표에 놓기
python code/compare_table.py results/baseline_splitA_loo.json results/tiger_clip_L4_loo.json \
                             results/tiger_<내TAG>_loo.json --k 10
```

`--k` 기본값이 10이라 생략해도 됩니다.

---

## 1. 준비물

| 필요한 것 | 어디에 |
|---|---|
| `Beauty_split_A.pkl` | **이 저장소 루트에 들어 있습니다** (10MB, `.gitignore` 예외) |
| 내 SID 텐서 | `sid/gsid_a01/L4/sid_tensor.pt` 등 — 이미 저장소에 있습니다 |
| 내 체크포인트 | 서버 `tiger_out/<TAG>/checkpoints/` |
| torch / numpy / scipy | 로컬 파이썬 환경 |

> ⚠️ **`Beauty_split_A.pkl`을 직접 다시 만들지 마십시오.** 전처리를 다시 돌리면 아이템
> 인기도가 달라지고, 그러면 **롱테일 집합과 콜드 버킷 경계가 다른 아이템 위에 그어집니다.**
> 실제로 구 전처리 기준 베이스라인은 zero-shot 타깃이 138개, 현행은 162개로 다릅니다
> (보고서 2-1절). 다른 split 위에서 잰 숫자는 같은 표에 올릴 수 없습니다.

---

## 2. 서버 덤프 — 그래프 SID일 때 바꿔야 하는 것

`code/server/tiger_eval_dump.sh`의 기본 `SID`는 **텍스트 SID 경로**입니다. 반드시 덮어쓰십시오.

| 변수 | 의미 | 그래프 SID일 때 |
|---|---|---|
| `TAG` | `tiger_out/$TAG` 폴더 이름 | 내 실행 태그 |
| `SID` | SID 텐서 절대경로 | **내 gsid 텐서로 교체 필수** |
| `NH` | `num_hierarchies` = SID 텐서의 **행 수** | 4단계 SID면 **5** (충돌 구분자 포함) |
| `TOPK` | 빔 폭 = 유저당 후보 수 | **200** (다이얼 곡선을 그리려면 필요) |

`NH`가 틀리면 `vocab_size`가 어긋나 체크포인트가 안 실립니다. 학습 때 쓴 값과 같아야 합니다.

### 덤프가 옳은지 먼저 검증하십시오 (권장 순서)

`TOPK=10`으로 한 번 돌리면 학습 때와 완전히 같은 설정입니다. 이때 나온 값이
`tiger_runs/<TAG>/metrics.csv`의 `test/recall@5`, `test/recall@10`과 **소수점 끝까지
일치해야** 덤프 경로가 평가 경로를 바꾸지 않았다는 것이 증명됩니다.

```bash
sbatch --export=ALL,TAG=<내TAG>,SID=<...>,NH=<...>,TOPK=10 tiger_eval_dump.sh
# 받아서
python code/tiger_to_eval.py --pred .../eval_dump_k10/test_predictions_rank0.pkl \
                             --sid sid/<내SID>/L4/sid_tensor.pt --label <내TAG>_k10 \
                             --out results/tiger_<내TAG>_k10_loo.json
```

텍스트 SID에서는 `0.04444842040538788`까지 비트 단위로 일치했습니다.

---

## 3. 절대 하면 안 되는 것 두 가지

두 경로 모두 **에러 없이 조용히 틀린 숫자를 줍니다.** 실제로 우리가 당했습니다.

1. **`src.inference`(`predict_step`) 산출물로 평가하지 마십시오.**
   전체 시퀀스(정답 포함)를 넣고 그 다음을 생성하는 실서비스 경로라 **정답을 가리지
   않습니다.** Recall@10이 0.0660 → 0.0961(**+46%**)로 부풀려집니다.
   기존 `tiger_runs/*/infer/merged_predictions.pkl`은 전부 이 문제가 있습니다.

2. **`src.train train=False`로 평가하지 마십시오.**
   `src/train.py`의 test 블록은 `ckpt_path`를 **설정에서 읽지 않고** ModelCheckpoint
   콜백의 `best_model_path`만 봅니다. 학습을 건너뛰면 그 값이 비어 있어
   **초기화 직후의 랜덤 가중치로 평가합니다.** 경고 한 줄만 나오고 진행됩니다.
   → `src/eval_dump.py`(= `tiger_eval_dump.sh`)를 쓰십시오.

---

## 4. 지표 읽는 법

| 지표 | 뜻 | 방향 |
|---|---|---|
| **`APLT@10`** | 추천 슬롯 중 롱테일(학습 인기 하위 50%, 6,051개) 상품의 비율. **다양성 대표축** | 높을수록 좋음 |
| `coverage@10` | 전체 12,101개 중 한 번이라도 추천된 상품 비율 | 높을수록 좋음 |
| `exposure_gini@10` | 노출 집중도 | 낮을수록 고르게 퍼짐 |
| `COLD_recall@50` | 학습 등장 5회 이하 상품이 정답일 때의 Recall@50 | 높을수록 좋음 |

**`APLT`**는 Average Percentage of Long Tail items의 약자로 인기편향 문헌의 표준 명칭입니다
(Abdollahpouri et al.). 코드 내부 키는 `tail_exposure@10`인데 **같은 값**이라 비교표에서는
한 칼럼으로만 싣습니다. 발표·논문에는 `APLT`로 쓰십시오.

> **콜드 축만 컷오프가 @50입니다.** 의도한 것입니다 — 콜드 아이템은 상위 10개에 거의 안
> 잡혀서 @10으로 재면 대부분 0이 됩니다. 다양성(노출)과 콜드(정확도)는 다른 축입니다.

---

## 5. 비교 대상 — 텍스트 SID 기준선 (@10)

내 결과를 여기에 놓고 보면 됩니다.

| 모델 | Recall@10 | NDCG@10 | **APLT@10** | Coverage@10 | COLD-R@50 |
|---|---|---|---|---|---|
| TIGER `clip_L4` (텍스트 SID) | 0.0692 | 0.0384 | **0.0497** | 0.3887 | 0.0429 |
| EASE_R | 0.0509 | 0.0265 | 0.0567 | 0.8370 | 0.0355 |
| ItemKNN | 0.0425 | 0.0221 | **0.5952** | 0.9577 | 0.0702 |
| MostPopular | 0.0121 | 0.0056 | 0.0000 | 0.0012 | 0.0000 |
| Random | 0.0007 | 0.0003 | 0.5007 | 1.0000 | 0.0035 |

**지금의 TIGER는 EASE_R보다 정확한데 롱테일 노출은 오히려 12% 낮습니다.** 그래프 SID가
이 격차를 줄인다면 그 자체로 기여입니다. 늘린다면 그것도 보고할 가치가 있습니다.

---

## 6. ★ 차이가 진짜인지 판정하는 자 — 노이즈 바닥

**이게 이 문서에서 가장 중요한 부분입니다.**

같은 레시피에 **시드만 바꾼** 두 실행(`clip_L4` vs `clip_L4_seed7`)의 간격을 재 뒀습니다.

```
노이즈 바닥:  NDCG@10  0.0016      APLT@10  0.0176
```

**이보다 작은 차이는 시드 하나 바꾼 것과 구별되지 않습니다.** 그래프 SID가 텍스트 SID보다
NDCG@10에서 0.001 높게 나왔다면 그건 SID의 효과가 아닙니다.

> ⚠️ **"유의하다(p<0.05)"와 "노이즈 바닥보다 크다"는 다른 질문입니다.**
> paired bootstrap은 *이 두 모델이 이 유저들 위에서 정말 다른가*를 잽니다.
> 그런데 **시드만 바꾼 seed7도 "유의"하게 나옵니다.** 유의성만으로는 개입의 효과라고
> 말할 수 없습니다. 반드시 이 자를 같이 대십시오.

### 그래프 SID를 텍스트 SID와 비교할 때 — `pareto_compare.py`를 쓰지 마십시오

`pareto_compare.py`는 **`--sid`를 하나만 받아 모든 실행에 똑같이 적용합니다.** SID 텐서가
다른 실행을 같이 넣으면 한쪽이 엉뚱한 테이블로 역매핑돼 **에러 없이 쓰레기 값이 나옵니다.**
(무효 SID 비율이 비정상적으로 높게 찍히는 것으로 알아챌 수는 있습니다.)

같은 SID를 쓰는 실행끼리(예: 내 gsid의 시드 변형끼리)는 그대로 쓸 수 있습니다:

```bash
python code/pareto_compare.py \
  --ref "<내TAG>=tiger_runs/<내TAG>/eval_dump_k200/test_predictions_rank0.pkl" \
  --run "<내TAG>_seed7=tiger_runs/<내TAG>_seed7/eval_dump_k200/test_predictions_rank0.pkl" \
  --sid sid/<내SID>/L4/sid_tensor.pt \
  --k 10 --lams 0,0.2,0.5,1,2,5
```

**SID가 다른 실행끼리 비교할 때는** 각자 `tiger_to_eval.py --sid <자기 것>`으로 JSON을
따로 만든 뒤 `compare_table.py`로 합치십시오(0절 방식). 노이즈 바닥은 위에 적힌 값
(NDCG@10 0.0016 · APLT@10 0.0176)을 그대로 자로 쓰면 됩니다 — 텍스트 SID에서 이미 재 둔
값이고, 학습 레시피가 같으므로 그대로 적용됩니다.

> 하네스를 고쳐 실행별 SID를 받게 만들 수도 있습니다. 필요하면 말씀해 주세요.

---

## 7. 실험 간 반드시 고정할 것

하나라도 다르면 성능 차이가 **SID 때문인지 설정 때문인지 구분이 안 됩니다.**

| 항목 | 값 | 왜 |
|---|---|---|
| `top_k_for_generation` (=`TOPK`) | 실험 간 동일 | 빔 폭이라 넓히면 **상위 10개도 달라집니다** (10 → 50에서 Recall@10 +4.9%) |
| `should_check_prefix` | `false` | 지금까지 보고한 지표가 전부 이 기준. 켜면 전 지표 재측정 (게다가 GRID 버그로 실행 불가 — 보고서 6-6절) |
| `tail_frac` | `0.5` | 롱테일 경계. 이 데이터의 실제 80/20 지점이 상위 45.5%라 0.5가 맞습니다 |
| split | `Beauty_split_A.pkl` | 1절 참고 |

---

## 8. 곡선까지 그리고 싶다면

TIGER는 빔서치라 점 하나만 찍힙니다. **GPU를 다시 쓸 필요 없이** 덤프에 들어 있는 후보를
인기도 페널티로 재정렬하면 파레토 곡선 전체가 나옵니다.

```bash
python code/pareto_dial.py --pred tiger_runs/<내TAG>/eval_dump_k200/test_predictions_rank0.pkl \
                           --sid sid/<내SID>/L4/sid_tensor.pt \
                           --label <내TAG> --k 20 --lams 0,0.2,0.5,1,2,3,5,8,12
```

`--k 20`은 그대로 두십시오. 지표는 @10·@20·@50이 전부 나오고, `--k`는 노출 인지 모드의
슬롯 기준(`k_slot`)에만 쓰입니다. 기존 곡선과 비교하려면 같은 값이어야 합니다.

> ⚠️ **이건 재정렬 다이얼이지 생성 다이얼이 아닙니다.** 후보 풀 밖의 상품은 λ를 아무리
> 키워도 추천될 수 없습니다. MaskGR의 확산 다이얼과 **같은 종류의 기여로 주장하면 안 됩니다.**
> "AR 모델에서 재정렬만으로 여기까지는 간다"는 **참조선**으로 쓰는 것이 정직합니다.

---

## 9. 파일 목록

```
code/evaluate.py          평가 하네스 (지표 정의는 전부 여기)
code/baselines.py         고전 베이스라인 구현
code/baselines_a.py       Split A 위에서 베이스라인 재계산
code/tiger_to_eval.py     SID → item_id 역매핑 + 지표 계산      ← 보통 여기만 쓰면 됨
code/pareto_dial.py       재정렬 다이얼 → 파레토 곡선
code/pareto_compare.py    체크포인트 간 곡선 비교 + 노이즈 바닥
code/reachability.py      도달가능성 (후보에 아예 안 오르는 상품)
code/significance.py      paired bootstrap
code/tail_sensitivity.py  롱테일 경계 민감도
code/compare_table.py     여러 결과 JSON → 비교표 + 파레토 판정

code/server/eval_dump.py             학습 없이 test 만 도는 엔트리포인트
code/server/prediction_dumper.py     eval_step 을 런타임에 가로채는 콜백
code/server/tiger_eval_dump.sh       sbatch 스크립트

results/*.json            텍스트 SID·베이스라인 결과 (비교 대상)
Beauty_split_A.pkl        평가 기준 split
```

**서버 GRID의 기존 파일은 한 줄도 고치지 않았습니다.** 신규 파일 3개만 추가하고, 테스트
시작 시점에 두 지점을 런타임에 감쌌다가 끝나면 원상복구합니다. 팀원의 학습·평가에
영향이 없습니다.

---

## 10. 막히면

- 지표 정의가 궁금하면 → `code/evaluate.py` (전부 주석 있음)
- 왜 이렇게 쟀는지 → `docs/TRACK_C_보고서.md`
- 결과 표 → `docs/TRACK_C_비교표.md`, 그림은 `docs/TRACK_C_아티팩트.html`
- 프로토콜 확정안(분할·경계·대표 지표) → 보고서 7절
