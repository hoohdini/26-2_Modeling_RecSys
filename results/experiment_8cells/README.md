# 실험표 8칸 지표 원본 (2026-08-25)

`docs/RESULTS_실험표.md` 의 모든 표가 이 27개 JSON 에서 나왔습니다.
학습 21회 · 덤프 21회 · 실패 0건. **전부 시드 42 단일**입니다.

## 파일 이름 규칙

```
<모델>_<주소>_<트랙>.json

  모델    tg = TIGER          mg = MaskGR
  주소    text = 텍스트        T1 = G-SID 튜닝식      a05c = G-SID 고전
          crab_text = CRAB    crab_T1 / crab_a05c = CRAB + G-SID
  트랙    A = LOO (Split A)   W3/W4/W5 = Temporal (Split B)
          MaskGR 의 Temporal 은 과거 명명을 따라 _B_W4 / _B_W5
```

## 실험표 대응

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

## 각 파일 안에

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

## 재현

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
