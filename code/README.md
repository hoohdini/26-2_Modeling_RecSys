# code/ — 무엇이 무엇에 기대는가

**`code/` 는 평평하게 유지합니다.** 스크립트들이 `sys.path.insert(0, dirname(__file__))` 로
형제 모듈을 직접 import 하기 때문에, 하위 폴더로 옮기면 import 가 깨집니다.

---

## 의존 관계

```
evaluate.py                         ← 뿌리. 지표 정의가 전부 여기 있다
  ├── baselines.py                  고전 베이스라인 구현 (EASE_R / ItemKNN / MostPopular / Random)
  │     ├── baselines_a.py          Split A 위에서 베이스라인 재계산
  │     ├── tail_sensitivity.py     롱테일 경계(tail_frac) 민감도
  │     └── significance.py         paired bootstrap
  ├── tiger_to_eval.py              SID → item_id 역매핑 + 지표          ★ 가장 많이 쓴다
  │     ├── pareto_dial.py          재정렬 다이얼 → 파레토 곡선
  │     │     └── pareto_compare.py 체크포인트 간 곡선 비교 + 노이즈 바닥
  │     ├── reachability.py         도달가능성 (후보에 아예 안 오르는 상품)
  │     └── significance.py
  └── compare_table.py              여러 결과 JSON → 비교표 + 파레토 판정

to_grid_v2.py                       Split A → TFRecord
  └── to_grid_b.py                  Split B 윈도우 → TFRecord (Temporal 트랙)

export_csv.py / export_sid_csv.py / export_tiger_metrics.py   → csv_export/ 생성 (독립)
splitb_protocol_analysis.py                                   Split B 진단 (독립)
run_maskgr.sh                                                 MaskGR 러너 (server/ 스크립트 호출)
```

**화살표는 "import 한다"** 는 뜻입니다. `evaluate.py` 를 고치면 그 아래가 전부 영향받습니다.

---

## 어떤 파이썬으로 돌리나

| 필요한 것 | 쓰는 스크립트 |
|---|---|
| `torch`, `numpy`, `scipy` | 평가 하네스 전부 (`evaluate` 계열) |
| `tensorflow` | `to_grid_v2.py`, `to_grid_b.py` 만 |
| `pandas` | `export_*.py` 만 |

`.pt` 텐서와 GRID/MaskGR 덤프 pkl 은 **torch 가 있어야 열립니다.** 없는 환경에서 돌리면
`ModuleNotFoundError: torch` 로 죽습니다. `requirements.txt` 참고.

---

## 실행 순서 (한 실험을 처음부터)

```bash
# 1. 분할 → TFRecord  (tensorflow 필요)
python code/to_grid_v2.py Beauty_split_A.pkl grid_data/beauty_A          # LOO
python code/to_grid_b.py  Beauty_split_B.pkl grid_data/beauty_B          # Temporal

# 2. 서버: 임베딩 → SID → 학습 → 덤프   (code/server/*.sh — 각 문서 참고)

# 3. 덤프 → 지표
python code/tiger_to_eval.py --pred <덤프.pkl> --sid sid/L4/sid_tensor.pt --label <이름>
python code/tiger_to_eval.py --pred <덤프.pkl> --sid ... --split Beauty_split_B.pkl --window W5

# 4. 표로
python code/compare_table.py results/*.json --k 10 --out docs/TRACK_C_비교표.md
```

각 스크립트는 `--help` 가 있습니다. 자세한 절차는
`docs/HANDOFF_트랙C_다양성평가.md` (다른 SID 로 실험할 때),
`docs/HANDOFF_MaskGR.md` (MaskGR).

---

## 손댈 때 지켜야 하는 것

- **`evaluate.py` 의 지표 정의를 바꾸면 지금까지의 모든 수치가 무효가 됩니다.**
  버킷 경계(`BUCKETS`), `tail_frac=0.5`, 컷오프 규약이 전부 여기 박혀 있습니다.
- 기본 경로가 `ROOT/Beauty_split_A.pkl` 입니다 (`ROOT` = 저장소 루트).
  **split 파일을 옮기면 모든 스크립트의 기본값이 깨집니다.**
- 새 지표를 추가할 때는 **기존 키를 바꾸지 말고 더하십시오.** 기존 결과 JSON 과
  `compare_table.py` 가 키 이름으로 읽습니다.

---

## server/ — 서버에서 도는 것

```
embed_beauty.sh          임베딩 추출 (flan-t5-xl)
sid_beauty.sh            SID 생성 (RQ-KMeans)
tiger_beauty.sh          TIGER 학습
tiger_eval_dump.sh       TIGER 평가 덤프          ★ 평가는 반드시 이 경로
eval_dump.py             학습 없이 test 만 도는 엔트리포인트 (GRID 에 복사)
prediction_dumper.py     eval_step 을 가로채는 콜백 (GRID 에 복사)

maskgr_train.sh          MaskGR 학습
maskgr_eval_dump.sh      MaskGR 평가 덤프
maskgr_eval_dump.py      MaskGR 엔트리포인트 (MaskGR 에 복사)
maskgr_dumper.py         MaskGR 덤프 콜백 (MaskGR 에 복사)

server_setup.sh          환경 구축
prefetch_model.sh        모델 캐시 미리 받기
verify_embed.py          임베딩 검증
```

`*_dumper.py` 와 `*_eval_dump.py` 는 **업스트림 저장소에 복사해 넣는 신규 파일**입니다.
기존 파일은 한 줄도 고치지 않습니다 — 팀 공유 저장소이기 때문입니다.

---

## 반복해서 당한 함정 8가지 (서버에서 돌리기 전에)

1. **`src.inference`(`predict_step`) 산출물로 평가하지 마십시오.** 정답을 가리지 않아 Recall@10 이 0.0660 → 0.0961(+46%)로 부풀려집니다. 올바른 경로는 `code/server/tiger_eval_dump.sh`. 관련 스크립트는 `_archive/` 에 있습니다.
2. **`src.train train=False` 로 평가하지 마십시오.** `ckpt_path` 를 설정에서 읽지 않아 랜덤 가중치로 평가하고 경고 한 줄만 남깁니다. `src/eval_dump.py` 를 쓰십시오.
3. **TFRecord 는 split 당 1파일**이라 `num_workers=0` + `timeout=0` + `persistent_workers=false` 를 세트로 줘야 합니다.
4. **`num_hierarchies` = SID 텐서의 행 수**(4단계 → 5), `vocab_size` = NH × `WIDTH`. `WIDTH` 는 SID 마다 다릅니다 — 텍스트·G-SID 256(VOCAB 1280), CRAB 306(VOCAB 1530). 틀리면 즉시 죽습니다.
5. **검증 표본을 셔플하지 않으면** 이력이 긴 유저만 뽑혀 검증 지표가 낙관적으로 나옵니다. `VAL_SHUFFLE=true`, 전수 검증 `LIMIT_VAL=1.0`.
6. **MaskGR 에서는 `+trainer.…` 가 아니라 `++trainer.…`.** 실험 설정이 `gradient_clip_val`·`limit_val_batches` 를 이미 정의하고 있어 `+` 는 "Could not append to config" 로 죽습니다. GRID 는 `+` 가 맞습니다.
7. **`limit_val_batches` 에 분수를 주지 마십시오.** `IterableDataset` 라 `1.0` 또는 정수만 받습니다.
8. **MaskGR 은 실패해도 3번 재시도합니다.** 설정 오류는 CPU 스모크(`trainer=cpu` + `++trainer.precision=32`)로 먼저 걸러내십시오.

Temporal 트랙 추가: Split B 의 내부 검증 정답은 자리표시자라 patience 8 조기종료가 13k~24k 스텝에서 이르게 걸립니다(최적 6k~16k). 레시피의 일부이니 바꾸지 말고 보고서에 멈춘 스텝을 적으십시오. 접두사 제약 덤프(`PREFIX=1`)는 일반 덤프(4분)보다 훨씬 오래(30분) 걸립니다.
