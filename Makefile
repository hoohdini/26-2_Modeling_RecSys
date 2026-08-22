# 자주 쓰는 명령 모음. 기존 스크립트를 그대로 부르기만 합니다 —
# 결과가 달라지는 일은 없습니다.
#
#   make help
#
# PY 를 바꿔 다른 인터프리터를 쓸 수 있습니다:
#   make table PY=/path/to/python

PY      ?= python
SPLIT_A ?= Beauty_split_A.pkl
SPLIT_B ?= Beauty_split_B.pkl
SID     ?= sid/L4/sid_tensor.pt
K       ?= 10

.PHONY: help table baselines tailsens tfrecord-a tfrecord-b eval eval-temporal check

help:  ## 이 목록
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  변수:  PY=$(PY)  SID=$(SID)  K=$(K)"
	@echo "  예:    make eval PRED=tiger_runs/clip_L4/eval_dump_k50/test_predictions_rank0.pkl LABEL=clip_L4"

check:  ## 의존성이 갖춰졌는지 확인 (아무것도 실행하지 않음)
	@$(PY) -c "import sys; print('python', sys.version.split()[0])"
	@$(PY) -c "import torch, numpy, scipy; print('  torch', torch.__version__, '/ numpy', numpy.__version__, '/ scipy', scipy.__version__)"
	@$(PY) -c "import tensorflow" 2>/dev/null && echo "  tensorflow OK (TFRecord 변환 가능)" || echo "  tensorflow 없음 — to_grid_*.py 는 못 돌립니다"
	@test -f $(SPLIT_A) && echo "  $(SPLIT_A) OK" || echo "  $(SPLIT_A) 없음 ★ 대부분의 스크립트가 이걸 기본값으로 씁니다"
	@test -f $(SID) && echo "  $(SID) OK" || echo "  $(SID) 없음"

# 비교표에 들어가는 결과 파일은 고정입니다. results/*.json 을 통째로 넣으면
# 후보 50개 풀 곡선(clip_L4_lam*)과 검증용 변형까지 섞여 커밋된 표와 행 구성이 달라집니다.
# 이 목록은 docs/TRACK_C_보고서.md 9절의 재현 명령과 같습니다.
TABLE_INPUTS = results/baseline_splitA_loo.json results/tiger_clip_L4_loo.json \
               results/pareto_clip_L4_k200_k20.json results/pareto_clip_L4_exposure_k20.json

table:  ## 비교표 다시 만들기 — 커밋된 표와 동일하게 (GPU 불필요, 몇 초)
	$(PY) code/compare_table.py $(TABLE_INPUTS) --k $(K) --out docs/TRACK_C_비교표.md

baselines:  ## 고전 베이스라인 재계산 (Split A 위에서)
	$(PY) code/baselines_a.py

tailsens:  ## 롱테일 경계(tail_frac) 민감도
	$(PY) code/tail_sensitivity.py

tfrecord-a:  ## Split A → TFRecord (LOO 트랙 입력) · tensorflow 필요
	$(PY) code/to_grid_v2.py $(SPLIT_A) grid_data/beauty_A

tfrecord-b:  ## Split B → TFRecord (Temporal 트랙 입력) · tensorflow 필요
	$(PY) code/to_grid_b.py $(SPLIT_B) grid_data/beauty_B

eval:  ## 덤프 → 지표 (LOO + 다양성).  PRED= LABEL= 필요
	@test -n "$(PRED)" || { echo "PRED= 가 필요합니다 (덤프 pkl 경로)"; exit 1; }
	@test -n "$(LABEL)" || { echo "LABEL= 이 필요합니다"; exit 1; }
	$(PY) code/tiger_to_eval.py --pred $(PRED) --sid $(SID) --label $(LABEL)

eval-temporal:  ## 덤프 → 지표 (Temporal).  PRED= LABEL= WINDOW= 필요
	@test -n "$(PRED)" || { echo "PRED= 가 필요합니다"; exit 1; }
	@test -n "$(WINDOW)" || { echo "WINDOW= 가 필요합니다 (예: W5)"; exit 1; }
	$(PY) code/tiger_to_eval.py --pred $(PRED) --sid $(SID) --label $(LABEL) \
	      --split $(SPLIT_B) --window $(WINDOW)
