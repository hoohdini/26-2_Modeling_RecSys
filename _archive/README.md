# _archive — 지금은 안 쓰지만 지우지 않은 것들

**삭제하지 않았습니다.** 기록으로 남길 가치가 있거나, 다시 필요해질 수 있어서 여기 모아
뒀을 뿐입니다. 되돌리려면 `git mv` 로 원래 자리에 옮기면 됩니다.

```bash
git mv _archive/code_server/tiger_infer.sh code/server/     # 예시
```

---

## ⚠️ 여기 있는 이유가 "위험해서"인 파일

### `code_server/tiger_infer.sh`
GRID 의 `src.inference`(**`predict_step`**) 경로로 추론하는 스크립트입니다.
**이 경로는 정답을 입력에서 가리지 않습니다.** 전체 시퀀스를 넣고 "그 다음"을 생성하는
실서비스용 경로라, 그 산출물로 평가하면 지표가 부풀려집니다.

```
Recall@10   학습 로그(정답 가림) 0.0660   →   predict_step 산출물 0.0961  (+46%)
```

기존 `tiger_runs/*/infer/merged_predictions.pkl` 3개는 전부 이 문제가 있습니다.
**평가에 쓰면 안 됩니다.** 올바른 평가 경로는 `code/server/tiger_eval_dump.sh`
(= `eval_step` 을 가로채는 덤프)입니다. 근거는 `docs/TRACK_C_보고서.md` 1절.

> 실서비스형 추론이 실제로 필요해지면 그때 꺼내 쓰면 됩니다. 지금 문제는
> "평가에 쓰기 쉬운 자리에 있다"는 것이라 옮겼습니다.

### `code_server/inspect_tiger_predictions.py` · `code/verify_tiger_predictions.py`
위 누출을 **진단하는 데 쓴 도구**입니다. 진단이 끝나 더는 안 쓰지만, 무엇을 어떻게
확인했는지가 남아 있어야 나중에 같은 의심이 들 때 다시 볼 수 있습니다.

---

## 역할이 끝난 파일

### `code_server/merge_preds.py`
GRID 의 `inference_utils.py` 버그(단일 GPU 에서 `barrier()` 로 병합 실패) 때문에
pkl 조각을 손으로 병합하던 스크립트입니다. **그 버그를 패치한 뒤로는 필요 없습니다.**
(`docs/TOKENIZE_EMBED_SID_REPORT.md` 227행에 "패치 후엔 불필요"로 이미 적혀 있습니다.)
