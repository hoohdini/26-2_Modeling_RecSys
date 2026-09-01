# GRID 패치

`GRID/` 는 vendored 사본(git 아님)이라 수정 내역을 여기 기록합니다.
서버 적용본: `/data1/dsl05/recsys/GRID/`

## 001 — 접두사 제약 디코딩 들여쓰기 버그

**파일**: `src/models/modules/semantic_id/tiger_generation_model.py:305`
**적용**: 2026-09-01 · 원본 백업 `*.bak_20260901`

`should_check_prefix=True` 로 켜면 첫 계층에서 마스크가 잘못 적용됩니다.

```python
if self.should_check_prefix:
    if generated_ids is None:
        valid_prefix_mask = self._check_valid_prefix(...)   # 1-D [num_emb]
        candidate_logits[:, ~valid_prefix_mask] = -inf      # 열 마스킹 — 올바름
    else:
        valid_prefix_mask = self._check_valid_prefix(...).reshape(-1, num_emb)
    candidate_logits[~valid_prefix_mask] = -inf   # ← else 밖 (버그)
```

마지막 줄이 두 분기 모두에서 실행됩니다. 첫 분기는 이미 열 마스킹을 끝냈는데,
1차원 마스크로 **행(빔/유저)** 을 다시 마스킹해 특정 유저의 로짓이 통째로 `-inf` 가 됩니다.

**수정**: 마지막 줄을 `else` 블록 안으로 4칸 들여씀.

```diff
-            candidate_logits[~valid_prefix_mask] = float("-inf")
+                candidate_logits[~valid_prefix_mask] = float("-inf")
```

**검증**: `invalid_sid_rate` 0.00954 → **0.00000**. 정확도 −1%, 롱테일 +8%.

⚠️ 공유 저장소 원본에는 반영하지 않았습니다. 팀 결정이 필요합니다.
근거와 측정치: [`docs/JOBS_결과정리.md`](../docs/JOBS_결과정리.md) §8
