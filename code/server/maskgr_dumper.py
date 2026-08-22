# -*- coding: utf-8 -*-
"""MaskGR 의 test 단계 생성 결과를 파일로 떨구는 Lightning 콜백.

GRID 쪽 prediction_dumper.py 와 하는 일이 같다. MaskGR 의
`DiscreteDiffusionModule` 이 GRID 와 **동일한 두 지점**을 갖고 있어서 그대로 옮겼다.

    eval_step(batch, loss_to_aggregate)              -> 배치에서 user_id
    self.evaluator(marginal_probs=, generated_ids=, labels=)  -> 생성 결과

MaskGR 특유의 주의점
--------------------
1) `beam_search_generation` 이 돌려주는 두 번째 값은 **빔 점수(누적 로그확률)**이지
   TIGER 의 marginal_probs 와 정확히 같은 물건이 아니다. 그래도 "후보 순위를 매기는
   점수"라는 역할은 같아서 `scores` 로 저장한다. **재정렬 다이얼에 쓸 때는 이 차이를
   반드시 기억할 것** — TIGER 곡선과 직접 겹쳐 그리면 안 된다.
2) `generated_ids` 는 (batch, num_candidates, num_hierarchies) 로 GRID 와 같은 모양이라
   `code/tiger_to_eval.py` 이하 트랙 C 하네스가 **수정 없이 그대로** 읽는다.
3) MaskGR 은 `src/inference.py` 가 저장소에 없다(Makefile 은 부르는데 파일이 없음).
   그래서 평가 경로는 `src/maskgr_eval_dump.py` 를 따로 둔다.

설치
----
  MaskGR/src/callbacks/__init__.py       (빈 파일)
  MaskGR/src/callbacks/maskgr_dumper.py  (이 파일)

사용 (hydra 오버라이드)
-----------------------
  +callbacks.pred_dump._target_=src.callbacks.maskgr_dumper.MaskGRPredictionDumper
  +callbacks.pred_dump.out_dir=/경로/eval_dump
"""
import logging
import os
import pickle

import torch
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


def _as_int_list(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        if x.dim() == 2:          # (batch, seq) — user_id 는 시퀀스로 패딩돼 들어온다
            x = x[:, 0]
        return [int(v) for v in x.reshape(-1).tolist()]
    return [int(v) if not isinstance(v, (str, bytes)) else v for v in x]


class _EvaluatorTap:
    """SIDRetrievalEvaluator 를 감싸 호출 인자를 가로채고 그대로 위임한다."""

    def __init__(self, inner, owner):
        self._inner = inner
        self._owner = owner

    def __call__(self, marginal_probs, generated_ids, labels, **kw):
        self._owner._capture(marginal_probs, generated_ids, labels)
        return self._inner(marginal_probs=marginal_probs,
                           generated_ids=generated_ids, labels=labels, **kw)

    def __getattr__(self, name):      # metrics, compute, reset ... 전부 원본으로
        return getattr(self._inner, name)


class MaskGRPredictionDumper(Callback):
    """test 단계의 생성 SID 를 out_dir/test_predictions_rank{r}.pkl 로 저장한다.

    레코드 한 건 (GRID 덤프와 동일 스키마):
      {"user_id": int, "semantic_ids": (C, H) int16, "scores": (C,) float32,
       "label_sid": (H,) int16}
    C = diffusion_config.num_candidates, H = num_hierarchies.

    추가로 실행 설정을 out_dir/dump_meta.json 에 남긴다. 확산 다이얼을 스윕할 때
    어떤 설정의 덤프인지 잃어버리면 곡선을 못 그린다.
    """

    def __init__(self, out_dir, filename="test_predictions", save_scores=True):
        super().__init__()
        self.out_dir = out_dir
        self.filename = filename
        self.save_scores = save_scores
        self.rows = []
        self._pending_users = None
        self._user_src = None
        self._orig_eval_step = None
        self._orig_evaluator = None
        self._n_batch = 0

    # ---------- 설치 / 해제 ----------
    def on_test_start(self, trainer, pl_module):
        self.rows, self._n_batch = [], 0
        os.makedirs(self.out_dir, exist_ok=True)

        self._orig_eval_step = pl_module.eval_step

        def tapped_eval_step(batch, loss_to_aggregate=None):
            self._pending_users = self._extract_users(batch[0])
            return self._orig_eval_step(batch, loss_to_aggregate)

        pl_module.eval_step = tapped_eval_step

        self._orig_evaluator = pl_module.evaluator
        pl_module.evaluator = _EvaluatorTap(self._orig_evaluator, self)

        self._write_meta(pl_module)
        log.info(f"[dumper] 설치 완료 -> {self.out_dir}")

    def on_test_end(self, trainer, pl_module):
        if self._orig_eval_step is not None:
            pl_module.eval_step = self._orig_eval_step
        if self._orig_evaluator is not None:
            pl_module.evaluator = self._orig_evaluator

        rank = getattr(trainer, "global_rank", 0) or 0
        path = os.path.join(self.out_dir, f"{self.filename}_rank{rank}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.rows, f)

        uniq = len({r["user_id"] for r in self.rows})
        log.info(f"[dumper] 배치 {self._n_batch:,} · 레코드 {len(self.rows):,} "
                 f"(고유 user {uniq:,}) · user_id 출처={self._user_src} -> {path}")
        if self._user_src == "counter":
            log.warning("[dumper] user_id 를 찾지 못해 일련번호를 썼다. "
                        "split 과 조인할 수 없으니 데이터로더 설정을 확인할 것.")

    # ---------- 내부 ----------
    def _write_meta(self, pl_module):
        """확산 설정을 같이 남긴다 — 다이얼 스윕에서 덤프를 구분하려면 필수."""
        import json
        cfg = getattr(pl_module, "diffusion_config", None)
        meta = {"num_hierarchies": getattr(pl_module, "num_hierarchies", None),
                "vocab_size": getattr(pl_module, "vocab_size", None)}
        if cfg is not None:
            try:
                meta["diffusion_config"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                            for k, v in dict(cfg).items()}
            except Exception:
                meta["diffusion_config"] = str(cfg)
        try:
            with open(os.path.join(self.out_dir, "dump_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
        except Exception as e:      # 메타 실패가 덤프를 막으면 안 된다
            log.warning(f"[dumper] dump_meta.json 기록 실패: {e}")

    def _extract_users(self, model_input):
        """user_id 를 얻을 수 있는 경로를 순서대로 시도한다."""
        u = _as_int_list(getattr(model_input, "user_id_list", None))
        if u:
            self._user_src = self._user_src or "user_id_list"
            return u
        seqs = getattr(model_input, "transformed_sequences", {}) or {}
        for key in ("user_id", "user_ids"):
            if key in seqs:
                u = _as_int_list(seqs[key])
                if u:
                    self._user_src = self._user_src or f"transformed_sequences[{key}]"
                    return u
        self._user_src = self._user_src or "counter"
        n = model_input.mask.size(0)
        base = len(self.rows)
        return list(range(base, base + n))

    def _capture(self, marginal_probs, generated_ids, labels):
        self._n_batch += 1
        gen = generated_ids.detach().cpu()               # (B, C, H)
        B, C, H = gen.shape
        prob = marginal_probs.detach().cpu().reshape(B, C).float()
        lab = labels.detach().cpu().reshape(B, H)
        users = self._pending_users
        if users is None or len(users) != B:
            users = list(range(len(self.rows), len(self.rows) + B))
            self._user_src = "counter"
        for i in range(B):
            rec = {"user_id": users[i],
                   "semantic_ids": gen[i].to(torch.int16).clone(),
                   "label_sid": lab[i].to(torch.int16).clone()}
            if self.save_scores:
                rec["scores"] = prob[i].clone()
            self.rows.append(rec)
        self._pending_users = None
