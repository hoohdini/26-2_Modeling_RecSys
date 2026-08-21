# -*- coding: utf-8 -*-
"""테스트 단계(eval_step)의 생성 결과를 파일로 떨구는 Lightning 콜백.

왜 필요한가
----------
GRID 의 `src.inference`(predict_step) 는 정답을 입력에서 가리지 않는다. 전체 시퀀스를
넣고 "그 다음"을 생성하는 실서비스 경로다. 그 산출물로 평가하면 지표가 부풀려진다
(실측: clip_L4 에서 Recall@10 이 0.0660 -> 0.0961 로 45% 부풀려짐).

반면 `test_step -> eval_step` 은 test 데이터로더가 NextKTokenMasking 으로 마지막
K 토큰(=정답 SID)을 마스킹해 넣어주므로 올바른 평가 입력이다. 다만 생성 결과를
지표로만 집계하고 버린다. 이 콜백은 그 중간 산출물을 가로채 저장한다.

원본을 고치지 않는 이유
----------------------
서버 GRID 는 팀원과 공유한다. 그래서 기존 파일은 한 줄도 건드리지 않고,
테스트 시작 시점에 두 지점만 런타임에 감싼다(테스트가 끝나면 원상복구).

  1) `pl_module.eval_step`  -> 배치에서 user_id 를 뽑아 보관
  2) `pl_module.evaluator`  -> generated_ids / marginal_probs / labels 를 가로챔

`evaluator` 는 nn.Module 이 아니라 평범한 속성(SIDRetrievalEvaluator)이라
그냥 감싸면 된다. 지표 집계는 원본이 그대로 수행하므로 로그 값도 달라지지 않는다.

설치
----
  이 파일을 서버의 GRID/src/callbacks/prediction_dumper.py 로 복사하고
  같은 폴더에 빈 __init__.py 를 둔다. (둘 다 새 파일이므로 기존 동작에 영향 없음)

사용 (hydra 오버라이드)
-----------------------
  +callbacks.pred_dump._target_=src.callbacks.prediction_dumper.TigerPredictionDumper
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


class TigerPredictionDumper(Callback):
    """test 단계의 생성 SID 를 out_dir/test_predictions_rank{r}.pkl 로 저장한다.

    레코드 한 건:
      {"user_id": int, "semantic_ids": (C, H) int16, "scores": (C,) float32,
       "label_sid": (H,) int16}
    C = model.top_k_for_generation, H = num_hierarchies.
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

        def tapped_eval_step(batch, loss_to_aggregate):
            self._pending_users = self._extract_users(batch[0])
            return self._orig_eval_step(batch, loss_to_aggregate)

        pl_module.eval_step = tapped_eval_step

        self._orig_evaluator = pl_module.evaluator
        pl_module.evaluator = _EvaluatorTap(self._orig_evaluator, self)
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
