# -*- coding: utf-8 -*-
"""학습 없이 test 단계만 실행하는 엔트리포인트 — 트랙 C 전용.

src.train 을 그대로 쓸 수 없는 이유
-----------------------------------
src/train.py 의 test 블록은 `ckpt_path` 를 **설정에서 읽지 않는다.**
ModelCheckpoint 콜백의 best_model_path 만 본다.

    ckpt_path = getattr(checkpoint_callback, "best_model_path", None)
    if ckpt_path == "": ckpt_path = None
    if not ckpt_path: "Best checkpoint not found! Using current weights..."

즉 `train=False test=True ckpt_path=...` 로 돌리면 학습을 건너뛴 뒤
**초기화 직후의 랜덤 가중치로 테스트**한다. 조용히 잘못된 지표가 나온다.

이 파일은 train.py 와 같은 방식으로 파이프라인을 세우되
trainer.test 에 cfg.ckpt_path 를 명시적으로 넘긴다. 기존 파일은 건드리지 않는다.

설치: 서버의 GRID/src/eval_dump.py 로 복사.
실행: python -m src.eval_dump experiment=tiger_train_flat ... ckpt_path='<경로>'
"""
from typing import Any, Dict, Optional, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras  # noqa: E402
from src.utils.custom_hydra_resolvers import *  # noqa: E402,F401,F403
from src.utils.launcher_utils import pipeline_launcher  # noqa: E402

command_line_logger = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("medium")


def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("ckpt_path 가 필요합니다. 없으면 랜덤 가중치로 평가하게 됩니다.")

    with pipeline_launcher(cfg) as pipeline_modules:
        command_line_logger.info(f"Testing from checkpoint: {ckpt_path}")
        pipeline_modules.trainer.test(
            model=pipeline_modules.model,
            datamodule=pipeline_modules.datamodule,
            ckpt_path=ckpt_path,
        )
        metric_dict = dict(pipeline_modules.trainer.callback_metrics)
        command_line_logger.info(f"Metrics: {metric_dict}")
    return metric_dict, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
