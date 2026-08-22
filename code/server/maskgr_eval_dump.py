# -*- coding: utf-8 -*-
"""학습 없이 test 단계만 실행하는 MaskGR 엔트리포인트 — 트랙 C 전용.

왜 따로 두는가
--------------
1) MaskGR 의 Makefile 은 `make inference` 로 `src/inference.py` 를 부르는데
   **저장소에 그 파일이 없다.** 그대로는 추론 산출물을 뽑을 수 없다.
2) `src/train.py` 의 test 블록은 GRID 보다는 낫다 — best_model_path 가 빈 문자열이면
   cfg.ckpt_path 로 폴백한다. 하지만 checkpoint 콜백이 없거나 best_model_path 가
   None 이면 여전히 **랜덤 가중치로 평가**하고 경고 한 줄만 남긴다.
   트랙 C 는 이 실수를 GRID 에서 이미 한 번 겪었으므로(리포트 1-3절) 명시적으로 막는다.

이 파일은 train.py 와 같은 파이프라인을 세우되 trainer.test 에 cfg.ckpt_path 를
반드시 넘긴다. 기존 파일은 한 줄도 건드리지 않는다.

설치: MaskGR/src/maskgr_eval_dump.py 로 복사
실행: python -m src.maskgr_eval_dump experiment=discrete_diffusion_train ... ckpt_path='<경로>'
"""
import os
from typing import Any, Dict, Optional, Tuple

os.environ["TORCHSNAPSHOT_ENABLE_SHARDED_TENSOR_ELASTICITY_ROOT_ONLY"] = "1"

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
        raise ValueError(
            "ckpt_path 가 필요합니다. 없으면 랜덤 가중치로 평가하게 됩니다. "
            "(GRID 에서 실제로 당한 함정 — 리포트 1-3절)"
        )
    if not os.path.exists(str(ckpt_path)):
        raise FileNotFoundError(f"체크포인트가 없습니다: {ckpt_path}")

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
