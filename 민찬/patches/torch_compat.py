"""PyTorch 2.6+ 에서 GRID 체크포인트를 읽기 위한 호환 계층.

PyTorch 2.6 부터 `torch.load` 의 `weights_only` 기본값이 True 로 바뀌었다.
GRID 는 체크포인트의 hyper_parameters 안에 ItemDataloaderConfig 같은 자체
데이터클래스를 통째로 절여 넣기 때문에, weights_only=True 로는 열리지 않는다.

Lightning 은 `torch.load(..., weights_only=True)` 처럼 값을 명시해서 넘기므로
기본값만 바꾸는 것으로는 우회되지 않는다. 그래서 GRID 가 만든 체크포인트에
한해 이 인자를 False 로 되돌린다.

안전성: 여기서 여는 파일은 모두 이 저장소가 직접 학습해 만든 로컬 체크포인트다.
외부에서 받은 체크포인트를 열 때는 이 패치를 쓰면 안 된다.
"""

import torch

_original_load = torch.load
_patched = False


def allow_full_unpickling() -> None:
    """torch.load 가 weights_only=False 로 동작하도록 되돌린다 (한 번만 적용)."""
    global _patched
    if _patched:
        return

    def load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = load
    _patched = True
