"""층 수(L)를 바꿔가며 SID 충돌률을 구간별로 잰다.

왜 이걸 먼저 재는가
    연구 질문은 "층 수를 줄이면 비인기 구간이 좋아지는가"다.
    그런데 층을 줄이면 충돌(서로 다른 아이템이 같은 SID를 받는 일)도 같이 늘어난다.
    두 효과를 분리하지 않으면 "L이 짧아서 나쁜지, 충돌이 많아서 나쁜지" 알 수 없다.
    docs/03_실험_계획.md 가 "충돌률을 반드시 같이 재야 한다"고 한 이유다.

    다행히 충돌률은 TIGER 학습 없이 Step 3~4 만으로 나온다. 맥에서 L 하나당 몇 분이면
    끝나므로, 본 실험 전에 미리 지형을 볼 수 있다.

핵심 질문
    TAIL(비인기) 아이템이 낮은 L 에서 HEAD 보다 더 많이 충돌하는가?
    그렇다면 "L 을 줄이면 TAIL 이 나빠진다"의 기계적 원인이 설명된다.

사용:
    python scripts/sweep_layers.py --data-dir data/amazon_data/beauty \
        --segments work/segments/beauty.json --layers 2,3,4,5,6,8
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
GRID_DIR = PROJECT / "code" / "grid"
SEGMENT_ORDER = ["HEAD", "BODY", "TAIL", "UNSEEN"]


def newest(pattern_root: Path, glob: str, experiment: str | None = None,
           since: float | None = None) -> Path:
    """방금 만든 산출물을 찾는다. GRID 는 실행마다 타임스탬프 폴더를 새로 만든다.

    🚨 그냥 "가장 최근 파일"을 쓰면 안 된다.
       다른 GRID 실행(예: 같은 맥에서 돌리는 TIGER 학습)이 겹치면 그쪽 체크포인트를
       집어 온다. 에러가 안 나고 결과만 조용히 틀린다. 실제로 밟을 뻔했다.
       그래서 두 가지로 거른다.
         since       이번 실행을 시작한 뒤에 만들어진 것만
         experiment  .hydra/overrides.yaml 에 그 experiment 가 적힌 실행만
    """
    candidates = []
    for path in pattern_root.glob(glob):
        if since is not None and path.stat().st_mtime < since:
            continue
        if experiment is not None:
            overrides = path.parent.parent / ".hydra" / "overrides.yaml"
            if not overrides.exists() or f"experiment={experiment}" not in overrides.read_text():
                continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"찾지 못함: {pattern_root}/{glob}"
            + (f" (experiment={experiment})" if experiment else "")
            + (" — 이번 실행 이후에 만들어진 것이 없습니다" if since else "")
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_grid(entrypoint: str, overrides: list[str]) -> None:
    cmd = [str(PROJECT / "scripts" / "grid_mac.sh"), entrypoint, *overrides]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-25:])
        raise RuntimeError(f"GRID 실행 실패 ({entrypoint}):\n{tail}")


def build_sids(data_dir: str, embedding_path: str, embedding_dim: int, n_layers: int,
               codebook_width: int, extra: list[str] | None = None) -> Path:
    """Step 3(코드북 학습) → Step 4(SID 생성). 생성된 SID 텐서 경로를 돌려준다.

    extra 는 Step 3(학습)에만 붙인다. trainer.max_steps 처럼 학습에만 있는 키를
    추론 쪽에 넘기면 Hydra 가 거부하기 때문이다.
    """
    common = [
        f"data_dir='{data_dir}'",
        f"embedding_path='{embedding_path}'",
        f"embedding_dim={embedding_dim}",
        f"num_hierarchies={n_layers}",
        f"codebook_width={codebook_width}",
    ]

    started = time.time()
    run_grid("train", ["experiment=rkmeans_train_flat", *common, *(extra or [])])
    ckpt = newest(GRID_DIR / "logs" / "train" / "runs", "*/*/checkpoints/*.ckpt",
                  experiment="rkmeans_train_flat", since=started)

    started = time.time()
    run_grid(
        "inference",
        ["experiment=rkmeans_inference_flat", *common,
         f"ckpt_path='{ckpt}'", "callbacks.bq_writer=null"],
    )
    return newest(GRID_DIR / "logs" / "inference" / "runs",
                  "*/*/pickle/merged_predictions_tensor.pt",
                  experiment="rkmeans_inference_flat", since=started)


def load_sids(path: Path) -> np.ndarray:
    """(N, L+1) 로 맞춘다. 마지막 열은 중복 제거용 자릿수다."""
    tensor = torch.load(path, weights_only=False)
    array = tensor.numpy()
    if array.shape[0] < array.shape[1]:
        array = array.T
    return array.astype(np.int64)


# 층 하나가 이 비트 미만이면 "무너졌다"고 본다.
# 256개 코드워드를 고르게 쓰면 8비트다. 4비트면 실질적으로 16개만 쓰는 셈이라
# 그 층은 SID 를 거의 구분하지 못한다.
COLLAPSE_BITS = 4.0


def level_entropy(column: np.ndarray) -> float:
    """한 층이 실제로 담고 있는 정보량(비트).

    코드워드를 몇 개 "썼는가"로는 부족하다. 256개를 다 쓰더라도 그중 하나가
    대부분을 삼키면 그 층은 사실상 아무것도 구분하지 못한다.
    실제로 그런 일이 일어난다 — docs/10_층수_충돌률_결과.md 참고.
    그래서 쏠림까지 반영하는 엔트로피로 잰다.
    고르게 쓰면 log2(256)=8비트, 한 곳에 몰리면 0비트에 가깝다.
    """
    _, counts = np.unique(column, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def collision_stats(sids: np.ndarray, segments: dict) -> dict:
    """충돌 = 중복 제거 자릿수를 빼고 봤을 때 같은 SID 를 가진 아이템이 또 있는 것."""
    prefix = sids[:, :-1]
    n_items = len(prefix)

    _, inverse, counts = np.unique(prefix, axis=0, return_inverse=True, return_counts=True)
    group_size = counts[inverse]              # 각 아이템이 속한 충돌 그룹의 크기
    collides = group_size > 1

    bits = [level_entropy(sids[:, d]) for d in range(sids.shape[1] - 1)]
    stats = {
        "n_items": n_items,
        "n_unique_sids": int(len(counts)),
        "collision_rate": float(collides.mean()),
        "max_group_size": int(counts.max()),
        "codebook_used": [int(len(np.unique(sids[:, d]))) for d in range(sids.shape[1] - 1)],
        "level_bits": [round(b, 2) for b in bits],
        # 층 수보다 이 값이 충돌률을 훨씬 잘 설명한다
        # (16회 실행 실측 스피어만: 유효비트 -0.96 vs 층 수 -0.75)
        "effective_bits": round(sum(bits), 2),
        "n_collapsed_levels": sum(1 for b in bits if b < COLLAPSE_BITS),
        "dedup_digits_needed": int(sids[:, -1].max()) + 1,
        "by_segment": {},
    }

    for name in SEGMENT_ORDER:
        ids = [i for i in range(n_items) if segments.get(i) == name]
        if not ids:
            continue
        idx = np.array(ids)
        stats["by_segment"][name] = {
            "n_items": len(ids),
            "collision_rate": float(collides[idx].mean()),
            "mean_group_size": float(group_size[idx].mean()),
        }
    return stats


def render(results: list[dict]) -> str:
    lines = [
        f"{'L':>3}{'유효비트':>10}{'붕괴층':>8}{'고유SID':>10}"
        f"{'전체충돌':>10}{'HEAD':>9}{'BODY':>9}{'TAIL':>9}{'최대그룹':>9}",
        "─" * 78,
    ]
    for r in results:
        s = r["stats"]
        seg = s["by_segment"]
        lines.append(
            f"{r['n_layers']:>3}{s['effective_bits']:>10.1f}"
            f"{s['n_collapsed_levels']:>8}{s['n_unique_sids']:>10,}"
            f"{s['collision_rate']:>10.1%}"
            f"{seg.get('HEAD', {}).get('collision_rate', 0):>9.1%}"
            f"{seg.get('BODY', {}).get('collision_rate', 0):>9.1%}"
            f"{seg.get('TAIL', {}).get('collision_rate', 0):>9.1%}"
            f"{s['max_group_size']:>9,}"
        )

    lines += [
        "",
        "층별 정보량 (비트. 코드워드 256개를 고르게 쓰면 8.0, 한 곳에 몰리면 0)",
    ]
    for r in results:
        bits = r["stats"]["level_bits"]
        marks = "".join("·" if b >= COLLAPSE_BITS else "✗" for b in bits)
        lines.append(f"  L={r['n_layers']}: {bits}  {marks}")
    lines += [
        f"  ✗ = {COLLAPSE_BITS}비트 미만 = 무너진 층. 자리는 차지하지만 아이템을 구분하지 못한다.",
        "",
        "🚨 충돌률은 L 이 아니라 유효비트를 따라간다.",
        "   층을 늘려도 그 층이 무너지면 SID 공간은 넓어지지 않는다.",
        "   층 수 실험 결과를 읽을 때 유효비트를 같이 보지 않으면 해석이 뒤집힌다.",
    ]
    return "\n".join(lines)


def reanalyze(path: str, segments: dict) -> list[dict]:
    """이미 만들어 둔 SID 텐서로 통계만 다시 낸다.

    지표를 새로 추가했을 때 GRID 를 다시 돌릴 필요가 없다. Step 3~4 는 L 하나당
    몇 십 분씩 걸리므로 이 차이가 크다.
    """
    with open(path) as f:
        results = json.load(f)
    for r in results:
        sid_path = Path(r["sid_path"])
        if not sid_path.exists():
            print(f"⚠️ 건너뜀 (파일 없음): {sid_path}")
            continue
        r["stats"] = collision_stats(load_sids(sid_path), segments)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--embedding-path", default="work/step2_768/beauty/merged_predictions_tensor.pt")
    parser.add_argument("--embedding-dim", type=int, default=768)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--layers", default="2,3,4,5,6,8")
    parser.add_argument("--codebook-width", type=int, default=256)
    parser.add_argument("--out", default="work/eval/layer_sweep.json")
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="같은 L 을 몇 번 반복할지. 반복마다 시드를 1씩 올린다",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="첫 반복에 쓸 시드. 이후 반복은 +1 씩 올라간다",
    )
    parser.add_argument(
        "--extra", default="",
        help="Step 3 에 추가로 넘길 Hydra 오버라이드. 쉼표 구분 "
             "(예: trainer.max_steps=300)",
    )
    parser.add_argument(
        "--reanalyze", default=None,
        help="이미 저장한 스윕 결과(.json)의 SID 텐서로 통계만 다시 낸다. GRID 재실행 없음",
    )
    args = parser.parse_args()

    extra = [x for x in args.extra.split(",") if x]

    data_dir = str(Path(args.data_dir).resolve())
    embedding_path = str(Path(args.embedding_path).resolve())

    with open(args.segments) as f:
        segments = {int(k): v for k, v in json.load(f)["segments"].items()}

    if args.reanalyze:
        results = reanalyze(args.reanalyze, segments)
        print(render(results))
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n저장 완료: {args.out}")
        return

    results = []
    for n_layers in [int(x) for x in args.layers.split(",")]:
        for repeat in range(args.repeats):
            # 🚨 반복할 때는 시드를 바꿔야 한다.
            # 셔플 시드 패치를 적용하면 같은 시드는 비트 단위로 같은 결과를 낸다.
            # 시드를 안 바꾸면 같은 값을 세 번 적는 것뿐이다.
            seed = args.seed + repeat
            started = time.time()
            tag = f"L={n_layers}" + (f" seed={seed}" if args.repeats > 1 else "")
            print(f"[{tag}] Step 3~4 실행 중...", flush=True)
            sid_path = build_sids(data_dir, embedding_path, args.embedding_dim,
                                  n_layers, args.codebook_width,
                                  [*extra, f"seed={seed}"])
            stats = collision_stats(load_sids(sid_path), segments)
            results.append({
                "n_layers": n_layers,
                "repeat": repeat + 1,
                "seed": seed,
                "codebook_width": args.codebook_width,
                "extra": extra,
                "sid_path": str(sid_path),
                "seconds": round(time.time() - started, 1),
                "stats": stats,
            })
            seg = stats["by_segment"]
            print(
                f"[{tag}] 완료 {results[-1]['seconds']}초 · "
                f"전체 충돌 {stats['collision_rate']:.1%} · "
                f"HEAD {seg.get('HEAD', {}).get('collision_rate', 0):.1%} / "
                f"TAIL {seg.get('TAIL', {}).get('collision_rate', 0):.1%}",
                flush=True,
            )
            # 중간에 끊겨도 여기까지가 남도록 매번 저장한다.
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print(render(results))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
