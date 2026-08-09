"""층 수 스윕 결과를 그림으로 만든다.

sweep_layers.py 가 만든 json 을 받아 두 장을 그린다.

  왼쪽  L 에 따른 구간별 충돌률
        HEAD/BODY/TAIL 선이 갈라지면 "충돌이 비인기를 더 해친다"는 뜻이고,
        겹치면 충돌은 구간을 가리지 않는다는 뜻이다.
  오른쪽 L 에 따른 고유 SID 수
        아이템 수에 닿으면 그 지점부터 충돌이 사실상 사라진다.

사용:
    python scripts/plot_layer_sweep.py --input work/eval/layer_sweep.json \
        --out work/eval/layer_sweep.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 맥 기본 한글 폰트. 없으면 한글이 네모로 깨진다.
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SEGMENT_STYLE = {
    "HEAD": {"color": "#c0392b", "marker": "o", "label": "HEAD (인기 상위 20%)"},
    "BODY": {"color": "#7f8c8d", "marker": "s", "label": "BODY (중간 60%)"},
    "TAIL": {"color": "#2980b9", "marker": "^", "label": "TAIL (비인기 하위 20%)"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", default=["work/eval/layer_sweep.json"],
                        help="sweep_layers.py 출력 json. 여러 개 주면 모두 합칩니다")
    parser.add_argument("--out", default="work/eval/layer_sweep.png")
    parser.add_argument("--title", default="층 수(L)에 따른 SID 충돌 — Beauty, 코드북 256")
    args = parser.parse_args()

    results = []
    for path in args.input:
        with open(path) as f:
            results.extend(json.load(f))
    results.sort(key=lambda r: r["n_layers"])

    # 같은 L 을 여러 번 돌렸을 수 있다. L 별로 묶어서 평균과 최소~최대를 함께 그린다.
    # (실행 간 변동이 커서 선 하나만 그리면 없는 규칙성을 보여 주게 된다.)
    by_layer: dict[int, list] = {}
    for r in results:
        by_layer.setdefault(r["n_layers"], []).append(r)
    layers = sorted(by_layer)
    n_items = results[0]["stats"]["n_items"]

    def series(getter):
        """L 별 (평균, 최소, 최대, 개별값들)."""
        means, lows, highs, points = [], [], [], []
        for layer in layers:
            values = [getter(r) for r in by_layer[layer]]
            means.append(sum(values) / len(values))
            lows.append(min(values))
            highs.append(max(values))
            points.append(values)
        return means, lows, highs, points

    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(16.5, 4.8))

    for name, style in SEGMENT_STYLE.items():
        means, lows, highs, points = series(
            lambda r, n=name: r["stats"]["by_segment"].get(n, {}).get("collision_rate", 0) * 100
        )
        ax1.plot(layers, means, marker=style["marker"], color=style["color"],
                 label=style["label"], linewidth=2, markersize=7)
        ax1.fill_between(layers, lows, highs, color=style["color"], alpha=0.12)
        # 개별 실행도 점으로 찍는다. 평균만 보면 변동 폭이 안 보인다.
        for layer, values in zip(layers, points):
            ax1.scatter([layer] * len(values), values, color=style["color"],
                        s=12, alpha=0.45, zorder=3)

    means, lows, highs, _ = series(lambda r: r["stats"]["collision_rate"] * 100)
    ax1.plot(layers, means, color="black", linestyle="--", linewidth=1.2,
             label="전체 (평균)", alpha=0.6)

    ax1.set_xlabel("층 수 L")
    ax1.set_ylabel("충돌한 아이템 비율 (%)")
    n_runs = {layer: len(by_layer[layer]) for layer in layers}
    ax1.set_title(f"구간별 충돌률  (실행 수 {min(n_runs.values())}~{max(n_runs.values())}회, 음영=최소~최대)")
    ax1.set_xticks(layers)
    ax1.set_ylim(-3, 103)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)

    # 세 구간 선은 거의 완전히 겹쳐서 왼쪽 그림에서는 하나로만 보인다.
    # 겹친다는 것 자체가 결론이므로, 차이를 따로 확대해 보여 준다.
    _, _, _, head_points = series(
        lambda r: r["stats"]["by_segment"].get("HEAD", {}).get("collision_rate", 0) * 100)
    _, _, _, tail_points = series(
        lambda r: r["stats"]["by_segment"].get("TAIL", {}).get("collision_rate", 0) * 100)

    diffs = [[t - h for h, t in zip(hs, ts)] for hs, ts in zip(head_points, tail_points)]
    ax3.axhline(0, color="black", linewidth=1)
    for layer, values in zip(layers, diffs):
        ax3.scatter([layer] * len(values), values, color="#8e44ad", s=34, alpha=0.75, zorder=3)
    ax3.plot(layers, [sum(v) / len(v) for v in diffs], color="#8e44ad",
             linewidth=2, marker="D", markersize=6, label="평균")

    flat = [v for values in diffs for v in values]
    mean_diff = sum(flat) / len(flat)
    n_pos = sum(1 for v in flat if v > 0)
    ax3.set_xlabel("층 수 L")
    # AppleGothic 에 유니코드 마이너스(U+2212)가 없어 네모로 깨진다. ASCII 하이픈을 쓴다.
    ax3.set_ylabel("TAIL 충돌률 - HEAD 충돌률 (%p)")
    ax3.set_title(
        f"구간 간 차이 (전체 {len(flat)}회)\n"
        f"평균 {mean_diff:+.2f}%p · 양수 {n_pos}/{len(flat)}회 — 0 근처에 머묾"
    )
    ax3.set_xticks(layers)
    ax3.set_ylim(-5, 5)
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=9)

    means, lows, highs, points = series(lambda r: r["stats"]["n_unique_sids"])
    ax2.plot(layers, means, marker="o", color="#27ae60", linewidth=2, markersize=7)
    ax2.fill_between(layers, lows, highs, color="#27ae60", alpha=0.15)
    for layer, values in zip(layers, points):
        ax2.scatter([layer] * len(values), values, color="#27ae60", s=12, alpha=0.45, zorder=3)
    ax2.axhline(n_items, color="gray", linestyle=":", linewidth=1.2)
    # 기준선 바로 아래에 붙인다. 위로 두면 제목과 겹친다.
    ax2.annotate(f"전체 아이템 {n_items:,}개", xy=(layers[-1], n_items),
                 xytext=(-6, -14), textcoords="offset points", fontsize=9,
                 color="gray", ha="right")
    ax2.set_xlabel("층 수 L")
    ax2.set_ylabel("고유 SID 수")
    ax2.set_title("서로 구별되는 SID 개수")
    ax2.set_xticks(layers)
    ax2.grid(alpha=0.25)

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
