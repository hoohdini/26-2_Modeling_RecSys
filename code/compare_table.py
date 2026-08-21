#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""여러 결과 JSON 을 한 표로 합친다 — 정확도 x 다양성 비교 + 파레토 판정.

results/*.json 은 {label: {지표...}} 형식이면 무엇이든 받는다.
(baselines_a.py / tiger_to_eval.py 둘 다 이 형식으로 저장한다)

사용:
  python compare_table.py results/baseline_splitA_loo.json results/tiger_clip_L4_loo.json
  python compare_table.py --k 20 --out docs/TRACK_C_비교표.md results/*.json
"""
import argparse
import glob
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import pareto_table  # noqa: E402


def load_all(paths, k):
    """{label: 지표} 형식과 pareto_dial 의 {"points": {label: 지표}} 형식을 모두 받는다."""
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        blocks = [d["points"]] if isinstance(d.get("points"), dict) else [d]
        for block in blocks:
            for label, m in block.items():
                if label.startswith("_") or not isinstance(m, dict):
                    continue      # _test_EASE_vs_KNN 같은 부록 항목은 건너뛴다
                if f"ndcg@{k}" not in m:
                    continue
                m = dict(m)
                m["label"] = m.get("label", label)
                m["_file"] = os.path.basename(p)
                rows.append(m)
    return rows


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--k", type=int, default=10,
                    help="표에 쓸 컷오프. 정확도 파트(Recall@5/@10)와 맞춰 10 이 기본")
    ap.add_argument("--out", default=None, help="마크다운 저장 경로")
    a = ap.parse_args()

    paths = []
    for p in a.paths:
        paths.extend(sorted(glob.glob(p)) or [p])
    rows = load_all(paths, a.k)
    if not rows:
        print("읽을 결과가 없습니다.")
        return
    k = a.k
    need = [f"recall@{k}", f"ndcg@{k}", f"tail_exposure@{k}"]
    rows = [r for r in rows if all(n in r for n in need)]

    lines = []
    lines.append(f"| 모델 | Recall@{k} | NDCG@{k} | **APLT@{k}** | Coverage@{k} | "
                 f"Gini@{k} | COLD-R@50 | 주의 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -x[f"ndcg@{k}"]):
        warn = []
        if r.get("_INVALID"):
            warn.append("정답누출")
        src = r.get("_source", {})
        if src.get("truncated_at") and src["truncated_at"] < 50:
            warn.append(f"top{int(src['truncated_at'])}까지만")
        if src.get("exclude_seen") is False:
            warn.append("본상품 미제외")
        lines.append(
            f"| {r['label']} | {fmt(r[f'recall@{k}'])} | {fmt(r[f'ndcg@{k}'])} | "
            f"**{fmt(r[f'tail_exposure@{k}'])}** | {fmt(r[f'coverage@{k}'])} | "
            f"{fmt(r[f'exposure_gini@{k}'])} | "
            f"{fmt(r.get('COLD_recall@50', float('nan')))} | {', '.join(warn) or '-'} |")

    # APLT(유저별 비율의 평균) 와 tail_exposure(전체 슬롯 비율) 는 모든 추천 목록의
    # 길이가 같으면 수학적으로 동일하다. 그래서 한 칼럼으로 합쳤다.
    # 무효 SID 제거 등으로 K 개를 못 채운 유저가 있으면 갈라지므로 매번 확인한다.
    gap = max(abs(r[f"aplt@{k}"] - r[f"tail_exposure@{k}"]) for r in rows)
    lines.append("")
    lines.append(f"APLT@{k} = 추천 슬롯 중 롱테일(학습 인기 하위 50%, 6,051개) 상품의 비율. "
                 f"내부 키 `tail_exposure@{k}` 와 같은 값이라 한 칼럼으로 싣는다"
                 f"(최대 차이 {gap:.6f}).")
    if gap > 1e-4:
        lines.append("")
        lines.append(f"> **주의**: 두 값이 {gap:.6f} 만큼 갈라졌다. "
                     f"추천 목록이 {k}개를 못 채운 유저가 있다는 뜻이므로 원인을 확인할 것.")
        print(f"[경고] aplt 와 tail_exposure 가 최대 {gap:.6f} 차이납니다.")

    # 파레토: 정확도 x 롱테일 노출
    acc, div = f"ndcg@{k}", f"tail_exposure@{k}"
    pts = pareto_table(rows, acc=acc, div=div)
    lines.append("")
    lines.append(f"**파레토 판정 (NDCG@{k} x APLT@{k})**")
    lines.append("")
    lines.append("| 모델 | 정확도 | 다양성 | 프론티어 | 지배하는 모델 |")
    lines.append("|---|---|---|---|---|")
    for p in sorted(pts, key=lambda x: -x["acc"]):
        lines.append(f"| {p['label']} | {fmt(p['acc'])} | {fmt(p['div'])} | "
                     f"{'O' if p['on_frontier'] else 'X'} | "
                     f"{', '.join(p['dominated_by']) or '-'} |")

    md = "\n".join(lines)
    print(md)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
