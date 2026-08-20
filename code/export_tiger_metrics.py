"""TIGER 학습/테스트 지표를 정리해 CSV로 내보낸다.

입력: tiger_runs/<TAG>/metrics.csv   (GRID CSVLogger 출력을 내려받은 것)
출력: csv_export/tiger/
"""
import io
import os
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"D:\DSL\RecSys"
RUNS = os.path.join(ROOT, "tiger_runs")
OUT = os.path.join(ROOT, "csv_export", "tiger")
os.makedirs(OUT, exist_ok=True)
ENC = "utf-8-sig"

if not os.path.isdir(RUNS):
    print(f"실행 결과 폴더가 없습니다: {RUNS}")
    sys.exit(1)

summary_rows = []

for tag in sorted(os.listdir(RUNS)):
    mpath = os.path.join(RUNS, tag, "metrics.csv")
    if not os.path.exists(mpath):
        continue
    m = pd.read_csv(mpath)

    # ── 학습 곡선 (검증 지표만 남기기) ─────────────────────────
    val_cols = [c for c in m.columns if c.startswith("val/")]
    curve = m[["step"] + val_cols].dropna(subset=val_cols, how="all").copy()
    curve = curve.sort_values("step")
    p = os.path.join(OUT, f"tiger_{tag}_val_curve.csv")
    curve.to_csv(p, index=False, encoding=ENC)
    print(f"[{os.path.getsize(p)/1e3:7.1f} KB] {os.path.basename(p)}  ({len(curve)} 회 검증)")

    # ── 학습 손실 곡선 ────────────────────────────────────────
    loss_cols = [c for c in m.columns if c.startswith("train/")]
    if loss_cols:
        tr = m[["step"] + loss_cols].dropna(subset=loss_cols, how="all").sort_values("step")
        p = os.path.join(OUT, f"tiger_{tag}_train_loss.csv")
        tr.to_csv(p, index=False, encoding=ENC)
        print(f"[{os.path.getsize(p)/1e3:7.1f} KB] {os.path.basename(p)}  ({len(tr)} 포인트)")

    # ── 최종 테스트 지표 ──────────────────────────────────────
    test_cols = [c for c in m.columns if c.startswith("test/")]
    row = {"run": tag}
    if test_cols:
        t = m[test_cols].dropna(how="all")
        if len(t):
            for c in test_cols:
                v = t[c].dropna()
                if len(v):
                    row[c] = float(v.iloc[-1])
    # 검증 최고점(체크포인트 선택 기준)
    if "val/recall@5" in m.columns:
        v = m["val/recall@5"].dropna()
        if len(v):
            row["best_val/recall@5"] = float(v.max())
            best_idx = m["val/recall@5"].idxmax()
            row["best_step"] = int(m.loc[best_idx, "step"])
    row["val_checks"] = len(curve)
    summary_rows.append(row)

if summary_rows:
    s = pd.DataFrame(summary_rows)
    # 보기 좋은 열 순서
    order = ["run", "best_step", "best_val/recall@5", "val_checks",
             "test/recall@5", "test/recall@10", "test/ndcg@5", "test/ndcg@10", "test/loss"]
    cols = [c for c in order if c in s.columns] + [c for c in s.columns if c not in order]
    s = s[cols]
    p = os.path.join(OUT, "tiger_summary.csv")
    s.to_csv(p, index=False, encoding=ENC)
    print(f"\n[{os.path.getsize(p)/1e3:7.1f} KB] tiger_summary.csv")
    print(s.to_string(index=False))
else:
    print("정리할 실행 결과가 없습니다.")
