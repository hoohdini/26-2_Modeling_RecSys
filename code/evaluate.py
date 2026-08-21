#!/usr/bin/env python3
"""
평가 하네스 — 모델보다 먼저 만들어야 하는 것
  · 버킷별 정확도 (head/mid/low/few-shot/zero-shot + COLD 통합)
  · 다양성 (APLT, Coverage, Gini, ILD)
  · 통계 검정 (paired bootstrap)
  · 오라클 대비 회복률
  · 파레토 곡선 데이터

사용:
  from evaluate import Evaluator, paired_bootstrap, recovery_rate
  ev = Evaluator(train_counts, n_items, tail_frac=0.5)
  m  = ev.evaluate(preds, targets, K=(10,20,50))
"""
import json, math
from collections import Counter, defaultdict
import numpy as np

# 학습 등장 횟수 기준 버킷 (STAGE12_REPORT 확정안)
BUCKETS = [(0, 0, "zero-shot"), (1, 2, "few-shot"), (3, 5, "low"),
           (6, 20, "mid"), (21, 10**9, "head")]
COLD_MAX = 5          # COLD 통합 = 학습 등장 <= 5회


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if n == 0 or x.sum() == 0: return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


class Evaluator:
    """train_counts: Counter(item -> 학습 등장 횟수), n_items: 전체 아이템 수"""

    def __init__(self, train_counts, n_items, tail_frac=0.5):
        self.tc = Counter(train_counts)
        self.n_items = n_items
        # 롱테일 정의: 학습 인기 하위 tail_frac
        ranked = sorted(range(n_items), key=lambda i: -self.tc.get(i, 0))
        self.tail = set(ranked[int(n_items * (1 - tail_frac)):])
        self.bucket_of = {}
        for i in range(n_items):
            c = self.tc.get(i, 0)
            for lo, hi, name in BUCKETS:
                if lo <= c <= hi: self.bucket_of[i] = name; break

    # ---------- per-user 지표 ----------
    @staticmethod
    def _dcg(rel):
        return float(sum(r / math.log2(k + 2) for k, r in enumerate(rel)))

    def _user_metrics(self, rank, gt, K):
        """rank: 추천 리스트(내림차순), gt: 정답 집합"""
        top = rank[:K]
        hits = [1.0 if i in gt else 0.0 for i in top]
        n_hit = sum(hits)
        recall = n_hit / len(gt) if gt else 0.0
        hr = 1.0 if n_hit > 0 else 0.0
        idcg = self._dcg([1.0] * min(len(gt), K))
        ndcg = self._dcg(hits) / idcg if idcg > 0 else 0.0
        aplt = float(np.mean([1.0 if i in self.tail else 0.0 for i in top])) if top else 0.0
        return {"recall": recall, "hr": hr, "ndcg": ndcg, "aplt": aplt}

    # ---------- 전체 평가 ----------
    def evaluate(self, preds, targets, K=(10, 20, 50), per_user=False):
        """preds: {user: [item,...] 내림차순}, targets: {user: set(item)}"""
        users = [u for u in targets if u in preds and targets[u]]
        out = {"n_user_eval": len(users), "n_target": sum(len(targets[u]) for u in users)}
        pu = {}
        for k in K:
            rows = [self._user_metrics(preds[u], targets[u], k) for u in users]
            for m in ("recall", "hr", "ndcg", "aplt"):
                out[f"{m}@{k}"] = round(float(np.mean([r[m] for r in rows])), 6)
            pu[k] = {u: rows[j] for j, u in enumerate(users)}

            # 노출 기반 다양성
            exp = Counter()
            for u in users: exp.update(preds[u][:k])
            out[f"coverage@{k}"] = round(len(exp) / self.n_items, 6)
            v = np.zeros(self.n_items);
            for i, c in exp.items():
                if i < self.n_items: v[i] = c
            out[f"exposure_gini@{k}"] = round(gini(v), 6)
            out[f"tail_exposure@{k}"] = round(
                float(sum(c for i, c in exp.items() if i in self.tail) / max(1, sum(exp.values()))), 6)

        # ---------- 버킷별 (타깃 아이템 기준) ----------
        k0 = max(K)
        bt = defaultdict(lambda: {"hit": 0, "n": 0})
        for u in users:
            top = set(preds[u][:k0])
            for g in targets[u]:
                b = self.bucket_of.get(g, "head")
                bt[b]["n"] += 1
                bt[b]["hit"] += 1 if g in top else 0
        out[f"bucket_recall@{k0}"] = {b: {"n": d["n"], "recall": round(d["hit"]/max(1,d["n"]), 6)}
                                      for b, d in bt.items()}
        cold_n = sum(d["n"] for b, d in bt.items() if b in ("zero-shot", "few-shot", "low"))
        cold_h = sum(d["hit"] for b, d in bt.items() if b in ("zero-shot", "few-shot", "low"))
        out[f"COLD_recall@{k0}"] = round(cold_h / max(1, cold_n), 6)
        out["COLD_n_target"] = cold_n
        return (out, pu) if per_user else out


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """a, b: {user: score}. 같은 유저 집합에서 차이의 부트스트랩 신뢰구간."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys], float)
    if len(d) == 0: return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {"n": len(d), "mean_diff": round(float(d.mean()), 6),
            "ci95": [round(float(np.percentile(means, 2.5)), 6),
                     round(float(np.percentile(means, 97.5)), 6)],
            "p_value_gt0": round(float((means <= 0).mean()), 6)}


def recovery_rate(baseline, ours, oracle):
    """오라클 대비 회복률(%). (Ours-Base)/(Oracle-Base)*100"""
    denom = oracle - baseline
    if abs(denom) < 1e-12: return None
    return round((ours - baseline) / denom * 100, 2)


def pareto_table(points, acc="ndcg@20", div="tail_exposure@20"):
    """points: [{'label':..., 'w':..., metrics...}] → 파레토 곡선용 표 + 지배여부"""
    rows = [{"label": p.get("label"), "w": p.get("w"),
             "acc": p[acc], "div": p[div]} for p in points]
    for r in rows:
        r["dominated_by"] = [s["label"] for s in rows
                             if s is not r and s["acc"] >= r["acc"] and s["div"] >= r["div"]
                             and (s["acc"] > r["acc"] or s["div"] > r["div"])]
        r["on_frontier"] = len(r["dominated_by"]) == 0
    return rows


def load_split(pkl_path, split="temporal", window=0):
    """build/*.pkl → (train_counts, train_hist, targets, n_items)"""
    import pickle
    D = pickle.load(open(pkl_path, "rb"))
    n_items = D["n_item"]
    if split == "temporal":
        w = D["temporal"][window]
        hist = {u: list(v) for u, v in w["train"].items()}
        tgt = defaultdict(set)
        for u, g in w["test"]: tgt[u].add(g)
    else:  # leave-one-out
        hist = {u: list(v) for u, v in D["loo"]["train"].items()}
        tgt = {u: {g} for u, g in D["loo"]["test"].items()}
    tc = Counter()
    for v in hist.values(): tc.update(v)
    return tc, hist, dict(tgt), n_items


def load_split_a(pkl_path):
    """Beauty_split_A.pkl (loo_train/loo_val/loo_test 스키마) → (train_counts, hist, targets, n_items)

    구 스키마(build/Beauty.pkl 의 D["loo"]["train"] / D["n_item"])를 쓰는 load_split 과 달리
    현행 전처리 산출물을 직접 읽는다. 학습 등장 횟수는 loo_train 만으로 센다
    (loo_val 은 모델이 test 시점에 입력으로 볼 수 있으나, 버킷/롱테일 정의는
     '학습에 쓰인 신호'만으로 고정하는 편이 트랙 간 비교에 안전하다).
    """
    import pickle
    with open(pkl_path, "rb") as f:
        D = pickle.load(f)
    n_items = len(D["iid"])
    hist = {u: list(v) for u, v in D["loo_train"].items()}
    tgt = {u: {g} for u, g in D["loo_test"].items()}
    tc = Counter()
    for v in hist.values():
        tc.update(v)
    return tc, hist, tgt, n_items
