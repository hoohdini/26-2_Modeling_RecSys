# -*- coding: utf-8 -*-
"""v4 시뮬레이터 재보정 — 캐글 목표 분포(G3)에 맞추되 G-L 을 통과하는 설정을 고른다.

원칙 (docs/PREREG_생성데이터셋.md §3, JOBS_학습실패_원인분석.md §5)
    · 만지는 것: TEMP, GAMMA, K_MU, K_SIGMA (순환과 무관한 레버)
    · 안 만지는 것: OBS_NOISE(14), SIGMA_R(10), TOP_M(20)   — 순환 위험을 올린다
    · 프로필·텍스트·그래프는 v3 그대로. 상호작용만 다시 만든다 (JOBS_IN_DIR = data_gen/out)
    · 선택 규칙을 실행 전에 고정한다:
        1) G-L 통과(ContentKNN/Pop ≥ 1.5 AND ItemKNN/Pop ≥ 1.5)인 설정만 후보
        2) 후보 중 G3 점수(항목별 |오차|/WARN한계 의 합)가 가장 작은 것
        3) 동점이면 v3 값(temp 0.15, gamma 0.20, k 1.55/0.75)에 가까운 것

실행 (한 설정에 로컬 약 4분 · --jobs 로 병렬)
    python data_gen/v4/calibrate_simulator.py --target data_gen/spec/kaggle_target_dist.json --jobs 4
    python data_gen/v4/calibrate_simulator.py --target ... --smoke        # 2설정 · 담당자 3,000 명 (동작 확인)

산출
    data_gen/v4/out/calib/<cfg>/     interactions.csv, Jobs_split_A.pkl, gl.json, g3.json
    data_gen/v4/out/calib/table.md   전체 표
    data_gen/spec/v4_params.json     선택된 파라미터 (v4 생성 명령의 입력)
"""
import argparse
import io
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(ROOT, "data_gen")
V3_OUT = os.path.join(GEN, "out")
CALIB = os.path.join(GEN, "v4", "out", "calib")
PY = sys.executable
V3 = {"temp": 0.15, "gamma": 0.20, "k_mu": 1.55, "k_sigma": 0.75}

sys.path.insert(0, os.path.join(GEN, "v4"))
from gate_g3_stats import ITEMS, compare, from_jobs  # noqa: E402


def cfg_name(c):
    return "t%g_g%g_km%g_ks%g" % (c["temp"], c["gamma"], c["k_mu"], c["k_sigma"])


def run_one(c, a):
    name = cfg_name(c) + ("_smoke" if a.smoke else "")   # smoke 산출물이 본 실행에 재사용되지 않도록 폴더를 분리
    d = os.path.join(CALIB, name)
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, JOBS_IN_DIR=V3_OUT, JOBS_OUT_DIR=d, PYTHONIOENCODING="utf-8")
    log = io.open(os.path.join(d, "log.txt"), "w", encoding="utf-8")
    t0 = time.time()
    if not os.path.exists(os.path.join(d, "interactions.csv")):
        cmd = [PY, os.path.join(GEN, "simulate_interactions.py"), "--temp", str(c["temp"]),
               "--gamma", str(c["gamma"]), "--k-mu", str(c["k_mu"]), "--k-sigma", str(c["k_sigma"]),
               "--cand-pool", "12000", "--seed", "42", "--out", os.path.join(d, "interactions.csv")]
        if a.smoke:
            cmd += ["--recruiters", "3000", "--interactions", "30000"]
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not os.path.exists(os.path.join(d, "Jobs_split_A.pkl")):
        subprocess.run([PY, os.path.join(GEN, "build_splits.py")], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    gl_path = os.path.join(d, "gl.json")
    if not os.path.exists(gl_path):
        subprocess.run([PY, os.path.join(GEN, "gate_learnability.py"), "--split",
                        os.path.join(d, "Jobs_split_A.pkl"), "--json", gl_path],
                       env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    gl = json.load(io.open(gl_path, encoding="utf-8"))
    st = from_jobs(os.path.join(d, "interactions.csv"), os.path.join(V3_OUT, "profiles.csv"))
    rows = compare(st, a.target_obj, a.tol)
    g3 = 0.0
    n_fail = 0
    for _, key, g, t, err, v in rows:
        if err is None or v == "REPORT":
            continue
        tl = a.tol.get(key, a.tol.get("default_rel" if "/" in key and key.split("/")[-1].startswith("p") else "default_abs"))
        warn = tl[0] if isinstance(tl, list) else tl
        g3 += err / max(warn, 1e-9)
        n_fail += v == "FAIL"
    res = {"cfg": c, "name": name, "sec": round(time.time() - t0, 1),
           "gl": gl["ratio"], "gl_pass": gl["ratio"]["ContentKNN"] >= 1.5 and gl["ratio"]["ItemKNN"] >= 1.5,
           "g3_score": round(g3, 3), "g3_fail": n_fail,
           "top50": st["apps_per_job_applied_only"]["top50_share"], "gini": st["apps_per_job_applied_only"]["gini"],
           "per_user_p50": st["apps_per_user"]["p50"], "zero_share": st["share_jobs_with_zero_apps"],
           "n_inter": st["n_interactions"], "rows": rows}
    json.dump({k: v for k, v in res.items() if k != "rows"}, io.open(os.path.join(d, "g3.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="kaggle_target_dist.json (또는 beauty_ref_dist.json)")
    ap.add_argument("--tolerance", default=os.path.join(GEN, "spec", "g3_tolerance.json"))
    ap.add_argument("--temp", default="0.10,0.15,0.25")
    ap.add_argument("--gamma", default="0.10,0.20,0.35")
    ap.add_argument("--k-mu", default="1.55")
    ap.add_argument("--k-sigma", default="0.75,1.0")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-params", default=os.path.join(GEN, "spec", "v4_params.json"))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    a.target_obj = json.load(io.open(a.target, encoding="utf-8"))
    a.tol = json.load(io.open(a.tolerance, encoding="utf-8"))
    f = lambda s: [float(x) for x in s.split(",")]
    grid = [dict(temp=t, gamma=g, k_mu=km, k_sigma=ks)
            for t, g, km, ks in itertools.product(f(a.temp), f(a.gamma), f(a.k_mu), f(a.k_sigma))]
    if a.smoke:
        grid = grid[:2]
    os.makedirs(CALIB, exist_ok=True)
    print("설정 %d개 · 병렬 %d · 목표 %s" % (len(grid), a.jobs, a.target_obj.get("source", a.target)))

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        results = list(ex.map(lambda c: run_one(c, a), grid))

    # 선택 규칙 (고정)
    cand = [r for r in results if r["gl_pass"]]
    dist_v3 = lambda r: sum(abs(r["cfg"][k] - V3[k]) / max(abs(V3[k]), 1e-9) for k in V3)
    chosen = min(cand, key=lambda r: (r["g3_score"], dist_v3(r))) if cand else None

    L = ["| 설정 | temp | gamma | k_mu | k_sigma | C/P | K/P | G-L | G3점수 | G3 FAIL | top50 | Gini | 유저당 p50 | 0건 비율 | 선택 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: (not r["gl_pass"], r["g3_score"])):
        c = r["cfg"]
        L.append("| %s | %g | %g | %g | %g | %.2f | %.2f | %s | %.2f | %d | %.3f | %.3f | %.0f | %.3f | %s |" % (
            r["name"], c["temp"], c["gamma"], c["k_mu"], c["k_sigma"], r["gl"]["ContentKNN"], r["gl"]["ItemKNN"],
            "통과" if r["gl_pass"] else "미달", r["g3_score"], r["g3_fail"], r["top50"], r["gini"],
            r["per_user_p50"], r["zero_share"], "선택" if chosen and r["name"] == chosen["name"] else ""))
    table = "\n".join(L)
    io.open(os.path.join(CALIB, "table.md"), "w", encoding="utf-8", newline="\n").write(table + "\n")
    print(table)
    if chosen is None:
        print("\nG-L 을 통과한 설정이 없다. 격자를 바꿔 다시 돌릴 것 (temp 를 낮추면 적합도 항이 강해진다).")
        return 1
    params = {"chosen": chosen["cfg"], "chosen_name": chosen["name"], "g3_score": chosen["g3_score"],
              "gl": chosen["gl"], "target": a.target_obj.get("source", a.target), "smoke": a.smoke,
              "fixed": {"obs_noise": 14.0, "sigma_r": 10.0, "top_m": 20, "cand_pool": 12000, "seed": 42},
              "rule": "G-L 통과 후보 중 G3 점수 최소, 동점이면 v3 값에 가까운 것",
              "command": "python data_gen/simulate_interactions.py --temp %g --gamma %g --k-mu %g --k-sigma %g --cand-pool 12000 --seed 42" % (
                  chosen["cfg"]["temp"], chosen["cfg"]["gamma"], chosen["cfg"]["k_mu"], chosen["cfg"]["k_sigma"])}
    if not a.smoke:
        json.dump(params, io.open(a.out_params, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("\n선택: %s → %s" % (chosen["name"], a.out_params))
    else:
        print("\n(smoke 모드라 v4_params.json 을 쓰지 않았다) 선택: %s" % chosen["name"])
    print("명령: " + params["command"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
