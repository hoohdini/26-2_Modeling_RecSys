# -*- coding: utf-8 -*-
"""데이터셋 매니페스트 — 사전등록 §10 산출물 `data_gen/dataset_manifest.json`.

버전별로 파일 해시·행 수·생성 파라미터·커밋을 남긴다. 결과 JSON 과 대조해
"어느 데이터 위에서 나온 수치인가"를 답할 수 있게 한다.

    python data_gen/v4/dataset_manifest.py --version v3 --params-json data_gen/spec/v3_params.json
    python data_gen/v4/dataset_manifest.py --version v4 --dir data_gen/v4/out/v4 --params-json data_gen/spec/v4_params.json
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(ROOT, "data_gen")
FILES = ["profiles.csv", "profile_skill.npy", "profile_text.tsv", "profile_regtime.npy", "skill_dims.txt",
         "jobs_edges_final.csv", "jobs_edges_long.csv", "jobs_edges_rewired.csv",
         "interactions.csv", "jobs_interactions_long.csv", "Jobs_split_A.pkl", "Jobs_split_B.pkl", "jobs_text_emb.pt"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nrows(p):
    if p.endswith((".csv", ".tsv", ".txt")):
        with open(p, "rb") as f:
            return sum(1 for _ in f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--dir", default=os.path.join(GEN, "out"), help="데이터 파일 위치")
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--out", default=os.path.join(GEN, "dataset_manifest.json"))
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        commit = None
    entry = {"created": datetime.now().strftime("%Y-%m-%d %H:%M"), "commit": commit, "dir": os.path.relpath(a.dir, ROOT),
             "params": json.load(io.open(a.params_json, encoding="utf-8")) if a.params_json else None, "files": {}}
    for n in FILES:
        p = os.path.join(a.dir, n)
        if os.path.exists(p):
            entry["files"][n] = {"sha256": sha(p), "bytes": os.path.getsize(p), "rows": nrows(p)}
    man = json.load(io.open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {}
    man[a.version] = entry
    json.dump(man, io.open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("%s: 파일 %d개 · 커밋 %s → %s" % (a.version, len(entry["files"]), commit, a.out))


if __name__ == "__main__":
    main()
