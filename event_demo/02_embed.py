# -*- coding: utf-8 -*-
"""프로필 문장 → 임베딩 (경량 다국어 모델, 로컬 GPU/CPU).

모델  intfloat/multilingual-e5-small (118M, 384차원). 행사 서비스가 경량 모델을 쓸 계획이라 같은 급으로 맞춘다.
입력  D:/DSL/_event_data/profiles.csv
출력  D:/DSL/_event_data/emb.npy  (N, 384) float32, L2 정규화
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # anaconda MKL 과 torch 의 OpenMP 충돌 회피
import sys, time
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer

D = r"D:/DSL/_event_data"
MODEL = os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small")
sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv(os.path.join(D, "profiles.csv"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = SentenceTransformer(MODEL, device=dev)
t0 = time.time()
E = m.encode(["query: " + t for t in df["text"]], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
E = np.asarray(E, dtype=np.float32)
np.save(os.path.join(D, "emb.npy"), E)
print(f"모델 {MODEL} · 장치 {dev} · {E.shape} · {time.time()-t0:.1f}초")

# 간단 검증: 최근접 이웃이 같은 학과인 비율
S = E @ E.T; np.fill_diagonal(S, -1)
nn = S.argmax(1)
same_dept = (df["dept"].values[nn] == df["dept"].values).mean()
same_cohort = (df["cohort"].values[nn] == df["cohort"].values).mean()
print(f"최근접 이웃 같은 학과 {same_dept:.2f} · 같은 기수 {same_cohort:.2f}  (학과 무작위 기대 {(df['dept'].value_counts(normalize=True)**2).sum():.2f})")
