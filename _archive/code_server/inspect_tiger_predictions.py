"""학습된 TIGER 체크포인트로 실제 예측을 뽑아 육안 검증한다.

지표 코드를 거치지 않는 독립 검증. 유저 몇 명에 대해
  입력 히스토리(실제 상품명) / 모델이 생성한 SID 후보 / 정답 SID
를 나란히 출력한다.

서버에서 실행:
  cd /mnt/data1/dsl05/recsys/GRID
  /mnt/data1/dsl05/miniconda3/envs/grid/bin/python \
      ~/recsys/inspect_tiger_predictions.py <ckpt경로> [유저수]

주의
  - torch 2.6 부터 torch.load 기본값이 weights_only=True 라 weights_only=False 필요
  - 체크포인트가 src.* 객체를 피클로 담고 있어, 먼저 src.train 을 임포트해
    정상 임포트 순서를 태워야 순환 임포트를 피할 수 있다
"""
import os
import pickle
import sys

import rootutils
import torch

GRID_ROOT = "/mnt/data1/dsl05/recsys/GRID"
if not os.path.isdir(GRID_ROOT):
    GRID_ROOT = "/data1/dsl05/recsys/GRID"
rootutils.setup_root(os.path.join(GRID_ROOT, "src", "train.py"),
                     indicator=".project-root", pythonpath=True)
import src.train  # noqa: F401  (순환 임포트 회피용)
from src.models.modules.semantic_id.tiger_generation_model import SemanticIDEncoderDecoder

BASE = os.path.dirname(os.path.dirname(GRID_ROOT))          # .../dsl05
RECSYS = os.path.join(BASE, "recsys")
SID_PATH = f"{RECSYS}/sid_out/baseline/L4/infer/pickle/merged_predictions_tensor.pt"
DATA = f"{RECSYS}/grid_data/beauty_A"
PKL = f"{RECSYS}/Beauty_split_A.pkl"

ckpt_path = sys.argv[1]
n_users = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# ── SID <-> item ────────────────────────────────────────────────
sid = torch.load(SID_PATH, map_location="cpu", weights_only=False)   # (5, 12101)
sid_t = sid.t()
n_items, n_h = sid_t.shape
sid2item = {tuple(int(x) for x in sid_t[i]): i for i in range(n_items)}
print(f"SID: {n_items:,} 아이템 x {n_h} 자리 · 고유 {len(sid2item):,}개")

item_text = {}
if os.path.exists(PKL):
    with open(PKL, "rb") as f:
        item_text = pickle.load(f)["item_text"]
name = lambda i: (item_text.get(i, "") or "")[:66] if item_text else f"item {i}"

# ── 테스트 시퀀스 ───────────────────────────────────────────────
import tensorflow as tf

feat = {"user_id": tf.io.FixedLenFeature([], tf.int64),
        "sequence_data": tf.io.VarLenFeature(tf.int64)}
rows = []
for raw in tf.data.TFRecordDataset([f"{DATA}/testing/part-0.tfrecord.gz"],
                                   compression_type="GZIP"):
    ex = tf.io.parse_single_example(raw, feat)
    rows.append((int(ex["user_id"].numpy()),
                 tf.sparse.to_dense(ex["sequence_data"]).numpy().tolist()))
    if len(rows) >= n_users:
        break

# ── 모델 ────────────────────────────────────────────────────────
model = SemanticIDEncoderDecoder.load_from_checkpoint(ckpt_path, map_location="cpu")
model.eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"체크포인트 global_step={ck.get('global_step')} · device={dev}\n")

hits = 0
for u, seq in rows:
    ctx, target = seq[:-1], seq[-1]
    tgt_sid = tuple(int(x) for x in sid_t[target])
    print("=" * 76)
    print(f"user {u} · 히스토리 {len(ctx)}개")
    for i in ctx[-4:]:
        print(f"   이력 [{i:5d}] {name(i)}")
    print(f"   정답 [{target:5d}] {'-'.join(map(str,tgt_sid))}  {name(target)}")

    ids = torch.tensor([[int(x) for x in sid_t[i]] for i in ctx],
                       dtype=torch.long).view(1, -1).to(dev)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask)
    gen = out[0] if isinstance(out, (tuple, list)) else out
    gen = gen.detach().cpu().view(-1, n_h)

    print("   생성 후보:")
    hit = False
    for r, cand in enumerate(gen[:10], 1):
        t = tuple(int(x) for x in cand)
        it = sid2item.get(t)
        mark = "  <== 정답" if t == tgt_sid else ""
        hit = hit or (t == tgt_sid)
        print(f"     {r:2d}. {'-'.join(map(str,t)):16s} "
              f"{name(it) if it is not None else '(존재하지 않는 SID)'}{mark}")
    hits += hit
    print(f"   => {'HIT' if hit else 'MISS'}")

print("=" * 76)
print(f"표본 {len(rows)}명 중 HIT {hits}명")
