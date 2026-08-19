import pickle, torch, random
emb = torch.load(r"D:\DSL\RecSys\embeddings\beauty_A\merged_predictions_tensor.pt", map_location="cpu")
print("shape:", tuple(emb.shape), emb.dtype)

with open(r"D:\DSL\RecSys\Beauty_split_A.pkl", "rb") as f:
    d = pickle.load(f)
print("pkl keys:", list(d.keys())[:12])

item_text = d.get("item_text")
if isinstance(item_text, dict):
    texts = {int(k): v for k, v in item_text.items()}
else:
    texts = {i: t for i, t in enumerate(item_text)}
print("n item_text:", len(texts))

X = torch.nn.functional.normalize(emb, dim=1)
random.seed(0)
probes = random.sample(sorted(texts), 4)
for p in probes:
    sims = X @ X[p]
    sims[p] = -2
    top = torch.topk(sims, 3)
    print("\n=== QUERY", p, "===")
    print("  ", texts[p][:130])
    for s, j in zip(top.values.tolist(), top.indices.tolist()):
        print(f"   sim={s:.3f} [{j}] {texts[j][:120]}")

off = X @ X[probes[0]]
print("\nglobal cos-sim mean/std vs a random item:", float(off.mean()), float(off.std()))
