"""GRID의 LocalPickleWriter._merge_files 와 동일한 병합을 단독 수행.
GRID 원본은 단일 프로세스 실행에서 torch.distributed.barrier() 로 죽기 때문에
(on_predict_end 의 `if trainer.global_rank != None` 조건이 rank 0 에서도 참)
추론 결과 pkl 조각만 남고 병합이 안 된다. 그 조각들을 여기서 합친다."""
import os, pickle, shutil, sys
import torch
sys.path.insert(0, "/mnt/data1/dsl05/recsys/GRID")
from src.utils.tensor_utils import merge_list_of_keyed_tensors_to_single_tensor

out_dir = sys.argv[1]
backup = os.path.join(out_dir, "_shards_backup")

shards = sorted(f for f in os.listdir(out_dir) if f.endswith(".pkl") and f.startswith("predictions_"))
print(f"shards: {len(shards)}")
os.makedirs(backup, exist_ok=True)
for f in shards:
    shutil.copy2(os.path.join(out_dir, f), os.path.join(backup, f))
print("backed up shards ->", backup)

merged = []
for f in shards:
    with open(os.path.join(out_dir, f), "rb") as fh:
        merged.extend(pickle.load(fh))
print("merged rows:", len(merged))

with open(os.path.join(out_dir, "merged_predictions.pkl"), "wb") as fh:
    pickle.dump(merged, fh)

tensor = merge_list_of_keyed_tensors_to_single_tensor(
    data=merged, index_key="item_id", value_key="embedding"
)
torch.save(tensor.cpu(), os.path.join(out_dir, "merged_predictions_tensor.pt"))
print("tensor shape:", tuple(tensor.shape), "dtype:", tensor.dtype)

ids = sorted(int(r["item_id"]) for r in merged)
print("item_id min/max:", ids[0], ids[-1], "unique:", len(set(ids)))
zero_rows = int((tensor.abs().sum(dim=1) == 0).sum())
print("all-zero rows:", zero_rows)
print("norm mean:", float(tensor.norm(dim=1).mean()))
for f in shards:
    os.remove(os.path.join(out_dir, f))
print("MERGE_DONE")
