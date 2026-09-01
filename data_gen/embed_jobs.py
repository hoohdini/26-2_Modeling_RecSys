"""프로필 텍스트 → flan-t5-xl 인코더 임베딩 (12000 x 2048).

GRID 의 `sem_embeds_inference_flat` 설정을 그대로 재현한다.
    · google/flan-t5-xl 의 **인코더만** 사용
    · attention_mask 를 반영한 토큰 평균 풀링
    · max_length=128, truncation=True, padding="max_length"
    · fp32

이렇게 만든 텐서는 `sid_beauty.sh` 의 `EMB` 인자로 그대로 넣을 수 있다
(GRID 는 임베딩 경로를 받으므로 생성 주체가 무엇인지 상관하지 않는다).

서버 실행:
    sbatch embed_jobs.sh
"""
import argparse
import os

import torch
from transformers import AutoTokenizer, T5EncoderModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="TSV: id<TAB>text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/flan-t5-xl")
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    rows = []
    with open(args.texts, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                i, t = line.rstrip("\n").split("\t", 1)
                rows.append((int(i), t))
    rows.sort()
    ids = [i for i, _ in rows]
    assert ids == list(range(len(ids))), "item_id 는 0..N-1 로 조밀해야 한다"
    texts = [t for _, t in rows]
    print(f"텍스트 {len(texts):,}건", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = T5EncoderModel.from_pretrained(args.model, torch_dtype=torch.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    print(f"모델 적재 완료 · device={dev}", flush=True)

    # 한글이었다면 <unk> 로 뭉개졌을 것이므로 실제 unk 비율을 남긴다 (검증 기록)
    unk = tok.unk_token_id
    sample = tok(texts[:200], truncation=True, max_length=args.max_length)["input_ids"]
    n_unk = sum(t.count(unk) for t in sample)
    n_tok = sum(len(t) for t in sample)
    print(f"unk 비율(앞 200건): {n_unk}/{n_tok} = {n_unk / max(n_tok,1):.4%}", flush=True)
    over = sum(1 for t in tok(texts)["input_ids"] if len(t) > args.max_length)
    print(f"{args.max_length}토큰 초과: {over}/{len(texts)}", flush=True)

    out = torch.empty(len(texts), model.config.d_model, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, len(texts), args.batch):
            e = min(s + args.batch, len(texts))
            enc = tok(texts[s:e], return_tensors="pt", truncation=True,
                      max_length=args.max_length, padding="max_length").to(dev)
            h = model(**enc).last_hidden_state              # (B, L, D)
            m = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out[s:e] = pooled.float().cpu()
            if s % (args.batch * 20) == 0:
                print(f"  {e}/{len(texts)}", flush=True)

    torch.save(out, args.out)
    print(f"저장 {tuple(out.shape)} · 평균 {out.mean():.6f} · 노름중앙 "
          f"{out.norm(dim=1).median():.4f}\n→ {args.out}")


if __name__ == "__main__":
    main()
