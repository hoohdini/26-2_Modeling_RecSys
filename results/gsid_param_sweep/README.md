# β₀ 스윕 + 다중시드 검증 — 지표 JSON 원본 (2026-08-28)

`docs/RESULTS_실험표.md` 실험표 ⑤의 원본입니다. 전부 **재전처리(8/22) 이후의
신규 Split A/B** 로 채점 (COLD 분모 6,197 — 구 실험표 ①~④의 6,597 과 다름).
채점 하네스: 서버 `work/eval/tiger_to_eval.py`, 덤프는 전부 TOPK=200.

## 파일 명명

- `ns_<조건>_s<시드>.json` — LOO (Split A)
- `nsB_<조건>_<윈도우>[_s7].json` — Temporal (Split B, W4/W5). 시드 표기 없으면 42.

| 조건 코드 | 의미 | SID |
|---|---|---|
| `base` / `text` | baseline 텍스트 SID | `sid_out/baseline` |
| `T1` | 튜닝식 β₀0.5 (τ없음·κ0) | `sid_out/sid_T1_noTau_k0_b05` |
| `b07` | 튜닝식 β₀0.7 | `sid_out/sid_noTau_k0_b07` |
| `b10` | **튜닝식 β₀1.0 — 승자** | `sid_out/sid_noTau_k0_b10` |
| `a05c` | 고전 블렌딩 α0.5 (중심화) | `sid_out/gsid_a05_centered_nahye` |
| `a07` | 고전 블렌딩 α0.7 (중심화) | `sid_out/sid_blend_centered_a07` |

주의: 기존 조건(base/T1/a05c 시드42·7 등)도 **기존 덤프를 신규 스플릿으로 재채점**한
것이라, `results/experiment_8cells/` 의 같은 조건 수치와 다릅니다 (특히 COLD·APLT).
W5 는 tgfix 레시피(8k 스텝) 기준으로 통일했습니다.
