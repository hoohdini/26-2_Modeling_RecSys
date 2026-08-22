# 26-2 DSL Modeling — RecSys (생성형 추천)

Semantic ID 기반 생성형 추천(TIGER/MaskGR)에서 **신규·비인기(콜드) 아이템 추천 문제**를
도달가능성 / 도달확률 / 편향 세 축으로 분해해 개선하고, 정확도–롱테일 파레토 곡선으로
검증하는 프로젝트입니다.

## 저장소 구조

```
Data/            Amazon 2014 Beauty 원본 (meta_Beauty.json.gz, reviews_Beauty_5.json.gz)
preprocessing/   전처리 코드(preprocess.py)와 설계 문서
GRID/            snap-research/GRID 스냅샷 — 토큰화(Semantic ID)·임베딩 파이프라인 (팀 패치 2건 반영)
MaskGR/          snap-research/MaskGR 스냅샷 — Masked Diffusion 생성형 추천 모델
embeddings/      flan-t5-xl 아이템 임베딩 (12,101 x 2048)
sid/             RQ-KMeans Semantic ID (3단계 / 4단계)
csv_export/      엑셀에서 바로 열리는 CSV (데이터셋·임베딩·SID) — 발표자료용
code/            TFRecord 변환·CSV 내보내기·GPU 서버 실행 스크립트
docs/            작업 리포트와 인수인계 문서
```

## 📌 지금 상태 (2026-08-19)

**토큰화 → 임베딩 → SID 생성까지 완료**했습니다. 팀원별로 볼 문서는 이렇습니다.

| 문서 | 대상 |
|---|---|
| [`docs/TEAM_RESULTS.md`](docs/TEAM_RESULTS.md) | **전원** — 결과 요약, 결정할 것, 각자 할 일 |
| [`docs/TIGER_BASELINE_REPORT.md`](docs/TIGER_BASELINE_REPORT.md) | **TIGER 베이스라인 결과** — 실험표 ①번 칸 완료 |
| [`docs/REQUEST_전처리_splitB.md`](docs/REQUEST_전처리_splitB.md) | **전처리 담당자** — Split B 프로토콜 변경 요청 |
| [`docs/TRACK_C_시작.md`](docs/TRACK_C_시작.md) | **다양성 평가 트랙** — 새 세션 시작 문서 |
| [`docs/HANDOFF_graph_sid.md`](docs/HANDOFF_graph_sid.md) | **그래프 기반 SID 담당자** — 입력물·비교 기준·지뢰 |
| [`docs/TOKENIZE_EMBED_SID_REPORT.md`](docs/TOKENIZE_EMBED_SID_REPORT.md) | 상세 경위와 전체 수치 |
| [`docs/SERVER_공동사용.md`](docs/SERVER_공동사용.md) | **서버에 들어오는 모든 팀원** — 공용 계정 주의사항·규칙 |
| [`docs/GPU_서버_사용가이드.md`](docs/GPU_서버_사용가이드.md) | 서버 접속 정보·환경 구축·함정 🔒 **계정 비밀번호 포함 — 외부 유출 금지** |
| [`csv_export/README.md`](csv_export/README.md) | CSV 파일 설명 |

핵심 산출물:

```python
import torch
emb = torch.load("embeddings/beauty_A/merged_predictions_tensor.pt")  # (12101, 2048), 행=item_id
sid = torch.load("sid/L4/sid_tensor.pt").T                            # (12101, 5), 행=item_id
#   sid[i, :4] = 코드 4자리(0~255) / sid[i, 4] = 충돌 구분자(0부터)
```

| 설정 | 코드만으로 유일 | 구분자 필요 아이템 | 아이템당 토큰 |
|---|---|---|---|
| 3단계 × 256 | 87.97% | 1,456 (12.03%) | 4 |
| 4단계 × 256 | 93.69% | 763 (6.31%) | 5 |

**TIGER 베이스라인 (실험표 ①번 칸, 4단계 SID · 전체 테스트셋 22,363명):**

| 모델 | Recall@10 | NDCG@10 |
|---|---|---|
| EASE_R (고전 최강) | 0.0505 | 0.0262 |
| **TIGER `clip_L4`** | **0.0660** | **0.0372** |
| TIGER 논문 (Beauty) | 약 0.0648 | 약 0.0384 |

> ⚠️ **`Beauty_split_B.pkl`(절대시간 분할)은 아직 콜드스타트 실험에 쓸 수 없습니다.**
> train이 전체의 0.04~2.62%뿐입니다. 자세한 건 `docs/TEAM_RESULTS.md` 5절.

`GRID/`와 `MaskGR/`는 업스트림 레포를 복사해 넣은 것입니다(`.git` 제거).

> `GRID/` 에는 팀 패치 2건이 반영돼 있습니다 (업스트림 그대로 쓰면 다시 겪습니다):
> 1. `src/utils/inference_utils.py` — 단일 GPU 실행 시 `torch.distributed.barrier()` 로
>    병합 단계가 항상 실패하던 버그 수정
> 2. `src/utils/tensor_utils.py` — SID 충돌 구분자를 0부터 시작하도록 변경

| 폴더 | 업스트림 | 기준 커밋 |
|---|---|---|
| `GRID/` | https://github.com/snap-research/GRID | `2fe3475` (main) |
| `MaskGR/` | https://github.com/snap-research/MaskGR | `b2d44f2` (main) |

## 파이프라인 개요

1. **전처리** — `preprocessing/preprocess.py`
   - Amazon 2014 Beauty 5-core: 유저 22,363 / 상품 12,101 / 상호작용 198,502
   - 산출물: `uid/iid`, `user_seq`, `item_graph`, `item_text`, `item_salesrank`,
     `loo_train/val/test`(LOO 분할), `temporal_splits`(절대시간 분할), `cold_tiers_per_window`
   - 세부 근거와 함정(예: meta는 `ast.literal_eval`로 파싱, 중복 제거 금지, 동점 시각의
     2차 정렬 키)은 `preprocessing/전처리_설계문서.md` 참고
2. **토큰화·임베딩·SID** — `GRID/` (**완료**)
   - TFRecord 변환: `code/to_grid_v2.py` → `grid_data/beauty_A`
   - LLM 임베딩 추출: `experiment=sem_embeds_inference_flat` (google/flan-t5-xl 인코더, 2048차원)
   - Semantic ID 생성: RQ-KMeans (`rkmeans_train_flat`), 레벨 3·4 두 설정 × 코드북 256
   - 충돌 처리: 마지막 자리에 0부터 시작하는 일련번호 (TIGER 방식)
   - 실행 스크립트: `code/server/embed_beauty.sh`, `code/server/sid_beauty.sh`
   - ⚠️ GRID 입력 TFRecord는 GZIP 압축 필수, 폴더명은 `training/evaluation/testing`
   - ⚠️ `num_workers=0` + `timeout=0` + `persistent_workers=false` 를 세트로 줘야 함
3. **모델** — TIGER 베이스라인 재현 **완료** (`code/server/tiger_beauty.sh`),
   MaskGR(`MaskGR/`) 실험은 다음 단계
   - 학습 레시피: warmup 1,500 + cosine decay + gradient clipping 1.0, 배치 256
   - ⚠️ `num_hierarchies` = SID 텐서 행 수, `vocab_size` = NH × 256 (자세한 함정은 리포트 4절)

## 참고 논문

- TIGER — 재현 베이스라인 (SID 패러다임)
- GRID Handbook ([2507.22224](https://arxiv.org/abs/2507.22224)) — 코드베이스·설계 근거
- MaskGR ([2511.23021](https://arxiv.org/abs/2511.23021)) — Masked Diffusion 생성형 추천
- Can GR Reach Cold Items? — 콜드스타트 원인 진단·절대시간 분할 프로토콜
- CRAB — 코드북 재균형 방법론
