# 26-2 DSL Modeling — RecSys (생성형 추천)

Semantic ID 기반 생성형 추천(TIGER/MaskGR)에서 **신규·비인기(콜드) 아이템 추천 문제**를
도달가능성 / 도달확률 / 편향 세 축으로 분해해 개선하고, 정확도–롱테일 파레토 곡선으로
검증하는 프로젝트입니다.

## 저장소 구조

```
Data/            Amazon 2014 Beauty 원본 (meta_Beauty.json.gz, reviews_Beauty_5.json.gz)
preprocessing/   전처리 코드(preprocess.py)와 설계 문서
GRID/            snap-research/GRID 스냅샷 — 토큰화(Semantic ID)·임베딩 파이프라인
MaskGR/          snap-research/MaskGR 스냅샷 — Masked Diffusion 생성형 추천 모델
```

`GRID/`와 `MaskGR/`는 업스트림 레포를 그대로 복사해 넣은 것입니다(`.git` 제거).

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
2. **토큰화·임베딩** — `GRID/`
   - LLM 임베딩 추출: `experiment=sem_embeds_inference_flat` (google/flan-t5-xl)
   - Semantic ID 생성: RQ-KMeans (`rkmeans_train_flat`), num_hierarchies 3·4 두 설정
   - ⚠️ GRID 입력 TFRecord는 GZIP 압축 필수, 폴더명은 `training/evaluation/testing`
3. **모델** — TIGER 베이스라인 재현(`GRID/`) 및 MaskGR(`MaskGR/`) 실험

## 참고 논문

- TIGER — 재현 베이스라인 (SID 패러다임)
- GRID Handbook ([2507.22224](https://arxiv.org/abs/2507.22224)) — 코드베이스·설계 근거
- MaskGR ([2511.23021](https://arxiv.org/abs/2511.23021)) — Masked Diffusion 생성형 추천
- Can GR Reach Cold Items? — 콜드스타트 원인 진단·절대시간 분할 프로토콜
- CRAB — 코드북 재균형 방법론
