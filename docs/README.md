# docs/ — 문서 분류표

문서가 많아 무엇이 최신인지 알기 어렵습니다. 이 표 하나로 고르세요. **최종 갱신 2026-09-19.**
발표용 요약과 인용 금지 수치는 [`00_발표_가이드.md`](00_발표_가이드.md), 저장소 전체 진입점은 [`../README.md`](../README.md).

## 🟢 정본 — 수치를 인용할 때 보는 것

| 문서 | 내용 | 갱신 |
|---|---|---|
| [`00_발표_가이드.md`](00_발표_가이드.md) | 30초 스토리, 슬라이드 뼈대와 근거, 인용 금지 수치, 예상 질문 | 9/19 |
| [`SHARE_GSID_결과요약.md`](SHARE_GSID_결과요약.md) | Beauty G-SID 결과 전부. β₀ 스윕 승자, 3시드, 실패한 대조 2건 | 8/31 |
| [`RESULTS_실험표.md`](RESULTS_실험표.md) | Beauty 실험표 ①~⑤ 정본 (8칸 · Temporal · CRAB · 다이얼 · β₀ 스윕) | 8/28 |
| [`JOBS_결과정리.md`](JOBS_결과정리.md) | 구인구직 결과 정본. v3 3시드(§1~9), **v3 6시드 판정(§10)**, **v4 재현(§11)**, zero-shot §8 | 9/19 |
| [`DATASET_v4_파이프라인.md`](DATASET_v4_파이프라인.md) | 구인구직 v4 단계표, GPU 일정, 판정 규칙, 아직 없는 것 | 9/19 |
| `../results/reports/` | 9/15 이후 자동 판정표 원문: P1·P3(GSID_WEEK), v4 n=6, P2 Temporal | 9/19 |
| [`../data_gen/README.md`](../data_gen/README.md) | 구인구직 데이터셋 출처·스키마·재현·게이트·v4 | 9/19 |

## 🟡 배경 — 왜 그렇게 했는지 물어보면 여는 것

| 문서 | 내용 |
|---|---|
| [`PREREG_생성데이터셋.md`](PREREG_생성데이터셋.md) | 구인구직 사전등록. G0~G5 게이트 정의, 하지 않기로 한 것, 변경 절차 |
| [`G0_비중첩.md`](G0_비중첩.md) · [`G3_허용구간.md`](G3_허용구간.md) · [`G3_목표분포_캐글.md`](G3_목표분포_캐글.md) · [`G3_통계정합성.md`](G3_통계정합성.md) | 게이트별 정의와 판정 결과 |
| [`LICENSE_데이터출처.md`](LICENSE_데이터출처.md) | 고용24·캐글 이용허락. 파생 파일 공개 배포 불가 근거 |
| [`JOBS_학습실패_원인분석.md`](JOBS_학습실패_원인분석.md) | v1·v2 를 왜 버렸나. G-L 게이트의 근거 |
| [`JOBS_데이터셋_구축기록.md`](JOBS_데이터셋_구축기록.md) | v3 구축 경위 |
| [`TRACK_C_보고서.md`](TRACK_C_보고서.md) · [`TRACK_C_비교표.md`](TRACK_C_비교표.md) | 다양성·롱테일 트랙. 노이즈 바닥의 출처, 파레토 다이얼 |
| [`TIGER_BASELINE_REPORT.md`](TIGER_BASELINE_REPORT.md) · [`TOKENIZE_EMBED_SID_REPORT.md`](TOKENIZE_EMBED_SID_REPORT.md) | 파이프라인 초기 리포트 (베이스라인 재현, 임베딩·SID) |
| [`SPLIT_재전처리_영향.md`](SPLIT_재전처리_영향.md) · [`../preprocessing/전처리_설계문서.md`](../preprocessing/전처리_설계문서.md) · [`../preprocessing/전처리_수정문서.md`](../preprocessing/전처리_수정문서.md) | 분할 프로토콜과 8/22 재전처리 |
| [`../patches/README.md`](../patches/README.md) | GRID 접두사 제약 들여쓰기 버그 |

## ⚪ 운영 — 서버에 들어가거나 다른 사람 작업을 이어받을 때

| 문서 | 내용 |
|---|---|
| [`GPU_서버_사용가이드.md`](GPU_서버_사용가이드.md) 🔒 · [`SERVER_공동사용.md`](SERVER_공동사용.md) | 접속, Slurm, 공용 계정 규칙 |
| [`HANDOFF_graph_sid.md`](HANDOFF_graph_sid.md) · [`HANDOFF_MaskGR.md`](HANDOFF_MaskGR.md) · [`HANDOFF_트랙C_다양성평가.md`](HANDOFF_트랙C_다양성평가.md) | 파트별 인수인계 |
| [`../code/README.md`](../code/README.md) | 코드 의존 관계, 실행 순서, 반복해서 당한 함정 |

## 🔴 기록 — 끝났거나 낡은 것 (`_archive/`)

읽어도 되지만 **수치를 인용하지 마세요.** 지우지 않고 [`_archive/`](_archive/) 로 옮겼습니다.

| 문서 | 왜 여기 있나 |
|---|---|
| `_archive/TEAM_RESULTS.md` | 8/19 토큰화·임베딩 팀 공유. `TOKENIZE_EMBED_SID_REPORT.md` 가 상위 호환 |
| `_archive/TRACK_C_시작.md` | 8/21 트랙 C 세션 시작 문서. 보고서로 대체됨 |
| `_archive/RUNBOOK_텍스트SID_3트랙.md` | 텍스트 SID 3트랙 런북. 8/25 에 전부 완료 |
| `_archive/REQUEST_전처리_splitB.md` | Split B 프로토콜 변경 요청. 8/22 재전처리로 반영 완료 |
| `_archive/JOBS_야간작업_현황.md` | 구인구직 v1(폐기본) 시점 기록 |
| `_archive/TRACK_C_아티팩트.html` | 구 split 기준 그림 |

문서를 새로 만들 때는 이 표에 한 줄 추가하고, 끝난 문서는 `_archive/` 로 옮긴 뒤 여기 적어 두세요.
