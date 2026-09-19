# 구인구직 데이터셋 v4 — 파이프라인과 GPU 사용 계획

작성 2026-09-15. 노션 계획(2차 발표 준비 계획 → 2. 구인구직 데이터셋 구축 계획)의 실행 단계 1~7을 저장소 안의 스크립트와 일정으로 옮긴 것이다. 정본 근거는 `docs/PREREG_생성데이터셋.md`(사전등록), `docs/JOBS_데이터셋_구축기록.md`(v3), `docs/G3_허용구간.md`(오늘 고정).

## 0. 한 장 요약

| | |
|---|---|
| 목표 | v3 데이터(이미 H1 재현)를 그대로 두고, **상호작용만** 캐글 실측 분포에 맞춰 다시 만든 v4 로 같은 실험을 반복한다. v3 와 v4 를 별도 결과로 보고한다 |
| 오늘 끝난 것 | v4 스크립트 7종 작성·동작 확인, G3 허용 구간 고정, 사전등록 정정(LLM → 템플릿), 라이선스 표, 서버 스크립트(prep/train/seeds6), 매니페스트 v3 |
| 사람이 해야 열리는 것 | **캐글 다운로드**(규칙 동의 필요, 자동화 불가) → 그 뒤 분포 산출·매핑·캘리브레이션은 전부 스크립트 |
| GPU | 9/16 개정: 실험 트랙이 미착수였고 GPU 가 비어 있어 v4 GPU 를 9/16~9/18 로 당겼다(3절). 약 71 GPU시간 |
| 서버 상태 (9/15 밤 확인) | 9/9 이후 제출 작업 없음 = 실험 트랙 미착수. GPU 2장 가용. P1+P3 12셀을 23:42 에 제출했다 (`code/run_gsid_week.sh`, 자동 채점·판정 포함) |

## 1. 단계표

CPU 단계는 전부 로컬 PC(파이썬 3.13, numpy·pandas·scipy·scikit-learn)에서 돈다. torch 는 필요 없다. GPU 단계만 학과 서버다.

| # | 단계 | 어디서 | 명령 | 입력 → 출력 | 소요 | 게이트 |
|---|---|---|---|---|---|---|
| 1 | 캐글 다운로드 | 사람 | kaggle.com/c/job-recommendation/data 에서 규칙 동의 후 zip 을 `D:/DSL/_external/kaggle_job_recommendation/` 에 풀기 | users, apps, user_history, jobs, window_dates tsv | 10분 | 라이선스 표(`LICENSE_데이터출처.md`) 확인 |
| 2 | 목표 분포 산출 | 로컬 CPU | `python data_gen/v4/kaggle_stats.py` | tsv → `data_gen/spec/kaggle_target_dist.json`, `docs/G3_목표분포_캐글.md`, 저장소 밖 `title_freq.tsv` | 5~15분 (jobs.tsv 1.5GB 포함) | |
| 3 | 직무명 1차 매핑 | 로컬 CPU | `python data_gen/v4/map_kaggle_titles.py` | title_freq → 저장소 밖 `title_map.csv`, `title_review_200.csv`, 저장소 안 `title_map_summary.json` | 1분 | auto ≥ 0.60 / review 0.35~0.60 / none |
| 4 | 매핑 검수 200건 | 사람 | 시트의 `human_ok` 열에 y/n, 틀리면 `human_job_title_en` 기입 → `python data_gen/v4/map_kaggle_titles.py --score-review <시트>` | 검수 일치율 → summary | 1~2시간 | 일치율을 그대로 보고 |
| 5 | 시뮬레이터 캘리브레이션 | 로컬 CPU | `python data_gen/v4/calibrate_simulator.py --target data_gen/spec/kaggle_target_dist.json --jobs 4` | 18설정 × 약 4분 (병렬 4 → 약 20분) → `data_gen/v4/out/calib/table.md`, `data_gen/spec/v4_params.json` | 20~70분 | 선택 규칙 고정: G-L 통과 후보 중 G3 점수 최소 |
| 6 | v4 생성 | 로컬 CPU | `JOBS_OUT_DIR=data_gen/v4/out/v4 python data_gen/simulate_interactions.py <v4_params 의 command>` → `JOBS_OUT_DIR=... python data_gen/build_splits.py` | v3 profiles·text·graph + 새 상호작용 → `Jobs_split_A.pkl`, `Jobs_split_B.pkl`, `interactions.csv` | 5분 | |
| 7 | G-L · G3 판정 | 로컬 CPU | `JOBS_OUT_DIR=... python data_gen/gate_learnability.py` · `python data_gen/v4/gate_g3_stats.py --interactions ... --profiles data_gen/out/profiles.csv --target ... --md docs/G3_통계정합성.md` | 판정표 | 1분 | **G-L 실패면 GPU 제출 금지** |
| 8 | 매니페스트 | 로컬 CPU | `python data_gen/v4/dataset_manifest.py --version v4 --dir data_gen/v4/out/v4 --params-json data_gen/spec/v4_params.json` | `data_gen/dataset_manifest.json` | 초 | |
| 9 | 서버 업로드 | 로컬 → 서버 | `scp -P 37220` 로 `~/recsys/jobs_v4/` 에 5개 파일 (run_jobs_v4.sh 머리말) | | 5분 | |
| 10 | prep | 서버 GPU·CPU | `bash run_jobs_v4.sh prep` → 안내대로 G-SID 생성·SID 3종·TFRecord | 임베딩 4분, SID 3×4분 | 약 20분 GPU | SID 유일률 확인 (v3: 99.9%) |
| 11 | train | 서버 GPU | `bash run_jobs_v4.sh train` | 9셀 (text·gsid·rewired × 42·7·13) + 덤프 | 37 GPU시간 | |
| 12 | 채점·판정 | 서버 CPU | `bash score_jobs_v4.sh` · `JOBS_VER=v4 SEEDS=42,7,13 python verdict_v3.py` | `results_jobs_v4/*.json`, 판정표 | 30분 | 사전등록 판정 규칙 + 짝지은 검정 + G1 |
| 13 | seeds6 | 서버 GPU | `bash run_jobs_v4.sh seeds6` | text·gsid × 3·21·99 | 25 GPU시간 | n=6 주지표 검정 (t > 2.015) |
| 14 | G4 인간 검수 | 로컬 + 사람 | `python data_gen/v4/make_g4_sheet.py` → 팀원 3명 → `--score answers_*.csv` | `docs/G4_인간검수.md` | 시트 1분, 검수 30분/인 | 임계값 없음, 수치 보고 |
| 15 | G2 시뮬레이터 격리 | 서버 GPU | **스크립트 미작성** (아래 4절) | Beauty 아이템·텍스트·그래프 + 우리 시뮬레이터 상호작용 → 2셀 | 8.5 GPU시간 | 결론 방향 재현 |
| 16 | 문서 | 로컬 | `JOBS_결과정리.md` 에 v4 절, 노션 갱신 | | | |

경로 오버라이드: `JOBS_IN_DIR`(프로필·텍스트·그래프 위치, 기본 `data_gen/out`)와 `JOBS_OUT_DIR`(상호작용·pkl 위치)로 v3 파일을 건드리지 않고 v4 를 만든다. 환경변수가 없으면 v3 과 완전히 같은 동작이다(오늘 재실행으로 199,851행 동일 확인).

## 2. GPU 사용 계획

전제: QOS 동시 2장, 셀당 30,000스텝 약 3시간 40분 + 덤프 27분(구인구직 v3 실측). 학과 서버 규칙(`SERVER_공동사용.md`)대로 작업 이름에 이름을 넣는다(`ME=이름 bash run_jobs_v4.sh ...`).

| 묶음 | 셀 | GPU시간 | 2장 벽시계 | 언제 | 없으면 |
|---|---|---|---|---|---|
| prep (임베딩·SID 3종) | 4 작업 | 0.3 | 20분 | 9/16 (P1+P3 뒤에 큐) | |
| train (본 실험 + G1) | 9 + 덤프 9 | 37 | 약 21시간 (5웨이브) | 9/16 밤 ~ 9/17 | H1 v4 재현 불가 |
| seeds6 (n=6) | 6 + 덤프 6 | 25 | 약 13시간 (3웨이브) | 9/17 ~ 9/18 | 주지표 NDCG@10 은 미확인 유지 |
| G2 | 2 + 덤프 2 | 8.5 | 약 4시간 | 9/21 ~ 9/22 | 생성 데이터 반론에 답 못 함 |
| 합계 (필수) | | **약 71** | **약 36시간** | | |
| 선택: v4 Temporal (W4·W5 × text·gsid × 시드 1) | 4 + 덤프 4 | 16 | 8시간 | 10월 여유 시 | 두 도메인 세 트랙 주장 불가 |

실험 트랙(노션 1번 페이지, P1~P5 약 125 GPU시간)과의 순서 (9/16 개정, 노션 1-1 페이지):

- 1순위 P1+P3 12셀(9/15 밤 제출) → 2순위 v4 prep+train → 3순위 v4 seeds6 → 4순위 P2 Temporal 12셀 → 5순위 P4·P5·G2. Slurm 큐에 이 순서로 넣는다.
- P2 제출·채점·판정 스크립트는 9/16 에 작성해 서버에 올렸다 (`code/run_p2_temporal.sh` · `score_p2_temporal.sh` · `finish_p2_temporal.sh` · `verdict_p2_temporal.py`). 같은 레시피로 이미 돌아 있는 4셀은 재사용하고 8셀만 새로 돈다. 제출은 seeds6 뒤 (`AFTER=<jid> bash run_p2_temporal.sh`).
- 실험 트랙의 P6(구인구직 Temporal)은 위가 다 끝난 뒤 여유 시.
- 셀이 3시간 40분보다 오래 걸리면 seeds6 를 3·21 두 시드(4셀)로 줄인다. P3 와 같은 규칙이다.
- 로컬 RTX 4060 Ti(8GB)로 flan-t5-xl 인코더를 fp16 으로 돌리면 임베딩은 로컬에서도 가능하다(인코더 1.22B, 약 2.4GB). 다만 서버는 fp32 라 수치가 미세하게 달라지므로 **연구 파이프라인은 서버 fp32 로 통일**하고, 로컬은 행사 서비스(3번 계획)의 SID 발급 대비책으로만 둔다.

## 3. 일정 (2026-09-16 개정 — 두 트랙 통합)

9/15 밤 서버 확인 결과 실험 트랙(P1~P5)이 시작되지 않았고 GPU 2장이 비어 있어, 10월로 잡았던 v4 GPU 일정을 9월로 당기고 실험 트랙과 한 줄로 합쳤다. 제출 기록은 노션 1-1 페이지와 `code/run_gsid_week.sh`.

| 날짜 | 작업 | GPU |
|---|---|---|
| 9/15 | 스크립트·허용구간·사전등록 정정·라이선스 표·서버 스크립트. 캐글 분포 산출, 직무명 매핑, G4 시트. **P1+P3 12셀 제출 (23:42)** | 47h (P1+P3) |
| 9/16 | 캘리브레이션 격자 → v4 생성 → G-L·G3 → 매니페스트 → 서버 업로드. P1+P3 종료 후 v4 prep+train 9셀 제출 | 37h (v4) |
| 9/17 | P1·P3 판정표 확인(자동 보고서). v4 seeds6 6셀 제출 | 25h |
| 9/18 | v4 채점·판정(G1 포함). P2 Temporal 12셀 제출 | 44h |
| 9/19~9/20 | v4 n=6 판정표, `JOBS_결과정리.md` v4 절. P2 채점 | |
| 9/21~9/22 | P4·P5, G2 2셀. 매핑 검수 200건, G4 검수 수집 | 40h |
| 9/23 | 버퍼·재실행. `SHARE_GSID_결과요약.md` 갱신, 발표 표 교체 | |
| 9/24~9/30 | 2차 발표 (v3 결과 + v4 재현 여부 + P1~P5) | 0 |
| 10/1~10/24 | G4 집계, v4b 여부 결정, 선택(v4 Temporal), 행사 서비스 SID 배치 발급 | 16h |

원래 10월 일정(1절 표의 근거였던 노션 2번 페이지 단계 5~7 기한)은 유지하되 v4 GPU 부분만 앞당겼다. 매핑 검수와 G4 인간 검수는 GPU와 무관하므로 원래대로 둔다.

## 4. 아직 없는 것 (정직하게)

| 항목 | 상태 | 계획 |
|---|---|---|
| G2 시뮬레이터 격리 스크립트 | 미작성 | Beauty 아이템에 직업 범주가 없어 시뮬레이터의 목표 직업 대신 **텍스트 임베딩 k-means 군집(492개)** 을 범주로, 77차원 스킬 벡터 대신 **임베딩 PCA 77차원**을 잠재 벡터로 쓴다. 같은 코드 경로(`simulate_interactions.py`)에 입력만 바꿔 넣는다. 로컬에 torch 가 없어 임베딩 로드는 서버에서 한다. 10/7 |
| 캐글 규칙 본문 재확인 | 로그인 필요라 오늘 못 함 | 다운로드하는 사람이 캡처 |
| 학과정보 API 이용허락 유형 | 미확인 | 9/23 |
| 매핑 사전(SYNONYM) | 미국 상위 직무 150개 손으로 작성 | 검수 결과로 보강. 가짜 데이터 시험에서 일반 직무명(machine operator 등)이 구체 직업(Laundry Machine Operator)에 붙는 경우가 있어 검수가 필수다 |

## 5. v4 판정 규칙 (v3 와 동일, 실행 전 고정)

| 주장 | 통과 조건 | 못 하면 |
|---|---|---|
| H1 v4 재현 (G-SID > 텍스트) | 3시드 부호 일치 AND 최소 효과 > 노이즈 바닥 (사전등록 방식), 짝지은 t 도 함께 보고 | v3 결과만 보고하고 v4 는 부정 결과로 기록 |
| 주지표 NDCG@10 | n=6 짝지은 t > 2.015 | 미확인 유지 |
| G1 | rewired 에서 이득 소멸 (7/7 기준 동일) | 결과 폐기 |
| G3 | FAIL 0개 | WARN·FAIL 을 기록하고 G-L 우선 원칙으로 설명 |
| G4 | 임계값 없음 | 수치 그대로 |
| v3 vs v4 차이 | 주장하지 않는다. 데이터가 다르므로 비교 불가 | |

## 6. 위험

| 위험 | 대비 |
|---|---|
| 캐글 목표(쏠림 낮음)와 G-L(협업 신호 필요)이 충돌 | 선택 규칙에서 G-L 이 우선. v3 §6 의 "합의 = 쏠림" 제약을 문서에 남긴다 |
| 서버 접속 불가(오늘처럼 타임아웃) | 교내망 또는 VPN 에서 접속. 로컬 CPU 단계는 전부 서버 없이 진행 가능 |
| 두 트랙이 같은 GPU 2장을 쓴다 | 2절 순서대로 큐에 넣는다. 셀이 3시간 40분보다 오래 걸리면 P4 부터 줄인다 |
| jobs.tsv(1.5GB) 파싱 실패 | `--skip-jobs` 로 나머지 통계는 산출, 직무명은 user_history 만으로 매핑 |
| 매핑 커버리지 낮음 | 분포 목표는 직무명과 무관한 통계 우선. 매핑은 G4 어휘 정렬용 |

## 7. 관련 파일

```
data_gen/v4/kaggle_stats.py          캐글 → 목표 분포(집계만)
data_gen/v4/map_kaggle_titles.py     직무명 매핑 + 검수 시트
data_gen/v4/gate_g3_stats.py         G3 게이트 (Beauty 참고선 생성 겸용)
data_gen/v4/calibrate_simulator.py   격자 캘리브레이션 + 선택 규칙
data_gen/v4/make_g4_sheet.py         G4 블라인드 시트 + 채점
data_gen/v4/dataset_manifest.py      해시 매니페스트
data_gen/spec/g3_tolerance.json      허용 구간 (docs/G3_허용구간.md 와 동일)
data_gen/spec/beauty_ref_dist.json   Beauty 참고선
data_gen/spec/v3_params.json         v3 파라미터 기록
code/run_jobs_v4.sh · score_jobs_v4.sh · code/server/embed_jobs.sh · verdict_v3.py(JOBS_VER 환경변수)
code/run_p2_temporal.sh · score_p2_temporal.sh · finish_p2_temporal.sh · verdict_p2_temporal.py   P2 Temporal (실험 트랙)
docs/G3_허용구간.md · LICENSE_데이터출처.md · PREREG_생성데이터셋.md §9
```
