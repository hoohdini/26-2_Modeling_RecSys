# 구인구직 생성 데이터셋

Amazon Beauty 에서 확인한 G-SID(그래프 결합 시맨틱 ID) 효과를 **사람에게 일어나는 일**로
옮겨 검증하기 위해 만든 데이터셋입니다.

```
user = 채용담당자 · item = 후보자 프로필 · cold item = 신규 가입자
```

| | |
|---|---|
| 프로필(아이템) | **12,000** · 492 직업 |
| 담당자(유저) | **20,000** · 채점 가능 18,741 |
| 상호작용 | **199,851** |
| 그래프 | 220,918 간선 |
| 스킬 차원 | 77 (지식 33 + 능력 44) |

---

## 1. 출처 — 무엇을 우리가 만들지 **않았나**

직무 어휘와 직업 분류를 **발명하지 않았습니다.** 고용24(워크넷) 공공 OpenAPI 에서 가져왔습니다.

| 우리가 만든 것 | 공공데이터에서 온 것 |
|---|---|
| 프로필의 조합(경력·학력·지역 배정) | 직업명·직업분류·KECO 코드 |
| 영문 프로필 텍스트 | 지식 33 / 능력 44 **레벨 벡터** |
| 담당자의 열람·컨택 로그 | 관련직업 관계(`relJobList`) |
| | 전공-직업 전이, 학과 분포 |

수집 스크립트는 `fetch_work24.py`, 실측 명세는 `spec/WORK24_API_실측명세.md` 입니다.
인증키는 `.env` 로 분리되어 있고 커밋되지 않습니다.

> ⚠️ **재배포 주의.** 이 저장소는 현재 private 입니다. public 으로 전환한다면
> 고용24 공공데이터의 이용허락 조건(출처표시 등)을 먼저 확인해야 합니다.

**개인정보 없음** — 프로필은 전부 합성입니다. 실명·연락처·식별자가 들어있지 않습니다.
`profiles.csv` 에는 이름 컬럼 자체가 없습니다.

---

## 2. 파일

| 파일 | 크기 | 내용 |
|---|---|---|
| `out/profiles.csv` | 2.3M | 12,000 프로필 속성 (직업·KECO·경력·학력·전공·지역·대표역량 3) |
| `out/profile_skill.npy` | 3.7M | (12000, 77) float32 스킬 레벨 벡터 |
| `out/profile_text.tsv` | 2.9M | 프로필 영문 텍스트 (전부 고유, unk 0.0000%) |
| `out/profile_regtime.npy` | 48K | 프로필 등록 시각 (일). 콜드 스타트를 만든다 |
| `out/skill_dims.txt` | 1K | 77 스킬 차원 이름 |
| `out/jobs_edges_final.csv` | 5.6M | 아이템 그래프 220,918 간선 (θ 적용 후) |
| `out/jobs_edges_long.csv` | 7.2M | 간선 전체 + `relation` 컬럼 — 어느 관계에서 왔는지 |
| `out/jobs_edges_rewired.csv` | 5.8M | **G1 대조군** — 차수 보존 무작위화 |
| `out/interactions.csv` | 6.4M | 199,851 상호작용 (user, position, item, timestamp) |
| `out/jobs_interactions_long.csv` | 3.9M | 상호작용 + `split` 컬럼 — 피클 없이 split 확인용 |
| `out/Jobs_split_A.pkl` | 5.6M | **LOO 트랙** — 학습/평가는 이 파일 위에서만 성립 |
| `out/Jobs_split_B.pkl` | 7.2M | **Temporal 트랙** — W3/W4/W5 윈도우 |

**올리지 않은 것** (`.gitignore`)

| | 이유 |
|---|---|
| `raw/` 2.0GB | 고용24 원본 XML 1,970건 — `fetch_work24.py` 로 재수집 |
| `out/jobs_text_emb.pt` 98MB | 텍스트 임베딩 — `embed_jobs.sh` 로 4분이면 재생성 |
| `tmp/` 22MB | 중간 산출물 |
| `out/v1_failed/` | 폐기한 v1 — [`docs/JOBS_학습실패_원인분석.md`](../docs/JOBS_학습실패_원인분석.md) |

Split 파일은 **재생성 레시피가 아니라 파일 자체를 공유**합니다. 다시 만들면 아이템 인기도가
달라지고 롱테일 집합·콜드 버킷 경계가 다른 아이템 위에 그어져 비교가 깨집니다.
(Beauty split 과 같은 이유 — `.gitignore` 주석 참고)

---

## 3. 재현 경로

```bash
python data_gen/fetch_work24.py                    # 1. 공공데이터 수집 (이어받기 지원)
python data_gen/build_profiles.py                  # 2. 프로필 속성
python data_gen/write_profile_text.py              #    → 영문 텍스트
sbatch embed_jobs.sh                               # 3. 임베딩 (서버 GPU, 약 4분)
python data_gen/build_job_graph.py                 # 4. 그래프 → G0 판정
python data_gen/rewire_null.py                     #    → G1 대조군

python data_gen/simulate_interactions.py \
       --temp 0.15 --gamma 0.20 --cand-pool 12000  # 5. 상호작용
python data_gen/build_splits.py                    #    → Split A/B
python data_gen/gate_learnability.py               # 6. G-L. 통과 전에는 GPU 제출 금지
```

난수는 전부 `--seed` 로 고정됩니다. 같은 시드면 같은 데이터가 나옵니다.

---

## 4. 게이트

| 게이트 | 무엇을 재는가 | 결과 |
|---|---|---|
| **G0** | 그래프와 텍스트의 정보 중첩 | ✅ 통과 (합집합 0.0052, 판정선 0.087) |
| **G-L** | **이 데이터셋에 배울 것이 있는가** | ✅ 통과 |
| G1 | 이득이 간선의 의미에서 오는가 | 진행 중 |
| G2·G3·G4 | 시뮬레이터 격리 · 통계 정합성 · 인간 검수 | 미실시 |

**G-L 은 이 프로젝트에서 추가한 게이트입니다.** (`gate_learnability.py`)
2026-08-31, 이 게이트가 없어서 학습 불가능한 데이터셋에 GPU 13시간을 썼습니다.
`G0` 는 그래프 품질 검사지 상호작용 품질 검사가 아니라 그 실패를 잡지 못했습니다.

```
             recall@10   vs MostPopular
MostPopular    0.02583        1.00x
ContentKNN     0.08276        3.20x
ItemKNN        0.07502        2.90x     (Amazon Beauty: 3.52x)
```

---

## 5. 알려진 한계

1. **순환이 완전히 사라지지 않았습니다.** 그래프(`skill_vector`)와 담당자 점수 함수가
   같은 스킬 벡터에서 나옵니다. 담당자는 요구 차원 20개만 관측 잡음(σ=14)을 통해 보고
   그래프는 전체 77차원을 쓰도록 분리했지만 결합은 남습니다. **크기는 G1 이 잽니다.**
2. **노출 쏠림이 Beauty 와 다릅니다** — 상위 50% 가 컨택의 92.6% (Beauty 82.4%).
   가용한 시뮬레이터 레버로는 내려가지 않습니다. 근거: `docs/JOBS_학습실패_원인분석.md` §6.
3. **채용공고 실데이터가 없습니다** (개인회원 API 차단). 직무 어휘는 직업정보의
   구조화된 지식·능력 벡터로 대체했습니다.
4. **전공 계열이 직업과 어색한 조합이 섞입니다.** 고용24 실제 분포를 그대로 뽑은
   결과이며 조작하지 않았습니다.
5. **담당자의 컨택 순서는 무작위입니다.** 점수 순이 아닙니다(실제 채용도 그렇습니다).
   따라서 순차 신호보다 집합 신호가 주된 학습 대상입니다.

---

## 6. v4 (2026-09-15 시작) — 캐글 분포로 재보정

v3 의 프로필·텍스트·그래프는 그대로 두고 **상호작용만** 캐글 CareerBuilder 2012 실측 분포에 맞춰 다시 만든다.
계획·GPU 일정·판정 규칙은 [`docs/DATASET_v4_파이프라인.md`](../docs/DATASET_v4_파이프라인.md),
허용 구간은 [`docs/G3_허용구간.md`](../docs/G3_허용구간.md), 출처 조건은 [`docs/LICENSE_데이터출처.md`](../docs/LICENSE_데이터출처.md).

```bash
python data_gen/v4/kaggle_stats.py                 # 1. 캐글 → 목표 분포 (원본은 D:/DSL/_external 에, 집계만 저장소에)
python data_gen/v4/map_kaggle_titles.py            # 2. 직무명 → 492 직업·KECO 1차 매핑 + 검수 시트 200건
python data_gen/v4/calibrate_simulator.py --target data_gen/spec/kaggle_target_dist.json --jobs 4
                                                   # 3. temp·gamma·k 격자 → G-L 통과 중 G3 최적 → spec/v4_params.json
JOBS_OUT_DIR=data_gen/v4/out/v4 python data_gen/simulate_interactions.py <v4_params.json 의 command>
JOBS_OUT_DIR=data_gen/v4/out/v4 python data_gen/build_splits.py
JOBS_OUT_DIR=data_gen/v4/out/v4 python data_gen/gate_learnability.py       # 4. G-L (제출 전 필수)
python data_gen/v4/gate_g3_stats.py --interactions data_gen/v4/out/v4/interactions.csv        --profiles data_gen/out/profiles.csv --target data_gen/spec/kaggle_target_dist.json --md docs/G3_통계정합성.md
python data_gen/v4/dataset_manifest.py --version v4 --dir data_gen/v4/out/v4 --params-json data_gen/spec/v4_params.json
```

`JOBS_IN_DIR` / `JOBS_OUT_DIR` 환경변수가 없으면 v3 스크립트는 예전과 완전히 같이 동작한다.
