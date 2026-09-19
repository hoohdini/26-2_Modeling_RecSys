# 학과 GPU 서버 공동 사용 — 팀원이 들어오기 전에 꼭 읽을 것

작성 2026-08-19 · 확인 완료 사항만 적었습니다

---

## 1. 가장 중요한 것: **계정 하나를 팀 전체가 같이 씁니다**

```
uid=5005(dsl05) gid=5005(dsl05)
```

`dsl05` 는 **개인 계정이 아니라 추천시스템팀 공용 계정**입니다. 팀원 누가 접속하든
**같은 유저, 같은 홈 디렉토리, 같은 파일**입니다. 그래서:

### ✅ 좋은 점 — 세팅이 이미 끝나 있습니다

들어오자마자 바로 실험을 돌릴 수 있습니다. 설치할 게 없습니다.

| 이미 준비된 것 | 위치 | 크기 |
|---|---|---|
| conda 환경 (python 3.10, torch 2.6.0+cu124, GRID 의존성 전부) | `~/miniconda3/envs/grid` | 7.8GB |
| flan-t5-xl 모델 캐시 (재다운로드 불필요) | `~/hf_home` | 11GB |
| GRID 코드 (팀 패치 2건 반영, `.project-root` 포함) | `~/recsys/GRID` | — |
| GRID 입력 TFRecord (items 12,101 / train·eval·test) | `~/recsys/grid_data/beauty_A` | — |
| **아이템 임베딩 (12,101 × 2048)** | `~/recsys/embeddings/beauty_A/` | 95MB |
| **baseline SID (3단계·4단계)** | `~/recsys/sid_out/baseline/L3`, `L4` | — |
| 실행 스크립트 | `~/recsys/embed_beauty.sh`, `~/recsys/sid_beauty.sh` | — |

**즉 "내가 만든 데이터에서 이어서 진행"이 맞습니다.** 임베딩을 다시 뽑을 필요도, 모델을
다시 받을 필요도 없습니다.

### ⚠️ 위험한 점 — 서로의 결과를 지울 수 있습니다

권한 분리가 **없습니다.** 팀원 누구나 남의 파일을 지우고 덮어쓸 수 있고,
남의 Slurm 작업을 취소할 수도 있습니다. 아래 규칙을 지켜주세요.

---

## 2. 지켜야 할 규칙 5가지

### ① 작업은 **본인 폴더** 안에서

```
~/recsys/work/<본인이름>/       ← 여기서 작업하세요 (이미 만들어 뒀습니다)
```

`~/recsys/` 바로 아래나 `~/recsys/GRID/` 안에 결과물을 흩뿌리지 마세요.

### ② 공유 입력물은 **읽기 전용으로 잠가 놨습니다**

```
~/recsys/embeddings/     (r--r--r--)
~/recsys/grid_data/      (r-xr-xr-x)
~/recsys/sid_out/baseline/  (r-xr-xr-x)
```

실수로 덮어쓰는 걸 막으려는 것입니다. **같은 계정이라 마음먹으면 `chmod` 로 풀 수 있지만,
푸는 순간 그건 의도적인 행동입니다.** 정말 바꿔야 하면 팀에 먼저 말해주세요.

### ③ SID를 만들 때는 **`TAG` 를 반드시 지정**

`sid_beauty.sh` 는 기본값 `TAG=baseline` 으로 `sid_out/baseline/` 에 씁니다.
그대로 돌리면 **기존 baseline 과 충돌**합니다 (지금은 덮어쓰기를 거부하도록 막아 놨습니다).

```bash
sbatch --export=ALL,TAG=gsid_홍길동,EMB=/mnt/data1/dsl05/recsys/work/홍길동/my_emb.pt,DIM=512 \
       --job-name=sid_홍길동 \
       ~/recsys/sid_beauty.sh
```

| 변수 | 뜻 | 기본값 |
|---|---|---|
| `TAG` | 출력 폴더 이름 → `sid_out/$TAG/L{3,4}/` | `baseline` |
| `EMB` | 입력 임베딩 `.pt` 경로 | 텍스트 임베딩 |
| `DIM` | 임베딩 차원 | 2048 |
| `LEVELS` | 만들 레벨 목록 | `"3 4"` |
| `WIDTH` | 레벨당 코드북 크기 | 256 |
| `FORCE=1` | 기존 출력 폴더를 지우고 재생성 | 0 (거부) |

> 그래프 SID·CRAB SID 담당자는 **`(12101 × D)` 텐서 하나만 만들면** `EMB`/`DIM` 만 바꿔서
> baseline 과 완전히 동일한 조건으로 SID를 만들 수 있습니다.

### ④ 작업 이름에 **본인 이름을 넣으세요**

`squeue -u dsl05` 를 하면 **팀원 전원의 작업이 섞여서** 나옵니다. 이름이 없으면 누구 건지
알 수 없고, 실수로 남의 작업을 `scancel` 하게 됩니다.

```bash
#SBATCH --job-name=tiger_홍길동
# 또는
sbatch --job-name=tiger_홍길동 ...
```

**본인 작업이 확실하지 않으면 `scancel` 하지 마세요.**

### ⑤ conda 환경에 **함부로 `pip install` 하지 마세요**

`~/miniconda3/envs/grid` 는 **모두가 같이 쓰는 환경**입니다. 한 사람이 패키지를 올리면
전원의 실험이 깨질 수 있습니다. 특히:

> **`cryptography` 는 반드시 `42.0.8` 이어야 합니다.** 최신 버전을 깔면 계산 노드에서
> `GLIBC_2.33 not found` 로 **모든 작업이 죽습니다** (마스터는 Ubuntu 22.04/glibc 2.35,
> 계산 노드는 20.04/glibc 2.31이라 그렇습니다).

패키지가 더 필요하면 환경을 복제해서 쓰세요:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create --clone ~/miniconda3/envs/grid --prefix ~/recsys/work/<이름>/env
```

---

## 3. GPU 자원 — 계정 전체가 **2장**을 나눠 씁니다

| 항목 | 값 |
|---|---|
| QOS(`normal`) 한도 | **cpu=32, gpu=2** — 계정 전체 합산, **동시 사용량** 기준 |
| 누적 GPU-시간 총량 한도 | **없음** (`GrpTRESMins` 미설정) |
| 작업당 최대 시간 | **없음** (`MaxWall` 미설정) |
| 쓸 수 있는 파티션 | `partition1` (hpc-stat1, **RTX 6000 Ada 48GB** 5장), `jobs` (hpc, 2장) |
| 못 쓰는 파티션 | `brl` (다른 랩 전용 QOS), `gpu` (down) |

> ✅ **오래 돌린다고 소진되는 "할당량"은 없습니다.** 한도는 "한 번에 몇 개까지"이지
> "총 몇 시간까지"가 아닙니다. 참고로 2026-08 기준 계정별 누적 사용 실적은
> dsl05가 161인 반면 dsl07은 14,259, dsl04는 7,750으로 **우리는 거의 안 쓴 편**입니다.
> `#SBATCH --time` 값은 스크립트에 넣은 안전장치일 뿐이라 필요하면 늘려도 됩니다.

**실질적인 제약은 팀원과의 경합입니다.** 두 명이 각각 GPU 1장씩 잡으면 계정이 꽉 차고,
세 번째 사람의 작업은 대기(PD)로 갑니다. 나중에 4x2 실험 그리드를 돌릴 때가 진짜 병목입니다
(8칸을 동시에 2개씩밖에 못 돌립니다). 긴 작업을 돌리기 전에 팀에 공유해주세요.

확인 명령:
```bash
sacctmgr -n show qos where name=normal format=Name,GrpTRESMins%20,MaxTRESPU%25,MaxWall
squeue -u dsl05 -o "%.8i %.20j %.2t %.10M %b %C"      # 지금 팀이 쓰는 양
scontrol show node hpc-stat1 | grep -E "CfgTRES|AllocTRES"   # 노드 여유
```

참고 소요 시간 (RTX 6000 Ada 1장):

| 작업 | 시간 |
|---|---|
| 임베딩 추출 (12,101 아이템) | 3분 50초 |
| SID 생성 (3단계+4단계 합) | 6분 59초 |

임베딩·SID는 **이미 끝나 있으니 다시 돌릴 필요 없습니다.**

---

## 4. 접속 방법

```bash
ssh -p 37220 dsl05@165.132.80.36
```

- 비밀번호는 [`docs/GPU_서버_사용가이드.md`](GPU_서버_사용가이드.md) 에 있습니다
  (이 저장소가 **private 이라 포함**돼 있습니다 — 🔒 외부에 붙여넣지 마세요)
- **2026-08-19 현재 초기 비밀번호를 아직 변경하지 않았습니다** (팀 합의). 변경 시 전원 공지 필요.
- 현재 SSH 키가 등록된 기기는 **1대뿐**입니다. 본인 기기를 등록하려면:

```bash
ssh-copy-id -p 37220 dsl05@165.132.80.36          # 맥/리눅스
```

윈도우는 `ssh-copy-id` 가 없으니 한 줄로 대신합니다 (비밀번호 1회 입력):

```powershell
ssh -p 37220 dsl05@165.132.80.36 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '<본인 id_ed25519.pub 내용>' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

> `Host key verification failed` 가 나면 먼저:
> `ssh-keyscan -p 37220 165.132.80.36 >> ~/.ssh/known_hosts`

---

## 5. 경로 함정 — 이거 모르면 무조건 한 번 당합니다

**공유 저장소의 마운트 경로가 노드마다 다릅니다.**

| 어디서 | 경로 |
|---|---|
| 마스터 노드 (ssh 로 접속하는 곳) | `/data1/dsl05` = 홈 디렉토리 |
| 계산 노드 (sbatch/srun 이 도는 곳) | **`/mnt/data1/dsl05`** (`/data1/dsl05` 는 **없음**) |

- **sbatch 스크립트 안은 전부 `/mnt/data1` 기준**으로 쓰세요.
- `#SBATCH --output=` 을 `/data1/...` 로 쓰면 슬럼이 로그 파일조차 못 만들어
  **로그 없이 즉시 FAILED** 납니다 (ExitCode 53). 원인 찾기 아주 어렵습니다.
- `--chdir` 도 `/mnt/data1` 기준으로 지정하세요.
- 반대로 **scp 로 결과를 받을 때는 마스터에 붙으므로 `/data1` 경로**를 씁니다.

또 하나: conda 가 `/data1` prefix 로 설치돼 있어 **계산 노드에서 `conda activate` 가 깨집니다.**
sbatch 안에서는 python 절대경로로 직접 부르세요:

```bash
PY=/mnt/data1/dsl05/miniconda3/envs/grid/bin/python
$PY -m src.train ...
```

---

## 6. 자주 쓰는 명령

| 명령 | 뜻 |
|---|---|
| `sbatch <script.sh>` | 작업 제출 |
| `squeue -u dsl05` | **팀 전체** 작업 확인 (`PD`=대기, `R`=실행) |
| `sacct -j <JobID> --format=JobID,State,ExitCode,Elapsed` | 끝난 작업 결과 — **실패 추적에 필수** |
| `tail -f ~/recsys/logs/<작업>_<JobID>.log` | 로그 실시간 |
| `sinfo -o "%P %n %G %e %C %t"` | 파티션·GPU 현황 |
| `scancel <JobID>` | **본인 작업만** 취소 |

---

## 7. 더 볼 것

- `docs/_archive/TEAM_RESULTS.md` — 지금까지 결과 요약, 결정할 것
- `docs/HANDOFF_graph_sid.md` — 그래프 SID 담당자용
- `docs/TOKENIZE_EMBED_SID_REPORT.md` — 상세 경위·전체 수치
- [`docs/GPU_서버_사용가이드.md`](GPU_서버_사용가이드.md) — 접속 정보 포함 서버 전체 가이드 🔒
