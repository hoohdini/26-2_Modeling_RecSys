# DSL GPU 서버(Slurm) 사용 가이드 — 추천시스템팀 (dsl05)

> 🔑 **비밀번호는 이 문서에 없습니다.** 팀 내부 채널에서 받으세요.
> 접속 주소·계정명·포트는 들어 있으니 저장소 밖으로 공유하지 마세요.
> (이 저장소는 private 입니다.)
> 공식 가이드: https://yonseiserver.github.io/blog/2026/server-usage/
> (하위 문서: [SSH·Job 실행](https://yonseiserver.github.io/blog/2025/ssh/), [Python 실행](https://yonseiserver.github.io/blog/2025/python/), [Slurm job 설정기](https://yonseiserver.github.io/slurm-job/))

---

## 1. 접속하기

우리 팀 계정은 **dsl05** (추천시스템팀)입니다.

| 서버 | IP | 용도 |
|---|---|---|
| Slurm Master Node | 165.132.80.36 | 작업 제출용 (기본 접속처) |
| hpc | 165.132.77.48 | 계산 노드 (파티션 `jobs`) |
| hpc-stat1 | 165.132.80.37 | 계산 노드 (파티션 `partition1`) |

포트는 모두 **37220**입니다.

```bash
ssh -p 37220 dsl05@165.132.80.36
```

- 비밀번호: **이 문서에 적지 않습니다.** 팀 내부 채널(노션 비공개 페이지 등)에서 받으세요.
  2026-08-20 현재 초기 비밀번호를 아직 변경하지 않았습니다(팀 합의). 나중에 팀이 합의한 시점에
  `passwd` 로 바꾸고, 바뀐 비밀번호는 안전한 채널(카톡 X, 노션 비공개 페이지 등)로 공유하세요.
- VSCode의 Remote-SSH 확장을 쓰면 서버 파일을 로컬처럼 편집할 수 있어 편합니다.

매번 비밀번호 입력이 귀찮으면 SSH 키를 등록하세요:
```bash
ssh-keygen -t ed25519          # 이미 키가 있으면 생략
ssh-copy-id -p 37220 dsl05@165.132.80.36     # 맥/리눅스
```
윈도우에는 `ssh-copy-id` 가 없으니 한 줄로 대신합니다 (비밀번호 1회 입력):
```powershell
ssh -p 37220 dsl05@165.132.80.36 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '<id_ed25519.pub 내용>' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```
> `Host key verification failed` 가 나면 서버 호스트 키가 등록 안 된 것입니다.
> `ssh-keyscan -p 37220 165.132.80.36 >> ~/.ssh/known_hosts` 로 먼저 등록하세요.
> (데스크탑에는 2026-08-19에 키 등록 완료 — 이제 비밀번호 없이 접속됩니다.)

## 2. 저장공간 규칙

> ⚠️ **2026-08-19 실측으로 정정**: 공유 저장소는 **마스터 노드와 계산 노드에서 마운트 경로가 다릅니다.**
> 같은 물리 볼륨인데 경로만 다르므로, 스크립트를 어디서 실행하느냐에 따라 경로를 바꿔야 합니다.

| 실행 위치 | 공유 저장소 경로 |
|---|---|
| 마스터 노드(`hpcmaster`, ssh로 접속하는 곳) | `/data1/dsl05` — **홈 디렉토리가 곧 이 경로** |
| 계산 노드(`hpc`, `hpc-stat1` 등, sbatch/srun이 도는 곳) | `/mnt/data1/dsl05` (`/data1/dsl05`는 **없음**) |

즉 **sbatch 스크립트 안에서는 반드시 `/mnt/data1/dsl05`** 를 써야 합니다. `#SBATCH --output=` 경로도
마찬가지라, `/data1/...` 로 써 두면 슬럼이 로그 파일조차 못 만들고 **로그 없이 즉시 FAILED** 납니다.
(`--chdir` 도 지정하지 않으면 계산 노드가 `$HOME`(=`/data1/dsl05`)로 이동하려다 실패해 `/tmp`로 떨어집니다.)

우리 프로젝트 실제 배치 (마스터 기준 경로로 표기):
```
/data1/dsl05/            (계산 노드에서는 /mnt/data1/dsl05)
├── miniconda3/
│   └── envs/grid/       # python 3.10 + torch 2.6.0+cu124
├── hf_home/             # HuggingFace 캐시 (flan-t5-xl 11GB) — 다운로드 완료
├── recsys/
│   ├── GRID/            # GRID 코드 (.project-root 포함)
│   ├── grid_data/beauty_A/   # TFRecord (items 12,101 / train·eval·test 각 22,363)
│   ├── embed_beauty.sh       # 임베딩 추출 sbatch 스크립트
│   ├── server_setup.sh       # 환경 구축 스크립트
│   ├── prefetch_model.sh     # 모델 사전 다운로드
│   └── logs/                 # slurm 로그
```

## 3. 처음 한 번만: 환경 설정

> **dsl05 계정에는 2026-08-19에 이미 구축 완료.** 아래는 재현/타 계정용 기록입니다.
> 자동화 스크립트는 레포의 `code/server/server_setup.sh` 에 있습니다 (마스터 노드에서 실행).

### 3-1. Miniconda + GRID 환경

```bash
# 마스터 노드에서 실행 (홈 = /data1/$USER)
mkdir -p ~/downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/downloads/miniconda.sh
bash ~/downloads/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -y --prefix $HOME/miniconda3/envs/grid python=3.10
conda activate $HOME/miniconda3/envs/grid

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install lightning==2.5.0 transformers==4.47.0 sentencepiece tokenizers \
  hydra-core==1.3.2 hydra-colorlog omegaconf rootutils python-dotenv rich \
  pandas pyarrow tensorflow-cpu==2.18.0 google-cloud-bigquery psutil "protobuf<5"
pip install "cryptography==42.0.8"     # ★ 3-2 참고. 반드시 이 버전
```

설치 결과: torch 2.6.0+cu124 / transformers 4.47.0 / lightning 2.5.0 / tensorflow-cpu 2.18.0.

### 3-2. ★ 반드시 알아야 할 두 가지 (마스터 ≠ 계산 노드)

**(가) OS·glibc 가 다릅니다.**

| 노드 | OS | glibc |
|---|---|---|
| 마스터 `hpcmaster` | Ubuntu 22.04 | 2.35 |
| 계산 `hpc`, `hpc-stat1` | Ubuntu 20.04 | **2.31** |

pip는 *설치하는 기계*(마스터, glibc 2.35) 기준으로 최신 휠을 고르기 때문에, 계산 노드에서
`GLIBC_2.33 not found` 로 죽는 패키지가 생깁니다. 실제로 `cryptography`(google-cloud-bigquery 의존)가
여기 걸렸습니다 → **`cryptography==42.0.8`** 로 고정하면 해결됩니다(manylinux_2_28 휠).
다른 패키지에서 같은 증상이 나면 동일하게 구버전으로 내리세요.

**(나) `conda activate` 가 계산 노드에서 깨집니다.**

conda를 마스터의 `/data1/$USER/miniconda3` 로 설치하면 내부 경로가 `/data1/...` 로 박히는데
계산 노드엔 그 경로가 없습니다. 그래서 **sbatch 스크립트에서는 `conda activate` 를 쓰지 말고
python 절대경로로 직접 호출**하세요:

```bash
PY=/mnt/data1/dsl05/miniconda3/envs/grid/bin/python
$PY -m src.inference ...
```
(python 바이너리는 자기 실행 경로로 sys.prefix를 잡으므로 `/mnt/data1` 경로로 부르면 정상 동작합니다.)

### 3-3. 모델 사전 다운로드

계산 노드는 인터넷이 없을 수 있으므로 **마스터에서 미리 받아** HF 캐시에 넣어둡니다
(`code/server/prefetch_model.sh`). flan-t5-xl 11GB, 약 16분 소요.

```bash
export HF_HOME=$HOME/hf_home
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/flan-t5-xl')"
```
작업 스크립트에서는 `HF_HOME` 지정 + `HF_HUB_OFFLINE=1` 로 캐시만 쓰게 합니다.

### 3-4. 데이터 업로드 (윈도우 데스크탑에서)

```powershell
tar -czf grid_bundle.tgz --exclude=notices.txt GRID grid_data
scp -P 37220 grid_bundle.tgz dsl05@165.132.80.36:~/recsys/
ssh -p 37220 dsl05@165.132.80.36 "cd ~/recsys && tar -xzf grid_bundle.tgz && touch GRID/.project-root"
```

## 4. Slurm으로 작업 실행

**원칙: GPU 작업은 서버에서 직접 실행하지 말고 sbatch로 제출.**
간단한 테스트/확인만 `srun ... bash -c "..."` 로.

자주 쓰는 명령어:

| 명령어 | 뜻 |
|---|---|
| `sbatch <script.sh>` | 작업 제출 |
| `squeue -u $USER` | 내 작업 확인 (`PD`=대기, `R`=실행 중) |
| `sacct -j <JobID> --format=JobID,State,ExitCode,Elapsed` | 끝난 작업 결과 확인 (**실패 추적에 필수**) |
| `scancel <JobID>` | 작업 취소 |
| `sinfo -o "%P %n %G %e %C %t"` | 파티션·노드별 GPU/메모리/CPU/상태 |
| `scontrol show node <노드>` | 그 노드의 GPU 할당 현황 (`AllocTRES`에 `gres/gpu`가 없으면 GPU가 놀고 있음) |

### 실측 파티션 현황 (2026-08-19)

| 파티션 | 노드 | GPU | 우리 계정(dsl05) |
|---|---|---|---|
| `jobs` | hpc | 2장 | 사용 가능 |
| `partition1` | hpc-stat1 | 5장 (**RTX 6000 Ada 48GB**) | 사용 가능 — **우리가 쓴 곳** |
| `brl` | brl0~brl5 | 각 8장 | 불가 (`brl_qos` 필요, 다른 랩 전용·대기열 김) |
| `gpu` | gpu01 | 1장 | 불가 (`down` 상태) |

우리 QOS(`normal`) 한도: **cpu=32, gpu=2**.

## 5. 우리 작업: GRID 임베딩 추출 sbatch 스크립트

실제로 성공한 스크립트는 레포의 **`code/server/embed_beauty.sh`** 이며, 서버에도
`/mnt/data1/dsl05/recsys/embed_beauty.sh` 로 올라가 있습니다. 핵심만 옮기면:

```bash
#!/bin/bash
#SBATCH --job-name=grid_embed_beauty
#SBATCH --partition=partition1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/data1/dsl05/recsys/GRID
#SBATCH --output=/mnt/data1/dsl05/recsys/logs/embed_%j.log
#SBATCH --error=/mnt/data1/dsl05/recsys/logs/embed_%j.err

set -euo pipefail
BASE=/mnt/data1/dsl05
PY=$BASE/miniconda3/envs/grid/bin/python        # conda activate 쓰지 말 것 (3-2 나)

export HF_HOME=$BASE/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd $BASE/recsys/GRID
$PY -m src.inference \
  experiment=sem_embeds_inference_flat \
  data_dir=$BASE/recsys/grid_data/beauty_A \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  data_loading.datamodule.predict_dataloader_config.batch_size_per_device=64 \
  data_loading.datamodule.predict_dataloader_config.num_workers=0 \
  data_loading.datamodule.predict_dataloader_config.persistent_workers=false \
  data_loading.datamodule.predict_dataloader_config.timeout=0
```

제출·확인:
```bash
sbatch ~/recsys/embed_beauty.sh
squeue -u $USER
sacct -j <JobID> --format=JobID,State,ExitCode,Elapsed
tail -f ~/recsys/logs/embed_<JobID>.log
```

**실측 성능**: RTX 6000 Ada 1장, batch 64 → 190 배치 **3분 50초** (12,101 아이템 전체).
참고로 인텔 맥북 CPU 추정치는 6.5시간이었습니다. VRAM은 fp32로 충분(48GB 중 일부만 사용).

결과 회수 (윈도우에서):
```powershell
scp -P 37220 "dsl05@165.132.80.36:/data1/dsl05/recsys/GRID/logs/inference/runs/<날짜>/<시간>/pickle/merged_predictions_tensor.pt" "D:\DSL\RecSys\embeddings\beauty_A\"
```
> scp는 **마스터 노드**에 붙으므로 여기서는 `/data1/...` 경로를 씁니다 (계산 노드 경로 `/mnt/data1` 아님).

## 6. 우리가 이미 밟은 함정들 (전부 실제로 겪은 것)

| # | 증상 | 원인 / 해결 |
|---|---|---|
| 1 | `Project root directory not found` | GRID 루트에 빈 `.project-root` 파일 필요 (`touch .project-root`) |
| 2 | 제출 즉시 FAILED, **로그 파일조차 없음** (ExitCode 53) | `#SBATCH --output` 경로를 `/data1/...` 로 씀. 계산 노드에 없는 경로 → **`/mnt/data1/...`** 로 (2절) |
| 3 | `GLIBC_2.33 not found` (cryptography) | 마스터(22.04)에서 받은 휠이 계산 노드(20.04)에서 안 돎 → `cryptography==42.0.8` 고정 (3-2 가) |
| 4 | `RuntimeError: generator raised StopIteration` | `items/` TFRecord가 **파일 1개**인데 `num_workers=2` 라 워커 하나가 빈 이터레이터를 받음 → **`num_workers=0`** (+ `timeout=0`, `persistent_workers=false` 세트) |
| 5 | 추론은 다 끝났는데 마지막에 `Default process group has not been initialized` | **GRID 코드 버그**. `src/utils/inference_utils.py` 의 `on_predict_end` 가 `if trainer.global_rank != None:` 로 검사 → rank 0에서도 참이라 단일 프로세스인데 `torch.distributed.barrier()` 호출. `if torch.distributed.is_available() and torch.distributed.is_initialized():` 로 수정(3곳). **패치 적용 완료** |
| 6 | macOS `._*` 파일 때문에 TFRecord 읽기 실패 | 업로드 전에 `find . -name "._*" -delete` |
| 7 | 모델은 flan-t5-xl(3B) 고정 | XXL(11B)은 팀 계획상 금지. 실제 로드는 인코더만 1.22B |
| 8 | TFRecord는 GZIP 압축이어야 GRID가 읽음 | `to_grid_v2.py` 가 그렇게 생성함 |

## 7. 문의

막히면 학과 서버 Slack의 `help` 채널에 서버명·명령어·에러 메시지를 포함해 질문.
