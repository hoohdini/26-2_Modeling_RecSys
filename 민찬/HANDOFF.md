# 🔄 HANDOFF — 새 세션은 이 파일부터 읽으세요

> **최종 갱신:** 2026-08-09 (새벽 작업)
> **사용자:** 민찬님 (연세대 DSL 랩)
> **작업 폴더:** `/Users/k_minchan/Desktop/YONSEI/DSL/모델링 프로젝트/`

---

## 0. 30초 요약

```
누가:   DSL 랩 5인 팀 (나혜·현희·성하·수민·민찬)
뭘:     TIGER 기반 생성형 추천에서 비인기·신규 아이템 추천 성능 개선
민찬:   ① 구간별 평가 인프라 (팀 공통)  ② 층 수 × 구간 실험 (연구 주제)
베이스:  GRID (Snap Research의 TIGER 구현)

지금:   환경 구축 ✅  파이프라인 완주 ✅  구간별 평가 코드 ✅
        →  다음 관문은 랩 GPU 확보. 맥에서 TIGER 학습은 불가능함이 확인됨
```

---

## 1. 지금 상태

### ✅ 완료
```
· 논문 20편 이상 정독 → docs/02_논문_정리.md
· Ghost 논문 SKT 주장 100% 데이터 검증 (38,613개 전수)
· conda 환경 grid · MPS 작동 · TFRecord 읽기 성공
· 노션 8/11 페이지 업로드 완료 (토글 7개 + 표 10개)
· 🆕 파이프라인 Step 3~6 완주 (A안, 768차원 임베딩)
· 🆕 GRID 실행 함정 10개 발견·해결 → docs/08_파이프라인_완주기록.md
· 🆕 GRID 코드 버그 2건 수정 (비분산 환경이면 누구나 밟음)
· 🆕 구간별 평가 코드 작성 + 지표 검증 14개 통과
· 🆕 Beauty 인기도 분포 실측 (HEAD 20%가 상호작용 60.3% 독식)
```

### ⬜ 안 된 것
```
· GRID 논문 수치 재현 (B안) — 랩 GPU 필요
· 랩 GPU 사양 확인 (민찬님이 확인 중)  ← 🔴 이제 가장 급함
· 시간 기준 분할 (진짜 COLD 구간 생성)
· 팀원들에게 공유 전달 (아래 5절)
```

---

## 2. 🚨 이번에 확인된 가장 중요한 사실

### 맥에서 TIGER 학습은 불가능합니다

```
Step 3 코드북 학습    3분      ✅ 맥에서 가능
Step 4 SID 생성      15초      ✅ 가능
Step 6 추론          1분 32초  ✅ 가능
Step 5 TIGER 학습    스텝당 27초  ❌ 320,000스텝 = 약 100일
```

⚠️ dataloader 워커를 올리면 **오히려 느려집니다** (0개 27초 / 6개 34초 / 12개 36초).
맥에서는 래퍼 기본값 0을 그대로 두세요.

⚠️ 속도를 잴 때는 `trainer.val_check_interval` 을 크게 잡으세요.
검증 시간이 섞이면 스텝당 2.3분처럼 5배 부풀려진 값이 나옵니다.

**→ Step 3·4·6 은 맥에서 개발·디버깅하고, Step 5만 랩 GPU에서 돌리면 됩니다.**
그래서 랩 GPU 사양 확인이 다른 무엇보다 급해졌습니다.

### GRID 기본 설정은 "증강 꺼진" 쪽입니다

```yaml
# configs/experiment/tiger_train_flat.yaml (배포된 기본값)
transform:
  _target_: src.data.loading.components.label_function.NextKTokenMasking
```

선택지는 `Identity`(증강 효과 있음) / `NextKTokenMasking`(없음) 둘인데
**기본값이 증강 없는 쪽**입니다. GRID 논문은 이 차이를 42%로 보고합니다.

⚠️ 단, "기본값으로는 논문 수치가 안 나온다"고 아직 단정하면 안 됩니다.
양쪽을 실제로 돌려 봐야 합니다. **재현 검증의 1번 확인 항목입니다.**

### Beauty 인기도 분포 (실측)

| 구간 | 아이템 | 비율 | 상호작용 | 점유율 |
|---|---|---|---|---|
| HEAD | 2,414 | 19.9% | 92,759 | **60.3%** |
| BODY | 7,240 | 59.8% | 53,471 | 34.8% |
| TAIL | 2,414 | 19.9% | 7,546 | **4.9%** |
| UNSEEN | 33 | 0.3% | 0 | 0.0% |

아이템 수는 같은데 상호작용이 **12배** 차이납니다.

### SID 충돌 (L=3, 코드북 256)

```
앞 3자리만으로 고유    92.5%
같은 SID 최대 공유     11개
+1 자리 붙이면        12,101 / 12,101 전부 고유
```

---

## 3. 다음에 할 일 (우선순위)

### 🔴 1순위 — 랩 GPU 사양 확인
```
① 어떤 GPU  ② 메모리  ③ 몇 장  ④ 한 번에 몇 시간  ⑤ 큐 대기

TIGER는 1,300만 파라미터라 메모리는 1~2GB면 충분합니다.
병목은 시간입니다. 맥 기준 스텝당 27초가 얼마나 줄어드는지가 전부입니다.
32만 스텝이 하루 안에 끝나려면 스텝당 0.27초, 즉 100배가 필요합니다.
```

### 🔴 2순위 — 재현 검증 (B안, GPU 확보 후)
```
Step 2부터 정석대로 (Flan-T5-XL, 2048차원)
목표:  Beauty  Recall@10 0.0639,  NDCG@10 0.0347   (±5% 이내)
       ※ GRID 논문 5시드 평균. TIGER 원논문(0.0374)과 비교하면 안 됨

🚨 가장 먼저 확인할 것: label_function 을 Identity 로 바꿔야 하는가
```

### 🟡 3순위 — 시간 기준 분할 (진짜 COLD)
```
현재 데이터는 leave-one-out이라 COLD가 사실상 없습니다 (33개, 0.3%).
전체 타임스탬프 90% 지점 기준으로 나눠 신규 아이템 구간을 만들어야 합니다.
평가 코드는 이미 COLD 구간을 받을 준비가 돼 있습니다.
```

### 🟡 4순위 — Phase 1 실험
```
Beauty × L ∈ {2,3,4,5} × 시드 3 = 12회
층 수만 바꾸면 됩니다 (코드 수정 불필요).
충돌률은 segment_eval.py 가 자동으로 같이 냅니다.
```

---

## 4. 환경 정보 (바로 쓸 수 있음)

```bash
conda activate grid       # 또는 /opt/anaconda3/envs/grid/bin/python
```

### 🚨 버전 고정이 필요합니다
```bash
pip install "transformers==4.47.0" "tokenizers==0.21.0"
# transformers 5.x 는 T5Stack(embed_tokens=...) 인자가 사라져서 동작하지 않습니다
```

### 최소 설치 목록에 빠져 있던 4개 (requirements.txt 에는 있음)
```bash
pip install rootutils==1.0.7 psutil==6.1.1 \
            hydra-colorlog==1.2.0 google-cloud-bigquery==3.29.0
# google-cloud-bigquery 는 BigQuery를 안 써도 필요합니다.
# inference_utils.py 가 최상단에서 무조건 import 합니다.
```

### 실행은 래퍼를 쓰세요
```bash
scripts/grid_mac.sh train experiment=rkmeans_train_flat "data_dir='<절대경로>'" ...
```
MPS 설정, dataloader 워커/타임아웃, experiment마다 다른 설정 경로를 자동 처리합니다.
전체 명령 예시는 `docs/08_파이프라인_완주기록.md` 6절에 있습니다.

### 하드웨어
```
Apple M5 Pro (arm64) · 24GB 통합 메모리
NVIDIA GPU 없음 → MPS(Apple GPU) 사용
```

---

## 5. 팀에 전달해야 할 것 (아직 안 함)

```
🔴 팀 전체 — GRID 기본 설정이 증강 꺼진 쪽입니다
     label_function 기본값 = NextKTokenMasking
     논문 어블레이션은 이 차이를 42%로 보고합니다
     누가 재현하든 이걸 먼저 확인해야 합니다

🔴 팀 전체 — 구간별 평가 도구가 생겼습니다
     scripts/segment_eval.py
     GRID에 묶여 있지 않습니다. 예측 결과만 넣으면 누구 구현이든 같은 표가 나옵니다
     TAIL(학습에 있지만 적음)과 COLD(학습에 없음)를 반드시 구분해 주세요

🔴 나혜님께 Ghost 논문
     CRAB만 보고 계신데, Ghost가 같은 주제를 한 달 뒤 더 깊게 다뤘고
     코드·데이터가 다 공개돼 있습니다. CRAB은 코드가 없습니다
     github.com/Esperanto-mega/Ghost

🔴 수민님께
     SA²CRQ가 환각 −28% (온라인 A/B 완료)
     환각률은 segment_eval.py 가 이미 계산합니다

🟡 GRID 저장소에 버그 제보 (mju@snap.com)
     ① 쓰기 시 상위 폴더 미생성 → 첫 실행 무조건 실패
     ② 단일 프로세스에서도 torch.distributed.barrier() 호출
     둘 다 NVIDIA 다중 GPU가 아니면 누구나 밟습니다
```

---

## 6. 파일 지도

```
모델링 프로젝트/
├── HANDOFF.md              ← 지금 이 파일
├── README.md
├── 노션_업로드용.md          노션 8/11 페이지에 업로드 완료
├── scripts/                 🆕
│   ├── grid_mac.sh              맥(MPS) 실행 래퍼
│   ├── build_step2_embeddings.py Step 2 건너뛰기
│   ├── item_segments.py         HEAD/BODY/TAIL 분류
│   └── segment_eval.py          ⭐ 구간별 평가 (팀 공통 자)
├── tests/
│   └── test_segment_eval.py     지표 검증 14개
├── work/                    🆕 (git 제외 대상)
│   ├── step2_768/beauty/        768차원 임베딩 .pt
│   ├── segments/beauty.json     구간 정의
│   └── eval/                    평가 결과
├── docs/
│   ├── 00~07 ...
│   └── 08_파이프라인_완주기록.md  🆕 ⭐ 함정 10개 + 실측치
├── code/
│   ├── grid/               베이스 (버그 2건 수정, 호환 계층 1개 추가)
│   └── ghost/              참고용
└── data/amazon_data/       beauty / sports / toys
```

---

## 7. 🚨 절대 놓치면 안 되는 것

### 실행 시 함정 (기존)
```
① TIGER 학습 단계에서 num_hierarchies 에 +1
     코드북 3층으로 학습했으면 TIGER는 4로
② 폴더명이 README와 다름 — 실제는 training / evaluation / testing
③ RQ-VAE/R-VQ 추론 설정 파일 없음 (GRID 이슈 #17) → RK-Means만 쓰므로 무관
```

### 실행 시 함정 (이번에 추가로 발견 — 10개)
```
docs/08_파이프라인_완주기록.md 2절에 전부 정리했습니다.
scripts/grid_mac.sh 가 그중 4개를 자동 처리합니다.

가장 잘 걸리는 3개:
  · code/grid/.project-root 파일이 없습니다 (touch 하면 됩니다)
  · 경로에 공백·한글이 있으면 Hydra 오버라이드를 작은따옴표로 감싸야 합니다
  · transformers 는 반드시 4.47.0
```

### 수치 비교 기준
```
❌ TIGER 원논문 NDCG@10 0.0374 와 비교하면 안 됨
✅ GRID 기준선 NDCG@10 0.0347 로 비교

🚨 그리고 A안(768차원)으로 나온 수치는 어느 쪽과도 비교하면 안 됩니다.
   임베딩 출처가 달라서 파이프라인 동작 확인용일 뿐입니다.
```

---

## 8. 대화 방식 (민찬님 선호 — 중요)

```
✅ 지표·용어가 나오면 그 자리에서 설명할 것
     "Recall@10" → "= 추천 10개 안에 정답이 있는 유저 비율"
✅ 예시를 들어 설명할 것
✅ 간결하게. 핵심은 빼지 말고
✅ 한국어로

❌ "아무도 안 했다" / "빈 자리다" 같은 단정 금지
     ✅ 원문 확인  ⚠️ 검색 요약만 봄  ❓ 못 찾음  으로 구분 표기
❌ 확인 없이 추측을 사실처럼 말하지 말 것
```

### 이전 세션에서 5번 틀린 판단
```
ApeGNN "6년째 안 풀렸다" → WWW 2023에 있었음
Firzen "이미 했다"       → 절반만 했었음
디코딩 "아무도 안 했다"   → TIGER 원논문 4장에 있었음
KG 프루닝 "기준이 없다"   → 제목이 그대로인 논문 존재
negative선택 "통째로 비었다" → IJCAI 2019에 있었음

대응: github.com/HaFred/awesome-generative-recsys 에서 먼저 확인.
      그래도 안 나오면 "없다"가 아니라 "못 찾았다"라고 말할 것.
```

### 도구 팁
```
· arXiv PDF는 WebFetch로 파싱 실패 → Read 툴로 pages 지정, 또는 arxiv.org/html/<id>
· GitHub 파일은 raw.githubusercontent.com 사용
· 로컬 clone 후 직접 읽는 게 가장 정확
```

---

## 9. 핵심 사실 요약 (재조사 불필요)

```
[TIGER]
· NeurIPS 2023, Google DeepMind. 공식 코드 없음
· 1,300만 파라미터. 메모리 1~2GB면 충분 (병목은 시간)
· Beauty Recall@10 +0.15% / NDCG@10 +17.43%
  → "많이 맞히는" 게 아니라 "순위를 잘 매기는" 모델
· 콜드스타트 강점은 ε로 자리를 강제 예약한 결과

[GRID = 우리 베이스]
· Snap Research. TIGER 재현도 1위 (Recall@10 −0.5%)
· 어블레이션 8종: 슬라이딩윈도우 42% / 유저토큰은 빼는 게 나음 /
  RK-Means > RQ-VAE / 제약 빔서치가 오히려 손해
· 라이선스: Snap 비상업 연구 전용 (랩 연구·논문은 허용)

[Ghost]
· LC-Rec 기반, Qwen2.5-3B. 인기편향 원인 규명 + SKT/AUO
· Tail HR@10 +63.91%, 노출 43:1 → 1.6:1, 전체 정확도 −7.46%
· 데이터로 검증: tail 38,613개 100% head 접두사 상속
· 맥에서는 실행 어려움 (Qwen 3B + DeepSpeed 2GPU)

[연구 주제 근거]
GRID:       "층 늘리면 하락" (전체 평균만)
VarLenRec:  "인기 짧게, 비인기 길게" (구간별)
SA²CRQ:     "인기 길게, 비인기 짧게" (정반대)
→ 겹치는 구간이 L=4 하나뿐. L=2,3을 구간별로 본 사람 없음
→ Ghost가 실제로 인기 5토큰/비인기 7토큰 쓰고 Tail +63.91% 냄 (VarLenRec 편)
```

---

## 새 세션 시작 시 이렇게 말씀하시면 됩니다

```
모델링 프로젝트/HANDOFF.md 읽고 이어서 하자.
```
