# 생성형 추천 모델링 프로젝트

> **팀 목표:** TIGER 기반 생성형 추천에서 **비인기·신규 아이템 추천(exploration) 성능 개선**
> **민찬 담당:** 구간별 평가 인프라 + 층 수 × 구간 실험

### 저장소에 올라가는 것 / 빠지는 것

```
올라감:  README · HANDOFF · docs/ · scripts/ · tests/ · patches/
빠짐:    data/   4.4GB — 아래 구글 드라이브에서 각자 내려받으세요
         code/   GRID·Ghost 저장소. 각자 clone 하세요 (수정분은 patches/ 에 있음)
         work/   중간 산출물. scripts/ 로 언제든 다시 만듭니다
```

**데이터 내려받기** — [P5 전처리 Amazon 데이터 (구글 드라이브)](https://drive.google.com/file/d/1B5_q_MT3GYxmHLrMK0-lAqgpbAuikKEz/view)
압축을 풀어 `data/amazon_data/` 에 두면 됩니다 (beauty / sports / toys).

**GRID 준비** — clone 후 수정분을 적용해야 돌아갑니다:
```bash
git clone https://github.com/snap-research/GRID.git code/grid
cp patches/torch_compat.py code/grid/src/utils/
(cd code/grid && git apply ../../patches/grid_local_fixes.patch)
```

---

## 📂 폴더 구조

```
모델링 프로젝트/
├── README.md                     ← 지금 이 파일
├── 요약.md                        ⭐ 한 장으로 보는 "무엇을 했나"
├── HANDOFF.md                    ⭐ 이어서 작업할 때 첫 번째로 읽을 것
├── docs/
│   ├── 00_야간작업보고서.md         8/8 새벽 작업 기록
│   ├── 01_배경과_목표.md            왜 이 주제인가
│   ├── 02_논문_정리.md              ⭐ 읽은 논문 전부 (가장 큰 자산)
│   ├── 03_실험_계획.md              층 수 × 구간 실험 설계
│   ├── 04_용어집.md                 지표·용어 설명
│   ├── 05_환경구축.md               설치 절차
│   ├── 06_코드데이터_분석.md         GRID/Ghost 코드·데이터 실측
│   ├── 07_진행상황.md               체크리스트
│   ├── 08_파이프라인_완주기록.md      ⭐ 실행 함정 10개 + 실측치
│   ├── 09_층수_충돌률_결과.md         ⭐ 첫 실험 결과 (16회 실행)
│   └── 10_COLD_구간_데이터셋.md       ⭐ 신규 아이템 구간을 만든 방법과 한계
├── scripts/
│   ├── grid_mac.sh                  맥(MPS)에서 GRID 실행하는 래퍼
│   ├── build_step2_embeddings.py    Step 2(3GB 모델) 건너뛰기
│   ├── item_segments.py             HEAD/BODY/TAIL 분류
│   ├── segment_eval.py              ⭐ 구간별 평가 — 팀 공통 자
│   ├── sweep_layers.py              층 수 × 충돌률 스윕
│   ├── plot_layer_sweep.py          스윕 결과 그림
│   ├── cold_split.py                신규 아이템(COLD) 데이터셋 생성
├── tests/
│   ├── test_segment_eval.py         지표 계산 검증 (신뢰구간·짝지은 검정 포함)
│   ├── test_cold_split.py           COLD 자르기 로직 검증 (임베딩 정렬 포함)
│   └── test_cold_dataset_loads.py   만든 데이터셋이 GRID 파서를 통과하는지
├── patches/                      code/grid 는 git 제외라 수정분을 여기 보존
├── code/
│   ├── grid/                     ⭐ Snap Research GRID — 우리 베이스
│   └── ghost/                    Ghost — 참고용 (데이터 268MB 포함)
├── work/                         중간 산출물 (git 제외, 언제든 재생성)
└── data/
    └── amazon_data/              beauty / sports / toys (4.4GB)
```

---

## 🚀 처음 오셨다면 이 순서로

| 순서 | 문서 | 무엇을 알 수 있나 |
|---|---|---|
| 0 | [요약](요약.md) | ⭐ **한 장으로 보는 전체** — 여기부터 보세요 |
| 1 | [01_배경과_목표](docs/01_배경과_목표.md) | 팀이 뭘 하려는지, 내 역할이 뭔지 |
| 2 | [04_용어집](docs/04_용어집.md) | Recall, NDCG 등 지표 읽는 법 |
| 3 | [02_논문_정리](docs/02_논문_정리.md) | 이 분야가 지금 어디까지 왔는지 |
| 4 | [03_실험_계획](docs/03_실험_계획.md) | 우리가 뭘 실험할 건지 |
| 5 | [08_파이프라인_완주기록](docs/08_파이프라인_완주기록.md) | 실제로 돌리는 법, 밟게 될 함정 |
| 6 | [HANDOFF](HANDOFF.md) | 지금 뭘 해야 하는지 |

---

## 📌 지금 상태 (2026-08-09)

```
✅ GRID / Ghost 저장소 확보
✅ Amazon 데이터 4.4GB 다운로드 + 압축 해제
✅ 논문 20편 이상 정독 · Ghost 핵심 주장 데이터로 검증
✅ 환경 구축 (conda grid · MPS 동작)
✅ GRID 파이프라인 Step 3~6 완주  ← 실행 함정 10개 해결
✅ 구간별 평가 코드 작성 + 지표 검증 통과

⬜ 논문 수치 재현 확인 (B안)   ← 랩 GPU 필요
⬜ 시간 기준 분할 (진짜 COLD 구간)
```

🚨 **맥에서 TIGER 학습(Step 5)은 불가능합니다** — 스텝당 27초, 기본 32만 스텝(≈100일).
Step 3·4·6 은 맥에서 되므로, Step 5만 랩 GPU에서 돌리면 됩니다.

**다음 할 일:** [HANDOFF.md](HANDOFF.md) 3절 참고

---

## 🎯 한 줄 요약

> TIGER는 **의미 번호(Semantic ID)** 를 한 칸씩 생성해서 추천하는 모델입니다.
> 그런데 이 방식이 **인기 아이템에 심하게 쏠립니다.**
> 우리는 **"번호를 몇 칸으로 만들 것인가"가 인기/비인기 구간마다 다른 답을 갖는지** 확인합니다.

