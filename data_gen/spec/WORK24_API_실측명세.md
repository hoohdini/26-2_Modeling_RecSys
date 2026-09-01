# 고용24 OpenAPI — 실측으로 알아낸 호출 규약

> 개발명세서 없이 **실제 호출로 알아낸** 것입니다. 2026-08-30 확인.
> 인증키는 `.env` (커밋 차단됨). 서비스마다 키가 다릅니다.

## 공통

```
BASE = https://www.work24.go.kr/cm/openApi/call/wk/callOpenApiSvcInfo{코드}.do
공통 파라미터: authKey, returnType=XML|JSON
```

응답 오류 메시지로 상태를 구분할 수 있습니다.

| 메시지 | 뜻 |
|---|---|
| HTML 404 페이지 | 그 서비스코드 자체가 없음 |
| `신청하신 OpenApi 서비스가 존재하지 않습니다` | 코드는 있으나 **이 키에 안 물림** |
| `개인회원은 사용할 수 없는 OPEN-API입니다` | 코드·키 정상, **계정 등급 부족** |
| `target 정보가 바르지 않습니다` | 코드·키 정상, **파라미터 더 필요** |

실재하는 `/wk/` 서비스코드: **210, 211, 212, 213, 214, 215, 216, 217, 220**

---

## ✅ 212 — 직업정보 (`WORK24_KEY_JOB`)

### 212L01 · 직업 목록

```
?authKey=..&returnType=XML&target=JOBCD&pageNum=1&pageSize=500
```
→ `total=492`. 필드: `jobClcd`(중분류코드) `jobClcdNM` `jobCd`(예 K000000847) `jobNm`

### 212D01 · 직업 상세 — **이 프로젝트의 핵심 자원**

```
?authKey=..&returnType=XML&target=JOBDTL&jobGb=1&dtlGb={1..6}&jobCd=K000000847
```

| dtlGb | 주요 필드 | 우리 용도 |
|---|---|---|
| **1** | `jobSum`(개요) `jobAbil` `knowldg` `jobChr` `jobEnv` `jobProspect` `sal` **`relJobList`** `relCertList` | **관련직업 간선** + 프로필 텍스트 그라운딩 |
| **2** | `execJob`(수행직무 서술) `jobsDo` **`relJobList`** | 프로필 텍스트의 실제 문장 재료 |
| **3** | **`kecoCd`/`kecoNm`**(고용직업분류) `schDpt`·`engnrDpt`·`natrlDpt` 등 학과 `edubg`(학력) `relCertList` `relOrgList` | KECO 코드 · **학과↔직업 매핑(career_path)** |
| 4 | `jobProspect*` `salProspect` `jobSatis` | 인기도/전망 캘리브레이션 |
| **5** | `knwldgNm`×33 · `jobAblNm`×44 (+각 레벨값 `*Lvl`) | **스킬 벡터 — skill_overlap 간선의 실제 근거** |
| 6 | `intrstNm` `valsNm` `jobChrNm` | 프로필 다양성 부여 |

> `jobGb=1` 만 유효(2·5 = 잘못된 값, 3·4 = 사용중지). `dtlGb` 없으면 오류.

---

## ✅ 213 — 학과정보 (`WORK24_KEY_MAJOR`)

```
213L01?authKey=..&returnType=XML&target=MAJORCD&srchType=A&pageNum=1&pageSize=1000
```
→ `total=923`. 필드: `majorGb` `knowSchDptNm`(학과명) `knowDtlSchDptNm`(세부학과명) 등
`srchType` 은 **A 만** 유효(M/L/S/1/2/3 전부 거부).

---

## ❌ 210 — 채용정보 (`WORK24_KEY_EMP`) — 개인회원 차단

```
210L01?authKey=..&returnType=XML&outType=1&pageNum=1&pageSize=10
→ <error>개인회원은 사용할 수 없는 OPEN-API입니다.</error>
```

키는 승인·유효하지만 **계정 등급이 개인회원이라 호출이 막힙니다.**
구 워크넷 엔드포인트(`openapi.work.go.kr/opi/opi/opia/wantedApi.do`)는
이 키를 아예 인정하지 않습니다 — 고용24 신규 체계로 이관됐습니다.

---

## ❓ 직무정보 · 공통코드 · 취업역량 — 코드 미발견

세 키 모두 `/wk/`·`/hr/`·`/cm/`·`/jb/` × 210~220 조합에서 잡히지 않았습니다.
다른 대역이나 경로에 있을 것으로 보입니다. **다만 급하지 않습니다** —
필요한 것(KECO 코드·지식/능력 어휘)이 212D01 안에 전부 들어 있습니다.

---

## 수집 계획

| 대상 | 호출 수 | 비고 |
|---|---|---|
| 직업 목록 | 1 | 492건 |
| 직업 상세 dtlGb 1·2·3·5 | 492 × 4 = **1,968** | 핵심 |
| 학과 목록 | 1 | 923건 |

총 **약 2,000회**. 한도를 모르므로 호출 간 간격을 두고 이어받기(resume) 가능하게 짭니다.

---

## 수집 완료 기록 (2026-08-30)

호출 1,970회 · **오류 0** · 17분 30초 · 한도 미도달.
`data_gen/raw/` 에 저장 (`jobcd.xml`, `majorcd.xml`, `jobdtl/{jobCd}_d{1,2,3,5}.xml` 1,968개).

| 자산 | 규모 | 그래프 용도 |
|---|---|---|
| 직업 | 492 | 노드의 원형 |
| **관련직업 간선** | **2,316** (방향) · 평균차수 **5.02** · 최대 16 · 간선 보유 **461/492** | `related_occupation` (1순위) |
| **지식 33 + 능력 44 = 77차원 벡터** | **492/492 완비** · 값 0~100 | `skill_vector` (2순위) |
| KECO 코드 | 고유 405개 | `same_occupation` (4순위, 강등 후보) |
| 학과 | 고유 96개 × 7계열 필드(각 492) | `career_path` (3순위) |
| 학과 목록 | 923 | 프로필 전공 속성 |

### skill_vector 임계값 곡선 (직업 수준, 중심화 후 코사인)

| θ | 평균 차수 | 중앙값 | 고립 |
|---|---|---|---|
| 0.60 | 45.1 | 41 | 7 |
| 0.65 | 33.6 | 26 | 15 |
| **0.70** | **23.7** | 15 | 31 |
| 0.75 | 15.1 | 8 | 77 |

**θ≈0.70 이 Amazon Beauty 의 평균 차수 25 와 맞습니다.**
연속값이라 밀도를 부드럽게 맞출 수 있다는 것이 확인됐습니다
(사전등록 §3-1 "밀도로 맞추는 것이지 결과로 맞추는 것이 아니다").

> ⚠️ 위는 **직업 492개 수준**의 그래프입니다. 실제 아이템은 후보자 프로필 약 12,000개라,
> 직업당 약 24개 프로필이 붙습니다. 직업 수준 간선을 그대로 프로필로 확장하면
> 같은 직업끼리 24-클리크가 되어 텍스트와 겹칠 위험이 큽니다.
> 프로필마다 지식·능력 벡터에 개인 편차를 주어 **프로필 수준에서 연속 유사도**로
> 간선을 만드는 방식으로 갑니다. 판정은 G0 이 합니다.
