# 데이터 출처와 이용허락 체크리스트

작성 2026-09-15. 노션 데이터셋 계획 3절의 표를 저장소에 옮기고, 오늘 확인한 것을 채웠다. 법률 자문이 아니라 공개 페이지에 적힌 조건을 옮긴 것이다.

## 1. 출처별 조건

| 출처 | 확인한 조건 | 확인 방법·날짜 | 우리 사용 방식 | 공개 배포 가능? |
|---|---|---|---|---|
| 고용24(워크넷) 직업정보 OpenAPI — 이미 수집한 492 직업, 지식·능력 벡터, 관련직업, KECO, 학력·전공 분포, 자격증 | **공공누리 제4유형: 출처표시, 상업적 이용금지, 변경금지** | 공공데이터포털 3071087 페이지, 2026-09-15 (수정일 2025-11-03) | 팀 내부 비상업 연구. 직업명·코드·분류는 그대로 인용. 지식·능력 점수는 프로필 생성의 씨앗으로 쓰고 개인 편차를 더한다 | **원본과 파생 파일(`data_gen/raw/`, `profile_skill.npy`, `profiles.csv` 의 실데이터 유래 열)은 공개 배포하지 않는다.** 변경금지 조항 때문이다. 공개 시에는 수집·생성 코드와 레시피만 공개하고 출처를 표시한다 |
| 고용24 학과정보 OpenAPI (학과 923) | 직업정보와 같은 계열로 본다. 개별 페이지 미확인 | 미확인 | 전공 속성 사전 | 위와 같이 취급 |
| 고용24 채용정보 OpenAPI | 공공누리 4유형. 고용24 직접 API 는 기업회원 전용 | 공공데이터포털 3038225, 고용24 안내 페이지 2026-09-15 | **쓰지 않는다** (개인회원 차단) | 해당 없음 |
| 워크넷 직무데이터사전 OpenAPI (NCS 능력단위) | **이용허락범위 제한 없음** | 공공데이터포털 15088880, 2026-09-15 (수정일 2025-07-21) | 아직 안 씀. 직무 어휘 보강이 필요하면 이 출처를 우선한다 | 가능 |
| 워크넷 직업사전 목록·상세 OpenAPI | 공공누리 4유형 | 공공데이터포털 15037284, 2026-09-15 | 안 씀 | 해당 없음 |
| 캐글 Job Recommendation Challenge (CareerBuilder 2012) | 경진대회 규칙 동의 후 학술·비상업 사용 가능, 재배포 금지 (노션 계획 3절 출처 1·2). **규칙 페이지는 로그인이 필요해 이번에 본문 재확인 못함** | 팀원이 규칙 동의 시 본문을 캡처해 이 표에 붙인다 | 저장소 밖 `D:/DSL/_external/kaggle_job_recommendation/` 에만 둔다. 저장소에는 집계 통계(`data_gen/spec/kaggle_target_dist.json`)와 매핑 요약(`title_map_summary.json`)만 넣는다. 직무명 문자열이 든 `title_map.csv`, `title_review_200.csv`, `g4_key.csv` 도 저장소 밖 | 원본·파생 파일 불가. 집계값은 가능 |
| KECO 2018, NCS-KECO 연계표 | 공개 분류 체계 | 노션 계획 3절 출처 7·8 | 매핑 기준 코드 | 가능 |

## 2. 지금 저장소가 지키고 있는 것

- `.env` (인증키 6개)는 `.gitignore` 로 차단돼 있고 커밋 이력에 없다 (`git log --all -- .env` 결과 없음, 2026-09-15 확인).
- `data_gen/raw/` (고용24 원본 XML 2.0GB) 는 커밋되지 않는다.
- 저장소는 private 이다. **public 전환 전에** 1절의 공개 배포 불가 항목을 히스토리에서 제거해야 한다. `profiles.csv`, `profile_skill.npy`, `jobs_edges_*.csv` 가 이미 커밋돼 있으므로 히스토리 정리가 필요하다.
- 발표 자료에는 출처 문구를 넣는다: 직업정보는 한국고용정보원 고용24 OpenAPI(공공누리 4유형), 분포 기준은 Kaggle Job Recommendation Challenge (CareerBuilder, 2012).

## 3. 남은 확인

| 항목 | 담당 | 기한 |
|---|---|---|
| 캐글 규칙 본문 캡처 (다운로드하는 사람) | 데이터셋 담당 | 9/16 |
| 학과정보 API 의 공공데이터포털 이용허락 유형 | 데이터셋 담당 | 9/23 |
| public 전환 시 히스토리 정리 범위 | 팀장 | 행사 이후 |

## 출처

1. 공공데이터포털, 한국고용정보원_워크넷_직업정보. https://www.data.go.kr/data/3071087/openapi.do
2. 공공데이터포털, 한국고용정보원_워크넷 채용정보 채용목록 및 상세정보. https://www.data.go.kr/data/3038225/openapi.do
3. 공공데이터포털, 한국고용정보원_워크넷_직무데이터사전. https://www.data.go.kr/data/15088880/openapi.do
4. 공공데이터포털, 한국고용정보원_워크넷 직업사전 목록 및 상세정보. https://www.data.go.kr/data/15037284/openapi.do
5. 고용24 Open-API 소개 (기업회원 전용 안내). https://m.work24.go.kr/cm/e/a/0110/selectOpenApiIntro.do
6. 공공누리 제4유형 설명. https://www.kogl.or.kr/info/licenseType4.do
7. Kaggle, Job Recommendation Challenge 규칙. https://www.kaggle.com/competitions/job-recommendation/rules
8. Kaggle, Job Recommendation Challenge 데이터. https://www.kaggle.com/c/job-recommendation/data
