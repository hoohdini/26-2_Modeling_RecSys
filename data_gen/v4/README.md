# data_gen/v4 — 캐글 분포 재보정 (2026-09-15)

전체 설명은 `../README.md` 6절과 `docs/DATASET_v4_파이프라인.md`.

| 스크립트 | 하는 일 | 캐글 데이터 필요 |
|---|---|---|
| `kaggle_stats.py` | 캐글 tsv → 목표 분포 JSON + 표 (집계만 저장소에) | 예 |
| `map_kaggle_titles.py` | 캐글 직무명 → 492 직업·KECO 매핑, 검수 시트 200건, `--score-review` | 예 |
| `gate_g3_stats.py` | 생성 데이터 vs 목표 분포 판정 (`--beauty` 로 참고선 생성) | 아니오 |
| `calibrate_simulator.py` | temp·gamma·k 격자 → G-L 통과 중 G3 최적 선택 | 목표 JSON 만 |
| `make_g4_sheet.py` | 블라인드 검수 시트 생성·채점 | 예 |
| `dataset_manifest.py` | 데이터 파일 해시·파라미터 기록 | 아니오 |

`out/` 은 커밋하지 않는다 (`.gitignore`). 캐글 원본과 직무명이 든 파일은 `D:/DSL/_external/kaggle_job_recommendation/` 에만 둔다.
