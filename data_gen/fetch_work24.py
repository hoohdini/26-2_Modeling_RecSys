"""고용24 OpenAPI 수집기 — 직업정보 492 × 상세 4종 + 학과정보 923.

호출 규약은 `data_gen/spec/WORK24_API_실측명세.md` 참고 (실측으로 알아낸 것).
인증키는 `.env` 에서 읽는다. 절대 인자로 받지 않는다(쉘 히스토리에 남으므로).

이어받기(resume)
    응답을 파일 하나씩 저장하고, 이미 있으면 건너뛴다.
    일일 호출 한도를 모르므로 중간에 끊겨도 다시 실행하면 이어진다.

사용
    python data_gen/fetch_work24.py            # 전부
    python data_gen/fetch_work24.py --only jobs
    python data_gen/fetch_work24.py --sleep 0.5
"""
import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_gen", "raw")
BASE = "https://www.work24.go.kr/cm/openApi/call/wk/callOpenApiSvcInfo{code}.do"

# 직업 상세에서 받을 종류. 4·6 은 이번 설계에 안 쓰지만 4는 캘리브레이션용으로 받아둔다.
DTL_GB = (1, 2, 3, 5)


def env(name):
    path = os.path.join(ROOT, ".env")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == name:
                    return v.strip()
    raise SystemExit(f"{name} 를 {path} 에서 찾지 못했습니다")


def get(code, key, params, retries=3, timeout=30):
    q = urllib.parse.urlencode({"authKey": key, "returnType": "XML", **params})
    url = BASE.format(code=code) + "?" + q
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{code} {params} 실패: {last}")


def err_of(xml):
    """오류 응답이면 메시지를, 정상이면 None 을 돌려준다."""
    m = re.search(r"<(?:message|error)>([^<]*)</(?:message|error)>", xml)
    return m.group(1).strip() if m else None


def save(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def fetch_jobs(key, sleep):
    """직업 목록 492 → 각 직업의 상세 dtlGb 1·2·3·5."""
    list_path = os.path.join(RAW, "jobcd.xml")
    if not os.path.exists(list_path):
        xml = get("212L01", key, {"target": "JOBCD", "pageNum": 1, "pageSize": 500})
        e = err_of(xml)
        if e:
            raise SystemExit(f"직업 목록 실패: {e}")
        save(list_path, xml)
        print(f"직업 목록 저장 → {list_path}")
    xml = open(list_path, encoding="utf-8").read()

    codes = re.findall(r"<jobCd>([^<]+)</jobCd>", xml)
    total = re.search(r"<total>(\d+)</total>", xml)
    print(f"직업 {len(codes)}개 (total={total.group(1) if total else '?'})")
    if len(codes) != len(set(codes)):
        print(f"  ⚠️ 중복 jobCd {len(codes) - len(set(codes))}건 — 고유 코드로 진행")
    codes = list(dict.fromkeys(codes))

    todo = [(c, d) for c in codes for d in DTL_GB
            if not os.path.exists(os.path.join(RAW, "jobdtl", f"{c}_d{d}.xml"))]
    print(f"상세 {len(codes) * len(DTL_GB)}건 중 {len(todo)}건 남음")

    done = fail = 0
    for i, (code, d) in enumerate(todo, 1):
        out = os.path.join(RAW, "jobdtl", f"{code}_d{d}.xml")
        try:
            xml = get("212D01", key, {"target": "JOBDTL", "jobGb": 1, "dtlGb": d, "jobCd": code})
        except RuntimeError as ex:
            print(f"  [{i}/{len(todo)}] {code} d{d} 네트워크 실패 — 중단: {ex}")
            break
        e = err_of(xml)
        if e:
            fail += 1
            # 한도 초과나 권한 오류면 더 진행할 이유가 없다
            if any(w in e for w in ("한도", "초과", "사용할 수 없", "유효하지 않")):
                print(f"  [{i}/{len(todo)}] 중단 — {e}")
                break
            if fail <= 5:
                print(f"  [{i}/{len(todo)}] {code} d{d} → {e}")
        else:
            save(out, xml)
            done += 1
        if i % 100 == 0:
            print(f"  [{i}/{len(todo)}] 저장 {done} · 오류 {fail}", flush=True)
        time.sleep(sleep)
    print(f"직업 상세: 새로 저장 {done} · 오류 {fail}")


def fetch_majors(key, sleep):
    path = os.path.join(RAW, "majorcd.xml")
    if os.path.exists(path):
        print(f"학과 목록 이미 있음 → {path}")
        return
    xml = get("213L01", key, {"target": "MAJORCD", "srchType": "A", "pageNum": 1, "pageSize": 1000})
    e = err_of(xml)
    if e:
        raise SystemExit(f"학과 목록 실패: {e}")
    save(path, xml)
    total = re.search(r"<total>(\d+)</total>", xml)
    print(f"학과 목록 저장 ({total.group(1) if total else '?'}건) → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("jobs", "majors"), default=None)
    ap.add_argument("--sleep", type=float, default=0.35, help="호출 간격(초)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    t0 = time.time()
    if args.only in (None, "majors"):
        fetch_majors(env("WORK24_KEY_MAJOR"), args.sleep)
    if args.only in (None, "jobs"):
        fetch_jobs(env("WORK24_KEY_JOB"), args.sleep)
    print(f"\n총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
