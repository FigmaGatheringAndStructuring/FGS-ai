#!/usr/bin/env python3
"""figma_links.csv 의 Figma 링크를 FGS-backend 배포 서버로 보내
페이지 layout(JSON)과 render 이미지(PNG)를 각각 저장한다.

index.json 에는 figma_links_2.csv 의 output / output2 컬럼도 함께 기록된다.

사용 예:
    python fetch_figma.py
    python fetch_figma.py --scale 2 --workers 4 --force
    python fetch_figma.py --only 00 05 10 --outdir result
    python fetch_figma.py --csv figma_links_2.csv --extra-csv ""
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

BASE_URL = "https://fgs-backend.juyoungk09.dev"
LAYOUT_PATH = "/api/figma/page"
IMAGE_PATH = "/api/figma/page/render"

USER_AGENT = "fgs-figma-fetch/1.0"

# index.json 에 함께 기록할 추가 컬럼 (figma_links_2.csv 의 output, output2 등).
KNOWN_COLUMNS = {"번호", "파일명", "URL", "공유 권한", "파일 검증"}


@dataclass(frozen=True)
class Row:
    index: str
    name: str
    url: str
    extra: dict[str, str]


def read_rows(csv_path: Path) -> list[Row]:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                rows: list[Row] = []
                for raw in reader:
                    url = (raw.get("URL") or "").strip()
                    if not url.startswith("http"):
                        continue
                    index = (raw.get("번호") or "").strip()
                    name = (raw.get("파일명") or index).strip()
                    extra = {
                        key: (raw.get(key) or "").strip()
                        for key in (reader.fieldnames or [])
                        if key and key not in KNOWN_COLUMNS
                    }
                    rows.append(
                        Row(
                            index=index,
                            name=safe_name(name),
                            url=url,
                            extra=extra,
                        )
                    )
                return rows
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"CSV 인코딩을 읽을 수 없습니다: {csv_path}")


def merge_extra_rows(rows: list[Row], extra_csv: Path) -> list[Row]:
    """다른 CSV(예: figma_links_2.csv)의 추가 컬럼을 번호 기준으로 합친다."""
    if not extra_csv.is_file():
        print(f"[WARN] 추가 컬럼 CSV 가 없습니다: {extra_csv.resolve()}", flush=True)
        return rows

    extra_by_index = {row.index: row.extra for row in read_rows(extra_csv)}
    merged: list[Row] = []
    for row in rows:
        extra = dict(row.extra)
        extra.update(
            {k: v for k, v in extra_by_index.get(row.index, {}).items() if v != ""}
        )
        merged.append(replace(row, extra=extra))
    return merged


def safe_name(value: str) -> str:
    cleaned = "".join("_" if ch in '\\/:*?"<>|' else ch for ch in value).strip()
    return cleaned or "unnamed"


def build_url(path: str, url: str, scale: float | None = None) -> str:
    query = {"url": url}
    if scale is not None:
        query["scale"] = f"{scale:g}"
    return f"{BASE_URL}{path}?{urllib.parse.urlencode(query)}"


def request_bytes(target: str, timeout: float) -> tuple[bytes, str]:
    req = urllib.request.Request(target, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.read(), res.headers.get("Content-Type", "")


def fetch_with_retry(
    target: str, timeout: float, retries: int, backoff: float
) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return request_bytes(target, timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            last_error = RuntimeError(f"HTTP {exc.code}: {body or exc.reason}")
            # 4xx 는 재시도해도 결과가 같을 가능성이 높다 (429 는 예외).
            if 400 <= exc.code < 500 and exc.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(backoff * (2 ** (attempt - 1)))
    raise RuntimeError(f"{target} 요청 실패: {last_error}")


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def process(
    row: Row,
    outdir: Path,
    scale: float,
    timeout: float,
    retries: int,
    backoff: float,
    force: bool,
) -> dict:
    layout_path = outdir / "layout" / f"{row.name}.json"
    image_path = outdir / "image" / f"{row.name}.png"
    result = {
        "번호": row.index,
        "파일명": row.name,
        "URL": row.url,
        **row.extra,
        "layout": None,
        "image": None,
        "error": None,
    }

    try:
        if force or not layout_path.exists():
            body, content_type = fetch_with_retry(
                build_url(LAYOUT_PATH, row.url), timeout, retries, backoff
            )
            try:
                parsed = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"layout 응답이 JSON 이 아님 ({content_type}): {exc}")
            if not isinstance(parsed, dict) or "error" in parsed:
                raise RuntimeError(f"layout 응답 오류: {str(parsed)[:300]}")
            # 보기 좋게 들여쓰기해서 다시 저장.
            write_atomic(
                layout_path,
                json.dumps(parsed, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            result["layout"] = str(layout_path)
        else:
            result["layout"] = str(layout_path)

        if force or not image_path.exists():
            body, content_type = fetch_with_retry(
                build_url(IMAGE_PATH, row.url, scale), timeout, retries, backoff
            )
            if not body:
                raise RuntimeError("image 응답이 비어 있음")
            if "json" in content_type.lower() or body[:1] == b"{":
                raise RuntimeError(f"image 응답 오류: {body.decode('utf-8', 'replace')[:300]}")
            write_atomic(image_path, body)
            result["image"] = str(image_path)
        else:
            result["image"] = str(image_path)

        print(f"[OK]   {row.index:>3} {row.name}", flush=True)
    except Exception as exc:  # noqa: BLE001 - 항목별 실패를 계속 진행
        result["error"] = str(exc)
        print(f"[FAIL] {row.index:>3} {row.name}: {exc}", flush=True)

    return result


def write_index(index_path: Path, results: list[dict]) -> list[dict]:
    """기존 index.json 과 병합해서 번호 기준으로 정렬해 저장한다."""
    merged: dict[str, dict] = {}
    if index_path.is_file():
        try:
            for entry in json.loads(index_path.read_text(encoding="utf-8")):
                if isinstance(entry, dict) and entry.get("번호"):
                    merged[entry["번호"]] = entry
        except (json.JSONDecodeError, OSError):
            pass
    for entry in results:
        merged[entry["번호"]] = entry

    ordered = sorted(merged.values(), key=lambda r: r["번호"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="figma_links.csv 링크로 layout(JSON)과 image(PNG)를 저장합니다."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="입력 CSV 경로 (기본: figma_links.csv, 없으면 figma_links_2.csv)",
    )
    parser.add_argument(
        "--extra-csv",
        default="figma_links_2.csv",
        help="output/output2 등 추가 컬럼을 가져올 CSV (기본: figma_links_2.csv)",
    )
    parser.add_argument("--outdir", default="out", help="출력 디렉터리 (기본: out)")
    parser.add_argument(
        "--scale", type=float, default=1.0, help="render scale (0.01~4.0, 기본: 1.0)"
    )
    parser.add_argument("--workers", type=int, default=3, help="동시 요청 수 (기본: 3)")
    parser.add_argument("--timeout", type=float, default=300.0, help="요청 타임아웃 초")
    parser.add_argument("--retries", type=int, default=3, help="재시도 횟수 (기본: 3)")
    parser.add_argument("--backoff", type=float, default=2.0, help="재시도 대기 기본 초")
    parser.add_argument(
        "--force", action="store_true", help="이미 저장된 파일도 다시 받기"
    )
    parser.add_argument(
        "--only", nargs="*", metavar="번호", help="지정한 번호만 처리 (예: --only 00 01)"
    )
    args = parser.parse_args()

    if not 0.01 <= args.scale <= 4.0:
        raise SystemExit("--scale 은 0.01~4.0 사이여야 합니다.")
    if args.workers < 1:
        raise SystemExit("--workers 는 1 이상이어야 합니다.")

    csv_path = Path(args.csv) if args.csv else Path("figma_links.csv")
    if not csv_path.is_file() and not args.csv:
        csv_path = Path("figma_links_2.csv")
    if not csv_path.is_file():
        raise SystemExit(f"CSV 파일을 찾을 수 없습니다: {csv_path.resolve()}")

    rows = read_rows(csv_path)
    if args.extra_csv:
        rows = merge_extra_rows(rows, Path(args.extra_csv))
    if args.only:
        wanted = set(args.only)
        rows = [r for r in rows if r.index in wanted or r.name in wanted]
    if not rows:
        raise SystemExit("처리할 링크가 없습니다.")

    outdir = Path(args.outdir)
    (outdir / "layout").mkdir(parents=True, exist_ok=True)
    (outdir / "image").mkdir(parents=True, exist_ok=True)

    print(f"대상 {len(rows)}건 -> {outdir.resolve()} (scale={args.scale:g})", flush=True)

    results: list[dict] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process,
                row,
                outdir,
                args.scale,
                args.timeout,
                args.retries,
                args.backoff,
                args.force,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r["번호"])
    processed = results
    index_path = outdir / "index.json"
    write_index(index_path, processed)

    failed = [r for r in processed if r["error"]]
    elapsed = time.perf_counter() - started
    print(
        f"\n완료: 성공 {len(processed) - len(failed)}건, 실패 {len(failed)}건, "
        f"{elapsed:.1f}초",
        flush=True,
    )
    print(f"목록: {index_path.resolve()} ({len(processed)}건 반영)", flush=True)
    if failed:
        print("실패 항목:", flush=True)
        for item in failed:
            print(f"  - {item['번호']} {item['파일명']}: {item['error']}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
