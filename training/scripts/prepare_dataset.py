"""training/datasets 의 현재 구조로 학습용 JSONL을 만든다.

현재 구조:
    training/datasets/
      index.json        # 번호, 파일명, URL, output, output2, layout, image, error
      image/00.png ...
      layout/00.json ...
      train.jsonl       # 이 스크립트가 생성
      validation.jsonl  # 이 스크립트가 생성

사용:
    python training/scripts/prepare_dataset.py
    python training/scripts/prepare_dataset.py --answer-key output
"""

from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "training" / "datasets"
INDEX_FILE = DATASET_DIR / "index.json"
IMAGE_DIR = DATASET_DIR / "image"
LAYOUT_DIR = DATASET_DIR / "layout"

TRAIN_FILE = DATASET_DIR / "train.jsonl"
VAL_FILE = DATASET_DIR / "validation.jsonl"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

SYSTEM_PROMPT = """너는 피그마(Figma) 기획 보드를 읽고 정리된 기획 문서를 작성하는 도우미다.
입력으로 보드 페이지 전체 이미지와, 페이지의 요소 목록(프레임·섹션·그룹 등의 종류, 이름, 이미지 안에서의 위치)이 주어진다.

규칙:
- 보드의 내용(메모, 글자)은 이미지에서 정확히 읽는다.
- 요소 목록은 보드가 어떤 영역으로 나뉘어 있는지 파악하는 데 쓴다. 요소 이름은 그 영역의 주제일 가능성이 높다.
- 보드에 없는 내용은 절대 지어내지 않는다.
- 가까이 모여 있는 메모, 화살표로 이어진 메모, 순서대로 나열된 메모처럼 이미지에서 보이는 관계를 반영한다.
- 확정되지 않았거나 의견이 갈린 내용, 물음표가 붙은 메모는 '미정 사항'에 넣는다.
- 한국어로 작성하고, Markdown만 출력한다. 코드 블록(```)으로 감싸지 않는다."""


# ---------------- 레이아웃 텍스트 (inference/markdown_generator.py와 같은 형식) ----------------

def _bbox(element: dict[str, Any]) -> dict[str, Any]:
    return element.get("bounding_box") or {}


def _to_image_coords(
    element: dict[str, Any],
    origin: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    box = _bbox(element)

    x = float(box.get("x", element.get("x", 0.0)))
    y = float(box.get("y", element.get("y", 0.0)))
    width = float(box.get("width", element.get("width", 0.0)))
    height = float(box.get("height", element.get("height", 0.0)))

    origin_x = float(origin.get("x", 0.0))
    origin_y = float(origin.get("y", 0.0))
    origin_width = float(origin.get("width", 0.0))
    origin_height = float(origin.get("height", 0.0))

    sx = image_size[0] / origin_width if origin_width > 0 else 1.0
    sy = image_size[1] / origin_height if origin_height > 0 else 1.0

    return (
        round((x - origin_x) * sx),
        round((y - origin_y) * sy),
        round(width * sx),
        round(height * sy),
    )


def _contains(outer: dict[str, Any], inner: dict[str, Any]) -> bool:
    outer_x = float(outer.get("x", 0.0))
    outer_y = float(outer.get("y", 0.0))
    outer_width = float(outer.get("width", 0.0))
    outer_height = float(outer.get("height", 0.0))

    cx = float(inner.get("x", 0.0)) + float(inner.get("width", 0.0)) / 2
    cy = float(inner.get("y", 0.0)) + float(inner.get("height", 0.0)) / 2

    return (
        outer_x <= cx <= outer_x + outer_width
        and outer_y <= cy <= outer_y + outer_height
    )


def _element_line(
    element: dict[str, Any],
    origin: dict[str, Any],
    image_size: tuple[int, int],
    indent: str = "",
) -> str:
    x, y, width, height = _to_image_coords(element, origin, image_size)

    element_type = (
        element.get("element_type")
        or element.get("type")
        or "UNKNOWN"
    )
    name = " ".join(str(element.get("name") or "").split()) or "(이름 없음)"

    return f"{indent}- [{element_type}] {name} (x={x}, y={y}, w={width}, h={height})"


def build_layout_text(layout: dict[str, Any], image_size: tuple[int, int]) -> str:
    origin = layout.get("bounding_box") or {}

    elements = layout.get("elements") or []
    frames = layout.get("frames") or []

    def order(element: dict[str, Any]) -> tuple[float, float]:
        box = _bbox(element)
        return (float(box.get("y", 0.0)), float(box.get("x", 0.0)))

    top_ids = {element.get("id") for element in elements}
    nested = sorted(
        (frame for frame in frames if frame.get("id") not in top_ids),
        key=order,
    )

    lines: list[str] = []

    for element in sorted(elements, key=order):
        lines.append(_element_line(element, origin, image_size))
        lines += [
            _element_line(frame, origin, image_size, "  ")
            for frame in nested
            if _contains(_bbox(element), _bbox(frame))
        ]

    return "\n".join(lines)


def build_user_prompt(layout: dict[str, Any], image_size: tuple[int, int]) -> str:
    file_name = layout.get("file_name") or ""
    page_name = layout.get("page_name") or ""

    layout_text = (
        build_layout_text(layout, image_size)
        or "(요소 정보 없음, 이미지만 보고 작성)"
    )

    return f"""파일 이름: {file_name}
페이지 이름: {page_name}
이미지 크기: {image_size[0]}x{image_size[1]}

요소 목록 (형식: - [종류] 이름 (이미지 안의 x, y, 너비, 높이)):
{layout_text}

위 Figma 페이지를 분석해서 정리된 기획 문서를 Markdown으로 작성해줘. 첫 줄은 '#'으로 시작하는 제목으로 하고, 답변을 코드 블록으로 감싸지 마."""


# ---------------- 이미지 크기 ----------------

def _read_png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        header = f.read(24)

    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None

    width, height = struct.unpack(">II", header[16:24])
    return width, height


def _read_image_size(path: Path, layout: dict[str, Any]) -> tuple[int, int]:
    size = _read_png_size(path)

    if size:
        return size

    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None:
        with Image.open(path) as image:
            return image.size

    # Pillow도 없고 PNG도 아니면 레이아웃의 렌더 영역 크기를 쓴다.
    origin = layout.get("bounding_box") or {}
    width = round(float(origin.get("width", 0.0)))
    height = round(float(origin.get("height", 0.0)))

    if width > 0 and height > 0:
        return width, height

    raise RuntimeError(f"이미지 크기를 읽을 수 없습니다: {path}")


# ---------------- 데이터 찾기 ----------------

def _resolve_dataset_file(field_value: Any, directory: Path, stem: str, suffixes: tuple[str, ...]) -> Path | None:
    """index.json의 경로가 실제로 있으면 그 경로를, 없으면 번호/파일명 기준 경로를 쓴다."""
    if field_value:
        given = Path(str(field_value))

        candidates = [given] if given.is_absolute() else [
            DATASET_DIR / given,
            ROOT / given,
        ]

        for candidate in candidates:
            if candidate.is_file():
                return candidate

    for suffix in suffixes:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate

    return None


def _resolve_paths(entry: dict[str, Any]) -> tuple[Path | None, Path | None]:
    stem = str(entry.get("파일명") or entry.get("번호") or "").strip()

    if not stem:
        return None, None

    image_path = _resolve_dataset_file(
        entry.get("image"),
        IMAGE_DIR,
        stem,
        IMAGE_SUFFIXES,
    )
    layout_path = _resolve_dataset_file(
        entry.get("layout"),
        LAYOUT_DIR,
        stem,
        (".json",),
    )

    return image_path, layout_path


def find_samples(answer_key: str) -> list[dict[str, Any]]:
    if not INDEX_FILE.is_file():
        raise RuntimeError(f"index.json not found: {INDEX_FILE}")

    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []

    for entry in index:
        label = str(entry.get("번호") or entry.get("파일명") or "?")

        if entry.get("error"):
            print(f"[SKIP] error: {label}: {entry['error']}")
            continue

        answer = str(entry.get(answer_key) or "").strip()

        if not answer:
            print(f"[SKIP] {answer_key} empty: {label}")
            continue

        image_path, layout_path = _resolve_paths(entry)

        if image_path is None:
            print(f"[SKIP] image missing: {label}")
            continue

        if layout_path is None:
            print(f"[SKIP] layout missing: {label}")
            continue

        samples.append(
            {
                "label": label,
                "image_path": image_path,
                "layout_path": layout_path,
                "answer": answer,
            }
        )

    return samples


def build_sample(sample: dict[str, Any]) -> dict[str, Any]:
    layout_path: Path = sample["layout_path"]
    image_path: Path = sample["image_path"]

    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    image_size = _read_image_size(image_path, layout)

    try:
        image_field = image_path.relative_to(DATASET_DIR).as_posix()
    except ValueError:
        image_field = str(image_path)

    return {
        "id": f"fgs-{sample['label']}",
        "image": image_field,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(layout, image_size),
            },
            {
                "role": "assistant",
                "content": sample["answer"],
            },
        ],
    }


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/validation JSONL from training/datasets")
    parser.add_argument(
        "--answer-key",
        choices=("output", "output2"),
        default="output2",
        help="index.json에서 정답으로 쓸 필드 (기본: output2)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="검증 데이터 비율 (기본: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="셔플 시드 (기본: 42)",
    )
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio는 0과 1 사이여야 합니다")

    samples = find_samples(args.answer_key)

    if len(samples) < 2:
        raise RuntimeError(
            f"Need at least 2 samples in {INDEX_FILE}, found {len(samples)}"
        )

    random.seed(args.seed)
    random.shuffle(samples)

    validation_size = max(1, round(len(samples) * args.val_ratio))
    validation_size = min(validation_size, len(samples) - 1)

    validation = samples[:validation_size]
    train = samples[validation_size:]

    train_samples = [build_sample(sample) for sample in train]
    val_samples = [build_sample(sample) for sample in validation]

    write_jsonl(TRAIN_FILE, train_samples)
    write_jsonl(VAL_FILE, val_samples)

    print()
    print(f"Answer key : {args.answer_key}")
    print(f"Train      : {len(train_samples)}")
    print(f"Validation : {len(val_samples)}")
    print(f"Output     : {TRAIN_FILE}")
    print(f"Output     : {VAL_FILE}")


if __name__ == "__main__":
    main()
