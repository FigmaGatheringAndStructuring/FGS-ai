"""training/datasets 의 현재 구조로 학습용 JSONL을 만든다.

`fetch_figma.py` 가 만든 index.json 의 `output`(=output1), `output2` 컬럼을
정답으로 사용한다. 같은 이미지에 대해 정답을 여러 개 쓰는 데이터셋도 만들 수 있다.

현재 구조:
    training/datasets/
      index.json        # 번호, 파일명, URL, output, output2, layout, image, error
      image/00.png ...
      layout/00.json ...
      split.json        # train/validation 을 고정하는 분할 정보 (자동 생성)
      output2/          # output2 만 정답으로 쓴 데이터셋
        train.jsonl
        validation.jsonl
        manifest.json
      output1_output2/  # output1(output) + output2 를 모두 정답으로 쓴 데이터셋
        train.jsonl
        validation.jsonl
        manifest.json

사용:
    # output2 만 사용
    python training/scripts/prepare_dataset.py --answer-keys output2

    # output1(output) + output2 를 모두 사용 (항목당 2개 샘플)
    python training/scripts/prepare_dataset.py --answer-keys output output2

    # 이름/출력 위치 직접 지정
    python training/scripts/prepare_dataset.py \
        --answer-keys output output2 \
        --name output1_output2 \
        --outdir training/datasets/output1_output2
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
SPLIT_FILE = DATASET_DIR / "split.json"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
ANSWER_KEYS = ("output", "output2")

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

def _resolve_dataset_file(
    field_value: Any,
    directory: Path,
    stem: str,
    suffixes: tuple[str, ...],
) -> Path | None:
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


def find_samples(answer_keys: list[str]) -> list[dict[str, Any]]:
    """index.json 에서 이미지/레이아웃/정답이 모두 있는 항목만 고른다."""
    if not INDEX_FILE.is_file():
        raise RuntimeError(f"index.json not found: {INDEX_FILE}")

    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []

    for entry in index:
        label = str(entry.get("번호") or entry.get("파일명") or "?")

        if entry.get("error"):
            print(f"[SKIP] error: {label}: {entry['error']}")
            continue

        answers: dict[str, str] = {}
        missing_answer = False

        for key in answer_keys:
            answer = str(entry.get(key) or "").strip()

            if not answer:
                print(f"[SKIP] {key} empty: {label}")
                missing_answer = True
                break

            answers[key] = answer

        if missing_answer:
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
                "answers": answers,
            }
        )

    return samples


# ---------------- train/validation 분할 ----------------

def load_or_create_split(
    labels: list[str],
    split_file: Path,
    val_ratio: float,
    seed: int,
    force: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """모든 데이터셋이 같은 검증 항목을 쓰도록 분할을 파일로 고정한다.

    반환값: (train_labels, validation_labels, validation_labels_in_split_file)
    """
    label_set = set(labels)

    if split_file.is_file() and not force:
        try:
            saved = json.loads(split_file.read_text(encoding="utf-8"))
            saved_validation = [str(x) for x in saved.get("validation", [])]
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] split 파일을 읽지 못해 새로 만듭니다: {exc}")
            saved_validation = []
    else:
        saved_validation = []

    if saved_validation:
        validation = [label for label in saved_validation if label in label_set]
        dropped = [label for label in saved_validation if label not in label_set]

        if dropped:
            print(
                f"[WARN] split 파일의 검증 항목 {len(dropped)}개가 이 데이터셋에 없습니다: "
                f"{', '.join(dropped[:5])}{' ...' if len(dropped) > 5 else ''}"
            )
    else:
        shuffled = list(labels)
        random.Random(seed).shuffle(shuffled)

        validation_size = max(1, round(len(shuffled) * val_ratio))
        validation_size = min(validation_size, len(shuffled) - 1)
        validation = sorted(shuffled[:validation_size])

        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "val_ratio": val_ratio,
                    "validation": validation,
                    "train": sorted(label for label in labels if label not in set(validation)),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
            newline="\n",
        )
        print(f"[SPLIT] 새 분할을 저장했습니다: {split_file}")

    validation_set = set(validation)
    train = [label for label in labels if label not in validation_set]
    val = [label for label in labels if label in validation_set]

    return train, val, validation


# ---------------- 샘플 만들기 ----------------

def build_sample(
    sample: dict[str, Any],
    answer_key: str,
    multi_answer: bool,
) -> dict[str, Any]:
    layout_path: Path = sample["layout_path"]
    image_path: Path = sample["image_path"]

    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    image_size = _read_image_size(image_path, layout)

    try:
        image_field = image_path.relative_to(DATASET_DIR).as_posix()
    except ValueError:
        image_field = str(image_path)

    sample_id = (
        f"fgs-{sample['label']}-{answer_key}"
        if multi_answer
        else f"fgs-{sample['label']}"
    )

    return {
        "id": sample_id,
        "label": sample["label"],
        "answer_key": answer_key,
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
                "content": sample["answers"][answer_key],
            },
        ],
    }


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for sample in samples:
            f.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )


def default_dataset_name(answer_keys: list[str]) -> str:
    # output 은 사용자 문서에서 output1 이라고 부르므로 이름만 output1 로 쓴다.
    return "_".join("output1" if key == "output" else key for key in answer_keys)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/validation JSONL from training/datasets"
    )
    parser.add_argument(
        "--answer-keys",
        nargs="+",
        choices=ANSWER_KEYS,
        default=None,
        help=(
            "index.json에서 정답으로 쓸 필드. 여러 개를 주면 항목당 정답 개수만큼 "
            "샘플을 만든다 (기본: output2)"
        ),
    )
    parser.add_argument(
        "--answer-key",
        dest="answer_keys",
        nargs="+",
        choices=ANSWER_KEYS,
        help=argparse.SUPPRESS,  # 이전 옵션 호환용
    )
    parser.add_argument(
        "--name",
        default=None,
        help="데이터셋 이름 (기본: answer key 조합, 예: output2 / output1_output2)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="출력 디렉터리 (기본: training/datasets/<name>)",
    )
    parser.add_argument(
        "--split-file",
        default=str(SPLIT_FILE),
        help="train/validation 분할을 고정하는 JSON 파일 (기본: training/datasets/split.json)",
    )
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="기존 split 파일을 무시하고 새로 분할",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="검증 데이터 비율 (기본: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="셔플 시드 (기본: 42)",
    )
    args = parser.parse_args()

    answer_keys = args.answer_keys or ["output2"]

    # 중복 제거 (순서 유지)
    answer_keys = list(dict.fromkeys(answer_keys))

    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio는 0과 1 사이여야 합니다")

    name = args.name or default_dataset_name(answer_keys)
    outdir = Path(args.outdir) if args.outdir else DATASET_DIR / name
    split_file = Path(args.split_file)

    samples = find_samples(answer_keys)

    if len(samples) < 2:
        raise RuntimeError(
            f"Need at least 2 samples in {INDEX_FILE}, found {len(samples)}"
        )

    labels = [sample["label"] for sample in samples]
    train_labels, validation_labels, split_validation = load_or_create_split(
        labels,
        split_file,
        args.val_ratio,
        args.seed,
        force=args.force_split,
    )

    by_label = {sample["label"]: sample for sample in samples}
    multi_answer = len(answer_keys) > 1

    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []

    for label in train_labels:
        for key in answer_keys:
            train_samples.append(build_sample(by_label[label], key, multi_answer))

    for label in validation_labels:
        for key in answer_keys:
            val_samples.append(build_sample(by_label[label], key, multi_answer))

    train_file = outdir / "train.jsonl"
    val_file = outdir / "validation.jsonl"

    write_jsonl(train_file, train_samples)
    write_jsonl(val_file, val_samples)

    manifest = {
        "name": name,
        "answer_keys": answer_keys,
        "source_index": str(INDEX_FILE),
        "split_file": str(split_file),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "train_items": len(train_labels),
        "validation_items": len(validation_labels),
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "validation_labels": validation_labels,
        "files": {
            "train": str(train_file),
            "validation": str(val_file),
        },
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Dataset    : {name}")
    print(f"Answer keys: {', '.join(answer_keys)}")
    print(f"Train      : {len(train_samples)} samples ({len(train_labels)} items)")
    print(f"Validation : {len(val_samples)} samples ({len(validation_labels)} items)")
    print(f"Split      : {split_file}")
    print(f"Output     : {train_file}")
    print(f"Output     : {val_file}")
    print(f"Manifest   : {outdir / 'manifest.json'}")

    if split_validation and set(split_validation) != set(validation_labels):
        print(
            "[WARN] split 파일의 검증 항목과 현재 데이터셋의 검증 항목이 다릅니다. "
            "공정한 비교를 위해 --force-split 으로 다시 만드세요."
        )


if __name__ == "__main__":
    main()
