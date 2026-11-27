#!/usr/bin/env python3
"""같은 검증 데이터로 여러 모델(베이스 + LoRA 어댑터들)을 비교한다.

기본 사용 흐름:
    1) output2 만 사용한 데이터셋과 output1+output2 데이터셋을 만든다.
       python training/scripts/prepare_dataset.py --answer-keys output2
       python training/scripts/prepare_dataset.py --answer-keys output output2

    2) 각 데이터셋으로 LoRA 모델을 학습한다.
       python training/scripts/train.py \
         --train training/datasets/output2/train.jsonl \
         --validation training/datasets/output2/validation.jsonl \
         --output training/outputs/fgs-output2
       python training/scripts/train.py \
         --train training/datasets/output1_output2/train.jsonl \
         --validation training/datasets/output1_output2/validation.jsonl \
         --output training/outputs/fgs-output1-output2

    3) 세 모델을 같은 output2 검증셋으로 비교한다.
       python training/scripts/evaluate.py \
         --validation training/datasets/output2/validation.jsonl \
         --adapter output2=training/outputs/fgs-output2 \
         --adapter output1_output2=training/outputs/fgs-output1-output2 \
         --output training/eval_results

`--adapter` 를 주지 않으면 베이스 모델만 평가한다.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image
    from peft import PeftModel


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "training" / "datasets"

METRIC_KEYS = (
    "rouge_l_precision",
    "rouge_l_recall",
    "rouge_l_f1",
    "token_f1",
    "exact_match",
    "length_ratio",
    "prediction_tokens",
    "reference_tokens",
    "prompt_tokens",
    "output_tokens",
)

_FENCE_RE = re.compile(r"^```(?:markdown|md)?\s*\n(.*?)\n```\s*$", re.DOTALL)
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_MARKDOWN_CHARS = re.compile(r"[#>*_`~|]")


# ---------------- 파일/입력 ----------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_image_path(item: dict[str, Any], annotation_path: Path) -> Path:
    image_value = item.get("image")

    if not image_value:
        raise ValueError(f"image 경로가 없습니다: {item.get('id', '?')}")

    image_path = Path(str(image_value))

    if image_path.is_absolute():
        candidates = [image_path]
    else:
        candidates = [
            annotation_path.parent / image_path,
            DATASET_DIR / image_path,
            ROOT / image_path,
        ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"이미지를 찾을 수 없습니다: {image_value} (기준: {annotation_path.parent})"
    )


def message_text(item: dict[str, Any], role: str) -> str:
    for message in item.get("messages", []):
        if message.get("role") != role:
            continue

        content = message.get("content", "")

        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                    elif "text" in part:
                        parts.append(str(part["text"]))
                else:
                    parts.append(str(part))
            return "\n".join(part for part in parts if part)

        return str(content)

    raise ValueError(f"'{role}' 메시지가 없습니다: {item.get('id', '?')}")


def build_chat_messages(item: dict[str, Any], image: Image.Image) -> list[dict[str, Any]]:
    """train.py 의 FGS3VLDataset.build_messages 와 같은 형식."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": message_text(item, "system")}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": message_text(item, "user")},
            ],
        },
    ]


# ---------------- 생성 ----------------

def postprocess(text: str) -> str:
    """모델이 코드 블록으로 감싸서 답한 경우 벗겨 낸다."""
    text = text.strip()
    match = _FENCE_RE.match(text)
    return (match.group(1) if match else text).strip() + "\n"


def generate_one(
    model: Any,
    processor: Any,
    item: dict[str, Any],
    image_path: Path,
    args: argparse.Namespace,
) -> tuple[str, int, int]:
    import torch
    from PIL import Image

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    messages = build_chat_messages(item, image)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = inputs.to(model.device)

    prompt_tokens = int(inputs["input_ids"].shape[-1])

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
    }

    if args.temperature > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[:, prompt_tokens:]
    text = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return postprocess(text), prompt_tokens, int(new_ids.shape[-1])


# ---------------- 평가 지표 ----------------

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"```(?:markdown|md)?", " ", text)
    text = _MARKDOWN_CHARS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)

    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current

    return previous[-1]


def compute_metrics(prediction: str, reference: str) -> dict[str, float]:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    lcs = lcs_length(pred_tokens, ref_tokens)

    if lcs > 0 and pred_tokens and ref_tokens:
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        rouge_l = 2 * precision * recall / (precision + recall)
    else:
        precision = recall = rouge_l = 0.0

    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())

    if overlap > 0 and pred_tokens and ref_tokens:
        token_f1 = 2 * overlap / (len(pred_tokens) + len(ref_tokens))
    else:
        token_f1 = 0.0

    return {
        "rouge_l_precision": precision,
        "rouge_l_recall": recall,
        "rouge_l_f1": rouge_l,
        "token_f1": token_f1,
        "exact_match": float(normalize(prediction) == normalize(reference)),
        "length_ratio": (
            len(pred_tokens) / len(ref_tokens) if ref_tokens else 0.0
        ),
        "prediction_tokens": float(len(pred_tokens)),
        "reference_tokens": float(len(ref_tokens)),
    }


def average_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in METRIC_KEYS}

    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in METRIC_KEYS
    }


# ---------------- 모델 ----------------

def load_base_model(args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor_kwargs: dict[str, Any] = {}

    if args.min_pixels:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels:
        processor_kwargs["max_pixels"] = args.max_pixels

    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)

    torch_dtype: Any = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn,
    )
    model.config.use_cache = True
    model.eval()

    return processor, model


def release_adapter(model: PeftModel) -> None:
    import torch

    try:
        model.unload()
    except Exception:  # noqa: BLE001 - PEFT 버전에 따라 unload 가 없을 수 있음
        try:
            model.delete_adapter("default")
        except Exception:  # noqa: BLE001
            pass

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------- 실행 ----------------

def parse_adapter(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--adapter 는 '이름=어댑터경로' 형식이어야 합니다. 예: output2=training/outputs/fgs-output2"
        )

    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()

    if not name or not path:
        raise argparse.ArgumentTypeError(
            "--adapter 는 '이름=어댑터경로' 형식이어야 합니다."
        )

    return name, resolve_path(path)


def evaluate_variant(
    name: str,
    model: Any,
    processor: Any,
    items: list[dict[str, Any]],
    annotation_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, item in enumerate(items, start=1):
        image_path = resolve_image_path(item, annotation_path)

        prediction, prompt_tokens, output_tokens = generate_one(
            model,
            processor,
            item,
            image_path,
            args,
        )

        reference = message_text(item, "assistant")
        metrics = compute_metrics(prediction, reference)

        row = {
            "id": item.get("id"),
            "label": item.get("label"),
            "answer_key": item.get("answer_key"),
            "image": str(image_path),
            "reference": reference,
            "prediction": prediction,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            **metrics,
        }
        rows.append(row)

        print(
            f"[{name}] {index}/{len(items)} {row['id']} "
            f"ROUGE-L={metrics['rouge_l_f1']:.4f} "
            f"TokenF1={metrics['token_f1']:.4f}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    summary = average_metrics(rows)
    summary["elapsed_seconds"] = elapsed
    summary["seconds_per_sample"] = elapsed / len(items) if items else 0.0

    return {
        "name": name,
        "summary": summary,
        "samples": rows,
    }


def safe_filename(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", name).strip("_") or "model"


def write_report(
    path: Path,
    results: dict[str, Any],
) -> None:
    models: dict[str, Any] = results["models"]

    lines = [
        "# 모델 비교 결과",
        "",
        f"- 검증 데이터: `{results['validation']}`",
        f"- 검증 샘플 수: {results['num_samples']}",
        f"- 베이스 모델: `{results['model']}`",
        f"- 생성 설정: greedy={results['generation']['temperature'] == 0}, "
        f"max_new_tokens={results['generation']['max_new_tokens']}",
        "",
        "| 모델 | ROUGE-L F1 | Δ vs base | Token F1 | Exact Match | 길이 비율 | 초/샘플 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    base_rouge = models.get("base", {}).get("summary", {}).get("rouge_l_f1", 0.0)

    for name, model_result in models.items():
        summary = model_result["summary"]
        delta = summary["rouge_l_f1"] - base_rouge
        lines.append(
            f"| {name} | {summary['rouge_l_f1']:.4f} | {delta:+.4f} | "
            f"{summary['token_f1']:.4f} | "
            f"{summary['exact_match']:.4f} | {summary['length_ratio']:.3f} | "
            f"{summary['seconds_per_sample']:.2f} |"
        )

    best_name = max(models, key=lambda key: models[key]["summary"]["rouge_l_f1"])
    lines += [
        "",
        f"ROUGE-L F1 기준 최고 모델: **{best_name}**",
        "",
        "## 샘플별 결과",
        "",
        "각 모델의 예측/정답 전문은 `predictions_<모델>.jsonl` 에 저장됩니다.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="같은 검증 데이터로 베이스 모델과 LoRA 모델들을 비교합니다."
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="베이스 모델 id 또는 로컬 경로 (기본: Qwen/Qwen3-VL-4B-Instruct)",
    )
    parser.add_argument(
        "--validation",
        default="training/datasets/output2/validation.jsonl",
        help="검증 JSONL (기본: training/datasets/output2/validation.jsonl)",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        type=parse_adapter,
        metavar="NAME=PATH",
        help=(
            "비교할 LoRA 어댑터. 여러 번 지정할 수 있다. "
            "예: --adapter output2=training/outputs/fgs-output2"
        ),
    )
    parser.add_argument(
        "--output",
        default="training/eval_results",
        help="결과 디렉터리 (기본: training/eval_results)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="앞에서 N개만 평가 (0이면 전체)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="샘플링/셔플 시드 (기본: 42)",
    )

    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="모델 dtype (기본: auto)",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="transformers device_map (기본: auto)",
    )
    parser.add_argument(
        "--attn",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="sdpa",
        help="attention 구현 (기본: sdpa)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=401408,
        help="이미지 최소 픽셀 수 (기본: 학습과 동일한 401408)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1003520,
        help="이미지 최대 픽셀 수 (기본: 학습과 동일한 1003520)",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="생성 최대 토큰 수 (기본: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0이면 greedy (기본: 0.0, 재현 가능한 비교)",
    )
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    validation_path = resolve_path(args.validation)

    if not validation_path.is_file():
        raise SystemExit(f"검증 파일이 없습니다: {validation_path}")

    items = load_jsonl(validation_path)

    if args.limit > 0:
        items = items[: args.limit]

    if not items:
        raise SystemExit("평가할 검증 샘플이 없습니다.")

    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, Path | None]] = [("base", None)]
    variants += [(name, path) for name, path in args.adapter]

    seen_names: set[str] = set()
    for name, _ in variants:
        if name in seen_names:
            raise SystemExit(f"모델 이름이 중복됩니다: {name}")
        seen_names.add(name)

    print(f"검증 파일: {validation_path} ({len(items)} samples)")
    print(f"베이스 모델: {args.model}")
    print(f"비교 모델: {', '.join(name for name, _ in variants)}")

    processor, base_model = load_base_model(args)

    results: dict[str, Any] = {
        "model": args.model,
        "validation": str(validation_path),
        "num_samples": len(items),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
        "models": {},
    }

    try:
        for name, adapter_path in variants:
            model: Any = base_model

            if adapter_path is not None:
                from peft import PeftModel

                print(f"\n어댑터 로딩: {name} <- {adapter_path}", flush=True)

                if not adapter_path.exists():
                    raise SystemExit(f"어댑터 경로가 없습니다: {adapter_path}")

                model = PeftModel.from_pretrained(
                    base_model,
                    str(adapter_path),
                    is_trainable=False,
                )
                model.eval()

            print(f"\n=== {name} 평가 시작 ===", flush=True)

            try:
                model_result = evaluate_variant(
                    name,
                    model,
                    processor,
                    items,
                    validation_path,
                    args,
                )
            finally:
                if adapter_path is not None:
                    release_adapter(model)

            results["models"][name] = {
                "adapter": str(adapter_path) if adapter_path is not None else None,
                "summary": model_result["summary"],
            }

            predictions_path = output_dir / f"predictions_{safe_filename(name)}.jsonl"
            with predictions_path.open("w", encoding="utf-8") as f:
                for row in model_result["samples"]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            results["models"][name]["predictions_file"] = str(predictions_path)

            print(
                f"[{name}] ROUGE-L={model_result['summary']['rouge_l_f1']:.4f} "
                f"TokenF1={model_result['summary']['token_f1']:.4f} "
                f"({model_result['summary']['elapsed_seconds']:.1f}s)",
                flush=True,
            )
    finally:
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_path = output_dir / "report.md"
    write_report(report_path, results)

    print()
    print("=== 최종 비교 ===")
    print(f"{'모델':<20} {'ROUGE-L':>10} {'TokenF1':>10} {'Exact':>10}")
    for name, model_result in results["models"].items():
        summary = model_result["summary"]
        print(
            f"{name:<20} {summary['rouge_l_f1']:>10.4f} "
            f"{summary['token_f1']:>10.4f} {summary['exact_match']:>10.4f}"
        )

    print()
    print(f"결과: {results_path}")
    print(f"리포트: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
