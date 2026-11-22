"""

    QWEN_MAX_NEW_TOKENS       생성할 최대 토큰 수 (기본 2048)
    QWEN_TEMPERATURE          0이면 greedy, 0보다 크면 샘플링 (기본 0.3)
    QWEN_TOP_P / QWEN_TOP_K   샘플링할 때만 사용 (기본 0.8 / 20)
    QWEN_REPETITION_PENALTY   반복 억제 (기본 1.05)
    MAX_LAYOUT_TEXT_CHARS     프롬프트에 넣을 요소 목록 최대 길이 (기본 30000자)
    GENERATE_QUEUE_TIMEOUT    GPU 차례를 기다리는 최대 시간(초) (기본 300)
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image

from api.schemas.generate import FigmaBoundingBox, FigmaPageElement, FigmaPageLayout
from inference import model_loader
from inference.errors import ModelBusyError, ModelUnavailableError

logger = logging.getLogger(__name__)

SECTIONS = ["개요", "주요 기능", "요구사항", "화면 흐름", "미정 사항"]

SYSTEM_PROMPT = """너는 피그마(Figma) 기획 보드를 읽고 정리된 기획 문서를 작성하는 도우미다.
입력으로 보드 페이지 전체 이미지와, 페이지의 요소 목록(프레임·섹션·그룹 등의 종류, 이름, 이미지 안에서의 위치)이 주어진다.

규칙:
- 보드의 내용(메모, 글자)은 이미지에서 정확히 읽는다.
- 요소 목록은 보드가 어떤 영역으로 나뉘어 있는지 파악하는 데 쓴다. 요소 이름은 그 영역의 주제일 가능성이 높다.
- 보드에 없는 내용은 절대 지어내지 않는다.
- 가까이 모여 있는 메모, 화살표로 이어진 메모, 순서대로 나열된 메모처럼 이미지에서 보이는 관계를 반영한다.
- 확정되지 않았거나 의견이 갈린 내용, 물음표가 붙은 메모는 '미정 사항'에 넣는다.
- 한국어로 작성하고, Markdown만 출력한다. 코드 블록(```)으로 감싸지 않는다."""


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 2048
    temperature: float = 0.3
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    max_layout_text_chars: int = 30_000
    queue_timeout: float = 300.0

    @classmethod
    def from_env(cls) -> "GenerationConfig":
        d = cls()

        def env(name: str, default: Any, cast: type) -> Any:
            value = os.getenv(name)
            return cast(value) if value else default  # 비어 있으면 기본값

        return cls(
            max_new_tokens=env("QWEN_MAX_NEW_TOKENS", d.max_new_tokens, int),
            temperature=env("QWEN_TEMPERATURE", d.temperature, float),
            top_p=env("QWEN_TOP_P", d.top_p, float),
            top_k=env("QWEN_TOP_K", d.top_k, int),
            repetition_penalty=env("QWEN_REPETITION_PENALTY", d.repetition_penalty, float),
            max_layout_text_chars=env("MAX_LAYOUT_TEXT_CHARS", d.max_layout_text_chars, int),
            queue_timeout=env("GENERATE_QUEUE_TIMEOUT", d.queue_timeout, float),
        )


@dataclass(frozen=True)
class GenerationResult:
    markdown: str
    prompt_tokens: int
    output_tokens: int
    elapsed_ms: int
    model_id: str


# GPU 하나에서 생성은 한 번에 하나씩만 한다.
_gpu_lock = threading.Lock()


# ---------------- 프롬프트 ----------------

def _to_image_coords(
    box: FigmaBoundingBox, origin: FigmaBoundingBox, image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """피그마 절대 좌표를 렌더 이미지의 픽셀 좌표로 바꾼다.

    FGS-backend는 layout.bounding_box 영역을 이미지 (0, 0)부터 scale배로 렌더하므로,
    이미지 크기 / bounding_box 크기가 곧 scale이다.
    """
    sx = image_size[0] / origin.width if origin.width > 0 else 1.0
    sy = image_size[1] / origin.height if origin.height > 0 else 1.0
    return (
        round((box.x - origin.x) * sx),
        round((box.y - origin.y) * sy),
        round(box.width * sx),
        round(box.height * sy),
    )


def _contains(outer: FigmaBoundingBox, inner: FigmaBoundingBox) -> bool:
    cx, cy = inner.x + inner.width / 2, inner.y + inner.height / 2
    return outer.x <= cx <= outer.x + outer.width and outer.y <= cy <= outer.y + outer.height


def build_layout_text(
    layout: FigmaPageLayout, elements: list[FigmaPageElement], image_size: tuple[int, int]
) -> str:
    """요소 목록을 위→아래, 왼→오른쪽 순서의 프롬프트용 텍스트로 만든다.

    최상위 요소(elements)를 먼저 쓰고, 그 안에 들어 있는 프레임(frames 중 최상위가 아닌 것)을
    들여쓰기로 붙인다.
    """
    def line(e: FigmaPageElement, indent: str) -> str:
        x, y, w, h = _to_image_coords(e.bounding_box, layout.bounding_box, image_size)
        name = " ".join(e.name.split()) or "(이름 없음)"
        return f"{indent}- [{e.element_type}] {name} (x={x}, y={y}, w={w}, h={h})"

    def order(e: FigmaPageElement) -> tuple[float, float]:
        return (e.bounding_box.y, e.bounding_box.x)

    top_ids = {e.id for e in layout.elements}
    nested = sorted((f for f in layout.frames if f.id not in top_ids), key=order)

    lines: list[str] = []
    for e in sorted(elements, key=order):
        lines.append(line(e, ""))
        lines += [line(f, "  ") for f in nested if _contains(e.bounding_box, f.bounding_box)]
    return "\n".join(lines)


def build_user_prompt(
    layout: FigmaPageLayout,
    elements: list[FigmaPageElement],
    image_size: tuple[int, int],
    max_chars: int,
) -> str:
    title = layout.file_name
    layout_text = build_layout_text(layout, elements, image_size) or "(요소 정보 없음, 이미지만 보고 작성)"
    if len(layout_text) > max_chars:
        logger.warning("요소 목록이 %d자라 %d자로 자름", len(layout_text), max_chars)
        layout_text = layout_text[:max_chars] + "\n... (이하 생략)"

    sections = "\n".join(f"## {s}" for s in SECTIONS)
    return f"""파일 이름: {title}
페이지 이름: {layout.page_name}
이미지 크기: {image_size[0]}x{image_size[1]}

요소 목록 (형식: - [종류] 이름 (이미지 안의 x, y, 너비, 높이)):
{layout_text}

위 보드를 아래 목차의 기획 문서로 정리해 줘. 첫 줄은 '# {title}'로 시작해.
{sections}

해당 내용이 없는 항목은 '- 보드에 관련 내용 없음'이라고 적어."""


_FENCE = re.compile(r"^```(?:markdown|md)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def postprocess(text: str) -> str:
    """모델이 코드 블록으로 감싸서 답한 경우 벗겨 낸다."""
    text = text.strip()
    m = _FENCE.match(text)
    return (m.group(1) if m else text).strip() + "\n"


# ---------------- 추론 ----------------

def generate_markdown(
    layout: FigmaPageLayout,
    elements: list[FigmaPageElement],
    image: Image.Image,
    config: GenerationConfig | None = None,
) -> GenerationResult:
    """elements는 JEV 필터를 거친 최상위 요소 (보통 layout.elements의 일부)."""
    config = config or GenerationConfig.from_env()
    user_prompt = build_user_prompt(layout, elements, image.size, config.max_layout_text_chars)

    if not _gpu_lock.acquire(timeout=config.queue_timeout):
        raise ModelBusyError(f"{config.queue_timeout:.0f}초 동안 차례가 오지 않았습니다")
    try:
        try:
            loaded = model_loader.load_model()
        except Exception as e:
            logger.exception("모델 로딩 실패")
            raise ModelUnavailableError(f"모델을 불러오지 못했습니다: {e}") from e

        started = time.perf_counter()
        text, prompt_tokens, output_tokens = _run_model(loaded, image, user_prompt, config)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
    finally:
        _gpu_lock.release()

    logger.info("생성 완료: %dms, 입력 %d토큰, 출력 %d토큰", elapsed_ms, prompt_tokens, output_tokens)
    if output_tokens >= config.max_new_tokens:
        logger.warning("최대 토큰(%d)에 도달해 문서가 잘렸을 수 있음", config.max_new_tokens)

    return GenerationResult(
        markdown=postprocess(text),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        elapsed_ms=elapsed_ms,
        model_id=loaded.config.model_id,
    )


def _run_model(
    loaded: model_loader.LoadedModel,
    image: Image.Image,
    user_prompt: str,
    config: GenerationConfig,
) -> tuple[str, int, int]:
    """Qwen3-VL로 실제 생성. (생성 텍스트, 입력 토큰 수, 출력 토큰 수)를 반환한다."""
    import torch

    model: Any = loaded.model
    processor: Any = loaded.processor

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "repetition_penalty": config.repetition_penalty,
    }
    if config.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=config.temperature,
                          top_p=config.top_p, top_k=config.top_k)
    else:
        gen_kwargs["do_sample"] = False

    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise ModelUnavailableError(
            "GPU 메모리가 부족합니다. QWEN_MAX_PIXELS를 줄이거나 더 작은 모델을 쓰세요"
        ) from e

    new_ids = output_ids[:, prompt_tokens:]
    text = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text, prompt_tokens, int(new_ids.shape[1])
