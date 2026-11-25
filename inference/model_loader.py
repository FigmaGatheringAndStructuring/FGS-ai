"""
    QWEN_MODEL_ID     Hugging Face 모델 id 또는 로컬 경로 (기본: Qwen/Qwen3-VL-8B-Instruct)
    QWEN_DTYPE        auto | bfloat16 | float16 | float32 (기본: auto, 모델 설정값 사용)
    QWEN_DEVICE_MAP   auto | cuda | cpu ... (기본: auto)
    QWEN_ATTN         sdpa | flash_attention_2 | eager (기본: sdpa)
    QWEN_MIN_PIXELS   이미지 최소 픽셀 수 (선택)
    QWEN_MAX_PIXELS   이미지 최대 픽셀 수 (선택, 큰 보드에서 메모리 조절용)
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any
from peft import PeftModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_LORA_MODEL_ID = "./training/outputs/fgs-qwen3-vl-lora"
_DTYPES = {"auto", "bfloat16", "float16", "float32"}


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    lora_model_id: str | None = DEFAULT_LORA_MODEL_ID
    dtype: str = "auto"
    device_map: str = "auto"
    attn_implementation: str = "sdpa"
    min_pixels: int | None = None
    max_pixels: int | None = None

    @classmethod
    def from_env(cls) -> "ModelConfig":
        dtype = os.getenv("QWEN_DTYPE", "auto")
        if dtype not in _DTYPES:
            raise ValueError(f"QWEN_DTYPE은 {sorted(_DTYPES)} 중 하나여야 합니다: {dtype!r}")
        return cls(
            model_id=os.getenv("QWEN_MODEL_ID", DEFAULT_MODEL_ID),
            lora_model_id=os.getenv(
                "QWEN_LORA_MODEL_ID",
                DEFAULT_LORA_MODEL_ID,
            ),
            dtype=dtype,
            device_map=os.getenv("QWEN_DEVICE_MAP", "auto"),
            attn_implementation=os.getenv("QWEN_ATTN", "sdpa"),
            min_pixels=_int_env("QWEN_MIN_PIXELS"),
            max_pixels=_int_env("QWEN_MAX_PIXELS"),
        )


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    processor: Any
    config: ModelConfig


def _int_env(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


_lock = threading.Lock()
_loaded: LoadedModel | None = None


def load_model(config: ModelConfig | None = None) -> LoadedModel:
    global _loaded
    if _loaded is not None:
        return _loaded
    with _lock:
        if _loaded is None:
            _loaded = _load(config or ModelConfig.from_env())
    return _loaded


def is_loaded() -> bool:
    return _loaded is not None


def current() -> LoadedModel | None:
    return _loaded


def unload_model() -> None:
    # 서버 종료, 테스트, 모델 교체시 모델 내리기
    global _loaded
    with _lock:
        _loaded = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _load(config: ModelConfig) -> LoadedModel:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    logger.info("모델 로딩 시작: %s (dtype=%s, device_map=%s, attn=%s)",
                config.model_id, config.dtype, config.device_map, config.attn_implementation)
    started = time.perf_counter()

    dtype = config.dtype if config.dtype == "auto" else getattr(torch, config.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        dtype=dtype,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
    )

    if config.lora_model_id:
        logger.info("LoRA 어댑터 로딩: %s", config.lora_model_id)

        model = PeftModel.from_pretrained(
            model,
            config.lora_model_id,
        )

    model.eval()

    processor = AutoProcessor.from_pretrained(config.model_id)
    if config.min_pixels is not None or config.max_pixels is not None:
        size = dict(processor.image_processor.size)
        if config.min_pixels is not None:
            size["shortest_edge"] = config.min_pixels
        if config.max_pixels is not None:
            size["longest_edge"] = config.max_pixels
        processor.image_processor.size = size

    logger.info("모델 로딩 완료: %.1f초, device=%s", time.perf_counter() - started, model.device)
    if torch.cuda.is_available():
        logger.info("GPU 메모리 사용량: %.1fGB", torch.cuda.memory_allocated() / 1024**3)
    return LoadedModel(model=model, processor=processor, config=config)
