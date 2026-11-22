"""실제 모델 없이 가짜 torch/transformers로 로더 동작만 확인한다."""

import sys
import threading
import types

import pytest

from inference import model_loader
from inference.model_loader import ModelConfig


class FakeModel:
    device = "cpu"

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


class FakeProcessor:
    def __init__(self):
        self.image_processor = types.SimpleNamespace(size={"shortest_edge": 1, "longest_edge": 2})


@pytest.fixture
def fake_libs(monkeypatch):
    calls = {"model": 0}

    class AutoModelForImageTextToText:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model"] += 1
            return FakeModel({"model_id": model_id, **kwargs})

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return FakeProcessor()

    torch = types.ModuleType("torch")
    torch.bfloat16 = "torch.bfloat16"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForImageTextToText = AutoModelForImageTextToText
    transformers.AutoProcessor = AutoProcessor
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model_loader.unload_model()
    yield calls
    model_loader.unload_model()


def test_loads_once_and_caches(fake_libs):
    first = model_loader.load_model(ModelConfig(model_id="m"))
    second = model_loader.load_model(ModelConfig(model_id="other"))
    assert first is second
    assert fake_libs["model"] == 1
    assert first.model.eval_called
    assert model_loader.is_loaded()


def test_concurrent_calls_load_once(fake_libs):
    threads = [threading.Thread(target=model_loader.load_model, args=(ModelConfig(),)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake_libs["model"] == 1


def test_dtype_and_pixels_are_applied(fake_libs):
    loaded = model_loader.load_model(ModelConfig(dtype="bfloat16", max_pixels=1000))
    assert loaded.model.kwargs["dtype"] == "torch.bfloat16"
    assert loaded.processor.image_processor.size == {"shortest_edge": 1, "longest_edge": 1000}


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("QWEN_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
    monkeypatch.setenv("QWEN_MAX_PIXELS", "4000000")
    cfg = ModelConfig.from_env()
    assert cfg.model_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert cfg.max_pixels == 4_000_000

    monkeypatch.setenv("QWEN_DTYPE", "int3")
    with pytest.raises(ValueError):
        ModelConfig.from_env()
