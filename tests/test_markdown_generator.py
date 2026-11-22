import contextlib
import sys
import types

import pytest
from PIL import Image

from api.schemas.generate import FigmaBoundingBox, FigmaPageElement, FigmaPageLayout
from inference.errors import ModelUnavailableError
from inference.markdown_generator import (
    GenerationConfig,
    _run_model,
    build_layout_text,
    build_user_prompt,
    postprocess,
)
from inference.model_loader import LoadedModel, ModelConfig


def el(id, name, x, y, w, h, kind="FRAME"):
    return FigmaPageElement(id=id, name=name, element_type=kind,
                            bounding_box=FigmaBoundingBox(x=x, y=y, width=w, height=h))


def make_layout(elements, frames=(), bounds=None):
    return FigmaPageLayout(
        file_key="k", file_name="스터디 앱", page_id="0:1", page_name="기획",
        elements=list(elements), frames=list(frames),
        bounding_box=bounds or FigmaBoundingBox(x=100, y=200, width=400, height=300),
    )


def test_layout_text_sorts_and_converts_to_image_coords():
    lower = el("1:2", "화면 흐름", 100, 400, 200, 100, "SECTION")
    upper = el("1:1", "로그인\n기능", 300, 200, 200, 100)
    layout = make_layout([lower, upper])
    # 이미지는 bounding_box(400x300)의 2배로 렌더됐다고 가정
    text = build_layout_text(layout, layout.elements, (800, 600))
    lines = text.splitlines()
    assert lines[0] == "- [FRAME] 로그인 기능 (x=400, y=0, w=400, h=200)"  # 위에 있는 것 먼저, 줄바꿈 정리
    assert lines[1] == "- [SECTION] 화면 흐름 (x=0, y=400, w=400, h=200)"


def test_nested_frames_are_listed_under_their_element():
    section = el("1:1", "기능", 100, 200, 400, 300, "SECTION")
    inner = el("2:1", "로그인 화면", 120, 220, 50, 50)
    layout = make_layout([section], frames=[inner])
    text = build_layout_text(layout, layout.elements, (400, 300))
    assert text.splitlines() == [
        "- [SECTION] 기능 (x=0, y=0, w=400, h=300)",
        "  - [FRAME] 로그인 화면 (x=20, y=20, w=50, h=50)",
    ]


def test_top_level_frames_are_not_duplicated():
    frame = el("1:1", "화면", 100, 200, 400, 300)
    layout = make_layout([frame], frames=[frame])  # 백엔드 frames에는 최상위 프레임도 들어 있음
    assert build_layout_text(layout, layout.elements, (400, 300)).count("화면") == 1


def test_prompt_truncates_long_layout_text():
    layout = make_layout([el(str(i), "가" * 100, 100, 200, 10, 10) for i in range(100)])
    prompt = build_user_prompt(layout, layout.elements, (400, 300), max_chars=500)
    assert "(이하 생략)" in prompt
    assert len(prompt) < 2000


def test_prompt_contains_title_and_page():
    layout = make_layout([])
    prompt = build_user_prompt(layout, [], (400, 300), max_chars=1000)
    assert "# 스터디 앱" in prompt and "페이지 이름: 기획" in prompt and "이미지만 보고" in prompt


@pytest.mark.parametrize("raw, expected", [
    ("```markdown\n# A\n```", "# A\n"),
    ("```\n# A\n```\n", "# A\n"),
    ("  # A\n\n", "# A\n"),
    ("# A\n```js\ncode\n```", "# A\n```js\ncode\n```\n"),  # 본문 안의 코드 블록은 유지
])
def test_postprocess(raw, expected):
    assert postprocess(raw) == expected


def test_config_from_env_treats_empty_as_default(monkeypatch):
    monkeypatch.setenv("QWEN_TEMPERATURE", "")
    monkeypatch.setenv("QWEN_MAX_NEW_TOKENS", "512")
    cfg = GenerationConfig.from_env()
    assert cfg.temperature == GenerationConfig().temperature
    assert cfg.max_new_tokens == 512


# ---------- _run_model: 가짜 torch / processor / model ----------

class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, idx):
        _, cols = idx
        return FakeTensor([r[cols] for r in self.rows])


class FakeInputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = {}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls["messages"] = messages
        return "PROMPT"

    def __call__(self, text, images, return_tensors):
        self.calls["images"] = images
        return FakeInputs(input_ids=FakeTensor([[1, 2, 3, 4]]))

    def batch_decode(self, ids, **kwargs):
        self.calls["decoded_ids"] = ids.rows
        return ["# 결과"]


class FakeModel:
    device = "cuda:0"

    def __init__(self, oom=False):
        self.oom = oom
        self.kwargs = None

    def generate(self, **kwargs):
        if self.oom:
            raise sys.modules["torch"].cuda.OutOfMemoryError()
        self.kwargs = kwargs
        return FakeTensor([[1, 2, 3, 4, 9, 9, 9]])


@pytest.fixture
def fake_torch(monkeypatch):
    class OOM(RuntimeError):
        pass

    torch = types.ModuleType("torch")
    torch.inference_mode = contextlib.nullcontext
    torch.cuda = types.SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None,
                                       is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)


def loaded_with(model, processor):
    return LoadedModel(model=model, processor=processor, config=ModelConfig())


def test_run_model_greedy(fake_torch):
    model, proc = FakeModel(), FakeProcessor()
    img = Image.new("RGB", (8, 8))
    text, p_tok, o_tok = _run_model(loaded_with(model, proc), img, "user", GenerationConfig(temperature=0))
    assert (text, p_tok, o_tok) == ("# 결과", 4, 3)
    assert proc.calls["decoded_ids"] == [[9, 9, 9]]  # 입력 부분은 잘라냄
    assert proc.calls["images"] == [img]
    assert model.kwargs["do_sample"] is False
    assert "temperature" not in model.kwargs
    user_content = proc.calls["messages"][1]["content"]
    assert user_content[0] == {"type": "image"}
    assert user_content[1]["text"] == "user"


def test_run_model_sampling(fake_torch):
    model = FakeModel()
    _run_model(loaded_with(model, FakeProcessor()), Image.new("RGB", (8, 8)), "u",
               GenerationConfig(temperature=0.5, top_p=0.9, top_k=10))
    assert model.kwargs["do_sample"] is True
    assert model.kwargs["temperature"] == 0.5
    assert model.kwargs["top_k"] == 10


def test_run_model_oom(fake_torch):
    with pytest.raises(ModelUnavailableError):
        _run_model(loaded_with(FakeModel(oom=True), FakeProcessor()), Image.new("RGB", (8, 8)), "u",
                   GenerationConfig())
