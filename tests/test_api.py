"""모델 없이 API 흐름을 확인한다. 실제 생성(_run_model)은 가짜로 바꾼다."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from api.main import app
from inference import markdown_generator, model_loader
from inference.model_loader import LoadedModel, ModelConfig
from tests.conftest import make_png_b64

FAKE_LOADED = LoadedModel(model=None, processor=None, config=ModelConfig(model_id="fake/qwen"))


def layout(**overrides):
    """FGS-backend GET /api/figma/page 응답과 같은 모양."""
    data = {
        "file_key": "abc123",
        "file_name": "스터디 앱",
        "page_id": "0:1",
        "page_name": "기획",
        "elements": [
            {"id": "1:1", "name": "기능", "element_type": "SECTION",
             "bounding_box": {"x": 100, "y": 200, "width": 640, "height": 480}},
        ],
        "frames": [],
        "bounding_box": {"x": 100, "y": 200, "width": 640, "height": 480},
    }
    data.update(overrides)
    return data


def body(**overrides):
    data = {"layout": layout(), "image": make_png_b64()}
    data.update(overrides)
    return data


@pytest.fixture
def client(monkeypatch):
    calls = []

    def fake_run(loaded, image, prompt, config):
        calls.append({"image": image, "prompt": prompt})
        return "```markdown\n# 스터디 앱\n\n## 개요\n- 소셜 로그인\n```", 120, 30

    monkeypatch.setattr(model_loader, "load_model", lambda config=None: FAKE_LOADED)
    monkeypatch.setattr(markdown_generator, "_run_model", fake_run)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.delenv("PRELOAD_MODEL", raising=False)
    with TestClient(app) as c:
        c.calls = calls
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": False, "model": None}


def test_generate_success(client):
    res = client.post("/generate", json=body())
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["markdown"] == "# 스터디 앱\n\n## 개요\n- 소셜 로그인\n"  # 코드 블록 벗겨짐
    assert data["model"] == "fake/qwen"
    assert data["usage"]["prompt_tokens"] == 120
    assert data["usage"]["output_tokens"] == 30
    assert data["filtered_elements"] == []

    prompt = client.calls[0]["prompt"]
    assert "[SECTION] 기능" in prompt and "# 스터디 앱" in prompt and "페이지 이름: 기획" in prompt
    assert client.calls[0]["image"].mode == "RGB"


def test_accepts_data_url_and_transparent_png(client):
    res = client.post("/generate", json=body(image="data:image/png;base64," + make_png_b64(mode="RGBA")))
    assert res.status_code == 200
    assert client.calls[0]["image"].mode == "RGB"


@pytest.mark.parametrize("image", ["not base64!!", "aGVsbG8="])  # 두 번째는 base64지만 이미지가 아님
def test_bad_image_returns_400(client, image):
    res = client.post("/generate", json=body(image=image))
    assert res.status_code == 400
    assert client.calls == []


def test_validation_error_returns_422(client):
    bad_layout = layout()
    del bad_layout["file_name"]
    assert client.post("/generate", json=body(layout=bad_layout)).status_code == 422
    assert client.post("/generate", json={"image": "x"}).status_code == 422
    # 백엔드 필드 이름(width/height)이 아니면 거부
    old_style = layout(bounding_box={"x": 0, "y": 0, "w": 1, "h": 1})
    assert client.post("/generate", json=body(layout=old_style)).status_code == 422


def test_api_key(client, monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "secret")
    assert client.post("/generate", json=body()).status_code == 401
    assert client.post("/generate", json=body(), headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/generate", json=body(), headers={"X-API-Key": "secret"}).status_code == 200


def test_model_load_failure_returns_503(client, monkeypatch):
    def broken(config=None):
        raise RuntimeError("CUDA not available")

    monkeypatch.setattr(model_loader, "load_model", broken)
    res = client.post("/generate", json=body())
    assert res.status_code == 503
    assert "CUDA not available" in res.json()["error"]


def test_busy_returns_503(client, monkeypatch):
    monkeypatch.setenv("GENERATE_QUEUE_TIMEOUT", "0.1")
    started, release = threading.Event(), threading.Event()

    def slow_run(loaded, image, prompt, config):
        started.set()
        release.wait(5)
        return "# ok", 1, 1

    monkeypatch.setattr(markdown_generator, "_run_model", slow_run)
    first = threading.Thread(target=lambda: client.post("/generate", json=body()))
    first.start()
    assert started.wait(5)
    try:
        res = client.post("/generate", json=body())
        assert res.status_code == 503
    finally:
        release.set()
        first.join()


def test_unexpected_error_returns_500(monkeypatch):
    def boom(loaded, image, prompt, config):
        raise ValueError("bug")

    monkeypatch.setattr(model_loader, "load_model", lambda config=None: FAKE_LOADED)
    monkeypatch.setattr(markdown_generator, "_run_model", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        res = c.post("/generate", json=body())
    assert res.status_code == 500
    assert res.json() == {"error": "서버 내부 오류: ValueError"}


def test_error_bodies_use_error_key(client, monkeypatch):
    res = client.post("/generate", json=body(image="aGVsbG8="))
    assert res.status_code == 400 and set(res.json()) == {"error"}

    bad_layout = layout()
    del bad_layout["file_name"]
    res = client.post("/generate", json=body(layout=bad_layout))
    assert res.status_code == 422
    assert set(res.json()) == {"error"}
    assert "body.layout.file_name" in res.json()["error"]

    monkeypatch.setenv("AI_API_KEY", "secret")
    res = client.post("/generate", json=body())
    assert res.status_code == 401 and set(res.json()) == {"error"}

    res = client.get("/nope")
    assert res.status_code == 404 and res.json() == {"error": "Not Found"}
