import base64
import io

import pytest
from PIL import Image

from inference import model_loader


def make_png_b64(size=(64, 48), mode="RGB") -> str:
    buf = io.BytesIO()
    Image.new(mode, size, (200, 100, 50, 128) if mode == "RGBA" else (200, 100, 50)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(autouse=True)
def _reset_model():
    model_loader.unload_model()
    yield
    model_loader.unload_model()
