"""Axum이 보낸 base64 이미지를 PIL 이미지로 바꾼다."""

import base64
import binascii
import io
import os

from PIL import Image, UnidentifiedImageError

from inference.errors import InvalidImageError

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(30 * 1024 * 1024)))  # 30MB
# 압축 폭탄 방지. 큰 피그마 보드도 들어올 수 있게 넉넉히 잡는다.
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(200_000_000)))


def decode_image(data: str) -> Image.Image:
    """base64 문자열(앞에 data:image/png;base64, 가 붙어 있어도 됨)을 RGB 이미지로 디코딩한다."""
    if data.startswith("data:"):
        _, _, data = data.partition(",")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise InvalidImageError("image가 올바른 base64 문자열이 아닙니다") from e

    if not raw:
        raise InvalidImageError("image가 비어 있습니다")
    if len(raw) > MAX_IMAGE_BYTES:
        raise InvalidImageError(f"이미지가 너무 큽니다 ({len(raw) // 1024 // 1024}MB > {MAX_IMAGE_BYTES // 1024 // 1024}MB)")

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise InvalidImageError(f"이미지를 열 수 없습니다: {e}") from e

    # 투명 배경(PNG)은 흰 배경으로 합친다. 피그마 보드 export는 투명일 수 있다.
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[-1])
        return background
    return img.convert("RGB")
