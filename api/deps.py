"""공통 의존성."""

import os
import secrets

from fastapi import Header, HTTPException, status


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """AI_API_KEY가 설정돼 있으면 X-API-Key 헤더가 일치해야함
    Axum을 외부 API로 빼면 0.0.0.0 으로 무조건 해야함
    """
    expected = os.getenv("AI_API_KEY")
    if not expected:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API 키가 올바르지 않습니다")
