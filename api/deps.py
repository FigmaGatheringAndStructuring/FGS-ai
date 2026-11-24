"""공통 의존성."""

import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# Swagger(/docs) 오른쪽 위 Authorize 버튼으로 키를 한 번 넣으면 모든 요청에 붙는다.
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="AI_API_KEY 값. 서버에 AI_API_KEY가 설정돼 있을 때만 검사한다.",
)


def verify_api_key(x_api_key: str | None = Security(api_key_header)) -> None:
    """AI_API_KEY가 설정돼 있으면 X-API-Key 헤더가 일치해야함.
    터널 등으로 외부에 공개할 때는 반드시 설정할 것.
    """
    expected = os.getenv("AI_API_KEY")
    if not expected:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API 키가 올바르지 않습니다")
