"""
실행 명령어 uvicorn api.main:app --host 127.0.0.1 --port 8001 --env-file .env
PRELOAD_MODEL=1는 서버가 실행시 미리 모델 불러옴
포트는 8001 (FGS-backend가 8000을 씀)
오류 응답은 FGS-backend와 같은 {"error": "..."} 형식
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routes import generate
from inference import model_loader

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("PRELOAD_MODEL") == "1":
        # 미리 불러오다 실패하면 서버 강종
        model_loader.load_model()
    yield
    model_loader.unload_model()


app = FastAPI(title="MoEMoE뀽 AI Server", version="0.1.0", lifespan=lifespan)
app.include_router(generate.router)


def error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


@app.exception_handler(StarletteHTTPException)
async def http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    # 400, 401, 503 등 직접 낸 오류와 404, 405 같은 라우팅 오류
    return error_response(exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception(_: Request, exc: RequestValidationError) -> JSONResponse:
    # 예: "body.layout.file_name: Field required; body.image: Field required"
    message = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
    )
    return error_response(422, message or "요청 형식이 올바르지 않습니다")


@app.exception_handler(Exception)
async def unhandled_exception(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("처리하지 못한 오류")
    return error_response(500, f"서버 내부 오류: {type(exc).__name__}")


@app.get("/health")
def health() -> dict[str, str | bool | None]:
    loaded = model_loader.current()
    return {
        "status": "ok",
        "model_loaded": loaded is not None,
        "model": loaded.config.model_id if loaded else None,
    }
