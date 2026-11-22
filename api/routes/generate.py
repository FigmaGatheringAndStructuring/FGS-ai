import logging

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import verify_api_key
from api.schemas.generate import ErrorResponse, GenerateRequest, GenerateResponse, Usage
from inference.analyzer import filter_elements
from inference.errors import InvalidImageError, ModelBusyError, ModelUnavailableError
from inference.image import decode_image
from inference.markdown_generator import generate_markdown

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "이미지 디코딩 실패"},
        401: {"model": ErrorResponse, "description": "API 키 불일치"},
        422: {"model": ErrorResponse, "description": "요청 형식 오류"},
        500: {"model": ErrorResponse, "description": "서버 내부 오류"},
        503: {"model": ErrorResponse, "description": "모델 사용 불가 또는 대기 시간 초과"},
    },
)
def generate(req: GenerateRequest) -> GenerateResponse:
    # sync 함수라 FastAPI가 쓰레드풀에서 실행함
    try:
        image = decode_image(req.image)
    except InvalidImageError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    layout = req.layout
    logger.info("생성 요청하기: file=%r, page=%r, 요소 %d개, 프레임 %d개, 이미지 %dx%d",
                layout.file_name, layout.page_name, len(layout.elements), len(layout.frames), *image.size)

    kept, removed = filter_elements(layout.file_name, layout.elements)

    try:
        result = generate_markdown(layout, kept, image)
    except (ModelBusyError, ModelUnavailableError) as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e

    return GenerateResponse(
        markdown=result.markdown,
        filtered_elements=removed,
        model=result.model_id,
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            output_tokens=result.output_tokens,
            elapsed_ms=result.elapsed_ms,
        ),
    )
