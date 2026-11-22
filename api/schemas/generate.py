"""
Axum: GET /api/figma/page 가 돌려주는 FigmaPageLayout JSON을 그대로 layout에 넣고
GET /api/figma/page/render 로 받은 PNG를 base64로 image에 넣으면 됨
"""

from pydantic import BaseModel, Field


class FigmaBoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class FigmaPageElement(BaseModel):
    id: str
    name: str
    element_type: str  # FRAME, SECTION, GROUP, COMPONENT
    bounding_box: FigmaBoundingBox


class FigmaPageLayout(BaseModel):
    file_key: str
    file_name: str
    page_id: str
    page_name: str
    elements: list[FigmaPageElement] = Field(max_length=5000)  # 최상위 렌더 단위
    frames: list[FigmaPageElement] = Field(default=[], max_length=5000)  # 페이지 안의 모든 프레임
    bounding_box: FigmaBoundingBox  # elements 전체를 감싸는 렌더 이미지 영역)


class GenerateRequest(BaseModel):
    layout: FigmaPageLayout
    image: str = Field(min_length=1)  # 렌더 PNG의 base64 (data:image/png;base64, 접두어 허용)


class Usage(BaseModel):
    prompt_tokens: int
    output_tokens: int
    elapsed_ms: int


class GenerateResponse(BaseModel):
    markdown: str
    filtered_elements: list[str] = []  # 주제와 관계없다고 판단해 뺀 element id
    model: str
    usage: Usage


class ErrorResponse(BaseModel):
    """FGS-backend의 ErrorResponse와 같은 형식."""

    error: str
