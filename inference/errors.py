"""추론 단계에서 나는 오류. API 계층에서 HTTP 상태 코드로 바꾼다."""


class InferenceError(Exception):
    """추론 관련 오류의 공통 부모."""


class InvalidImageError(InferenceError):
    """이미지를 디코딩할 수 없거나 너무 큼 → 400"""


class ModelBusyError(InferenceError):
    """다른 요청이 GPU를 쓰고 있어 대기 시간 안에 차례가 오지 않음 → 503"""


class ModelUnavailableError(InferenceError):
    """모델 로딩 실패, GPU 메모리 부족 등 → 503"""
