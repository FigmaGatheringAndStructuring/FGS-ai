"""JEV 기반 요소-주제 연관성 판단."""

from api.schemas.generate import FigmaPageElement

THRESHOLD = 0.5  # TODO: 실험으로 조정


def filter_elements(
    title: str, elements: list[FigmaPageElement]
) -> tuple[list[FigmaPageElement], list[str]]:
    """(남길 요소, 뺀 요소 id 목록)을 반환한다.

    TODO: JEV 모델로 연관성 점수를 계산. 지금은 모든 요소를 그대로 통과시킨다.
    """
    return elements, []
