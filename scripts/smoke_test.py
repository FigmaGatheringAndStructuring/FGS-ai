"""AI 서버 /generate 동작 확인용 스크립트.

1) 예시 보드 자동 생성 (백엔드·피그마 없이)
   python scripts/smoke_test.py

2) 피그마에서 직접 export한 PNG로
   python scripts/smoke_test.py --image board.png

3) FGS-backend에서 받은 레이아웃 + 렌더 이미지로
   python scripts/smoke_test.py --image page.png --layout layout.json

옵션: --url http://127.0.0.1:8001  --api-key <AI_API_KEY>  --out result.md  --dry-run
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgun.ttf",
]

# 예시 보드: (섹션 이름, 메모들)
SAMPLE_SECTIONS = [
    ("로그인", ["카카오 로그인", "구글 로그인", "자동 로그인?"]),
    ("스터디 찾기", ["과목별 검색", "지역 필터", "최대 8명"]),
    ("채팅", ["그룹 채팅", "읽음 표시", "새 메시지 알림"]),
]
SCALE = 2  # 렌더 배율 (백엔드 render scale 역할)
ORIGIN = (1000, 500)  # 피그마 절대 좌표 시작점 (좌표 변환 확인용)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    sys.exit("한글 폰트가 없습니다. 서버에서: apt install -y fonts-nanum")


def make_sample_board() -> tuple[Image.Image, dict]:
    """섹션 3개짜리 예시 보드 이미지와 FigmaPageLayout 형식의 레이아웃을 만든다."""
    sec_w, sec_h, gap = 300, 380, 60  # 피그마 좌표 단위
    width = len(SAMPLE_SECTIONS) * sec_w + (len(SAMPLE_SECTIONS) + 1) * gap
    height = sec_h + 2 * gap
    img = Image.new("RGB", (width * SCALE, height * SCALE), "white")
    draw = ImageDraw.Draw(img)
    title_font, note_font = load_font(22 * SCALE), load_font(18 * SCALE)

    elements = []
    for i, (name, notes) in enumerate(SAMPLE_SECTIONS):
        x, y = gap + i * (sec_w + gap), gap
        s = lambda v: v * SCALE  # noqa: E731
        draw.rounded_rectangle([s(x), s(y), s(x + sec_w), s(y + sec_h)], radius=s(12),
                               fill="#f2f2f7", outline="#c7c7cc", width=s(2))
        draw.text((s(x + 16), s(y + 12)), name, fill="black", font=title_font)
        for j, note in enumerate(notes):
            ny = y + 60 + j * 105
            draw.rectangle([s(x + 20), s(ny), s(x + sec_w - 20), s(ny + 85)], fill="#fff3a3")
            draw.text((s(x + 34), s(ny + 26)), note, fill="black", font=note_font)
        elements.append({
            "id": f"1:{i + 1}", "name": name, "element_type": "SECTION",
            "bounding_box": {"x": ORIGIN[0] + x, "y": ORIGIN[1] + y, "width": sec_w, "height": sec_h},
        })

    # 로그인 → 스터디 찾기 화살표
    ax1, ax2, ay = (gap + sec_w + 5) * SCALE, (2 * gap + sec_w - 5) * SCALE, (gap + sec_h // 2) * SCALE
    draw.line([ax1, ay, ax2, ay], fill="#ff3b30", width=4 * SCALE)
    draw.polygon([(ax2, ay), (ax2 - 14 * SCALE, ay - 8 * SCALE), (ax2 - 14 * SCALE, ay + 8 * SCALE)], fill="#ff3b30")

    layout = {
        "file_key": "smoke", "file_name": "스터디 매칭 앱 기획", "page_id": "0:1", "page_name": "기획",
        "elements": elements, "frames": [],
        "bounding_box": {"x": ORIGIN[0], "y": ORIGIN[1], "width": width, "height": height},
    }
    return img, layout


def layout_for_image(img: Image.Image, name: str) -> dict:
    """레이아웃 없이 이미지만 있을 때 쓰는 최소 레이아웃 (보드 전체를 요소 하나로)."""
    box = {"x": 0, "y": 0, "width": img.width, "height": img.height}
    return {
        "file_key": "local", "file_name": name, "page_id": "0:1", "page_name": "Page 1",
        "elements": [{"id": "1:1", "name": name, "element_type": "FRAME", "bounding_box": box}],
        "frames": [], "bounding_box": box,
    }


def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--image")
    ap.add_argument("--layout")
    ap.add_argument("--api-key")
    ap.add_argument("--out", default="result.md")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--dry-run", action="store_true", help="요청은 보내지 않고 예시 이미지만 저장")
    args = ap.parse_args()

    expected: list[str] = []
    if args.image:
        img = Image.open(args.image).convert("RGB")
        layout = (json.loads(Path(args.layout).read_text(encoding="utf-8")) if args.layout
                  else layout_for_image(img, Path(args.image).stem))
    else:
        img, layout = make_sample_board()
        expected = [n for _, notes in SAMPLE_SECTIONS for n in notes]
        img.save("sample_board.png")
        print("예시 보드 저장: sample_board.png", img.size)
    if args.dry_run:
        print(json.dumps(layout, ensure_ascii=False, indent=2))
        return

    # 헬스 체크
    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=10) as r:
            print("health:", r.read().decode())
    except urllib.error.URLError as e:
        sys.exit(f"서버에 연결할 수 없습니다 ({args.url}): {e}")

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key
    body = json.dumps({"layout": layout, "image": to_b64(img)}).encode()
    req = urllib.request.Request(f"{args.url}/generate", data=body, headers=headers)

    print(f"생성 요청 중... (이미지 {img.width}x{img.height}, 요소 {len(layout['elements'])}개)")
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            res = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"오류 {e.code}: {e.read().decode()}")
    elapsed = time.time() - started

    Path(args.out).write_text(res["markdown"], encoding="utf-8")
    print("\n" + "=" * 60 + "\n" + res["markdown"] + "=" * 60)
    print(f"모델: {res['model']}")
    print(f"토큰: 입력 {res['usage']['prompt_tokens']}, 출력 {res['usage']['output_tokens']}")
    print(f"시간: 모델 {res['usage']['elapsed_ms'] / 1000:.1f}s / 전체 {elapsed:.1f}s")
    print(f"저장: {args.out}")

    md = res["markdown"]
    sections = ["개요", "주요 기능", "요구사항", "화면 흐름", "미정 사항"]
    print("\n[자동 점검]")
    print(f"- 제목으로 시작: {'O' if md.lstrip().startswith('# ') else 'X'}")
    missing = [s for s in sections if s not in md]
    print(f"- 목차 {len(sections) - len(missing)}/{len(sections)}" + (f" (빠짐: {', '.join(missing)})" if missing else ""))
    if expected:
        found = [t for t in expected if t.replace("?", "") in md]
        print(f"- 메모 인식 {len(found)}/{len(expected)}"
              + (f" (못 찾음: {', '.join(t for t in expected if t not in found)})" if len(found) < len(expected) else ""))
        print(f"- '자동 로그인?'이 미정 사항에 있음: {'O' if '자동 로그인' in md.split('미정 사항')[-1] else 'X'}")


if __name__ == "__main__":
    main()
