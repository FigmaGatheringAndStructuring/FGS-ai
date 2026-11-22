# figma-ai

MoEMoE뀽의 AI 서버. Axum 백엔드만 호출하는 내부 서버다.

## 구조

- `api/` FastAPI 서버
  - `main.py` 앱, `/health`, 공통 오류 처리
  - `routes/generate.py` `POST /generate`
  - `schemas/generate.py` 요청·응답 형식
  - `deps.py` API 키 검사
- `inference/`
  - `model_loader.py` Qwen3-VL 로딩 (프로세스당 1번)
  - `markdown_generator.py` 프롬프트 구성, 생성, 후처리
  - `image.py` base64 이미지 디코딩
  - `analyzer.py` JEV 기반 요소 필터링 (TODO, 지금은 전부 통과)
  - `errors.py` 추론 오류 종류
- `training/` 데이터셋, 학습 스크립트, 설정
- `tests/`

## API

### `POST /generate`

요청은 FGS-backend 타입을 그대로 쓴다.

- `layout`: `GET /api/figma/page` 응답(`FigmaPageLayout`) 그대로
- `image`: `GET /api/figma/page/render` 로 받은 PNG를 base64로 인코딩한 값

```json
{
  "layout": {
    "file_key": "abc123",
    "file_name": "스터디 앱 기획",
    "page_id": "0:1",
    "page_name": "Page 1",
    "elements": [
      { "id": "1:2", "name": "기능 정리", "element_type": "SECTION",
        "bounding_box": { "x": 120, "y": 40, "width": 800, "height": 600 } }
    ],
    "frames": [
      { "id": "1:3", "name": "로그인 화면", "element_type": "FRAME",
        "bounding_box": { "x": 140, "y": 60, "width": 300, "height": 200 } }
    ],
    "bounding_box": { "x": 120, "y": 40, "width": 800, "height": 600 }
  },
  "image": "<base64 PNG, data:image/png;base64, 접두어 허용>"
}
```

요소 좌표는 이미지 크기와 `layout.bounding_box`로 렌더 scale을 계산해 이미지 픽셀 좌표로 바꾼 뒤 프롬프트에 넣는다.
보드의 글자 내용은 모델이 이미지에서 직접 읽는다.

응답 `200`
```json
{
  "markdown": "# 스터디 앱 기획\n\n## 개요\n...",
  "filtered_elements": [],
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "usage": { "prompt_tokens": 1830, "output_tokens": 642, "elapsed_ms": 21400 }
}
```

오류 응답은 FGS-backend의 `ErrorResponse`와 같은 형식이다.

```json
{ "error": "이미지를 열 수 없습니다: ..." }
```

| 코드 | 의미 | Axum에서 할 일 |
|---|---|---|
| 400 | 이미지 디코딩 실패 | 작업 실패 처리 (재시도 X) |
| 401 | `AI_API_KEY` 불일치 | 설정 확인 |
| 422 | 요청 형식 오류 | 작업 실패 처리 |
| 503 | 모델 사용 불가 / GPU 대기 시간 초과 | 잠시 후 재시도 |
| 500 | 서버 내부 오류 | 작업 실패 처리, 로그 확인 |

생성은 GPU 하나에서 한 번에 하나씩 처리한다. 요청 하나에 수십 초가 걸릴 수 있으므로
Axum의 HTTP 클라이언트 타임아웃은 넉넉하게(예: 10분) 잡는다.

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "model": "Qwen/Qwen3-VL-8B-Instruct" }
```

## 설치 (DGX Spark, Linux aarch64)

```bash
# 0. 확인: GPU와 CUDA 버전
nvidia-smi
python3 --version            # 3.10 이상

# 1. 가상환경
cd figma-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 2. PyTorch (CUDA 빌드) 먼저 설치
#    nvidia-smi에 나온 CUDA 버전에 맞는 명령을 https://pytorch.org/get-started/locally/ 에서 확인
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. 나머지 패키지
pip install -r requirements.txt

# 4. 환경 변수
cp .env.example .env         # 필요하면 값 수정

# 5. 모델 미리 받기 (선택, 8B 기준 약 17GB)
set -a; source .env; set +a
hf download Qwen/Qwen3-VL-8B-Instruct

# 6. 실행
uvicorn api.main:app --host 127.0.0.1 --port 8001 --env-file .env
curl http://127.0.0.1:8001/health    # {"status":"ok","model_loaded":true}
```

pip로 설치한 torch가 GPU를 못 잡으면(`cuda.is_available()`이 False) NVIDIA NGC PyTorch 컨테이너를 쓴다.

```bash
docker run --gpus all -it --rm -p 127.0.0.1:8001:8001 \
  -v $PWD:/app -v /data/hf-cache:/data/hf-cache -w /app \
  nvcr.io/nvidia/pytorch:<태그> bash
# 컨테이너 안에는 torch가 이미 있으므로 3단계부터 진행
pip install -r requirements.txt
```

## 로컬 개발 (Windows, 모델 없이)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install fastapi "uvicorn[standard]" pydantic pillow pytest httpx
pytest
uvicorn api.main:app --port 8001 --env-file .env   # PRELOAD_MODEL=0 으로 두기
```

모델 가중치는 저장소에 넣지 않는다.
