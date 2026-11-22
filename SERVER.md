# DGX Spark 서버 작업 목록

AI 서버(figma-ai)를 DGX Spark에 올려서 실제 모델로 확인하기까지 할 일.
root 계정, 코드 위치 `~/fgs-ai` 기준. 끝난 항목은 `[x]`로 체크한다.

포트: FGS-backend `8000`, AI 서버 `8001`

---

## 1. 환경 준비 (처음 한 번)

- [ ] 시스템 패키지
  ```bash
  apt update
  apt install -y python3.12-venv python3-pip git tmux
  ```
- [ ] GPU 확인: `nvidia-smi`에서 GPU 이름과 CUDA Version 확인
- [ ] 코드 받기
  ```bash
  git clone https://github.com/juyoungk09/fgs-ai.git ~/fgs-ai
  cd ~/fgs-ai
  ```
- [ ] 가상환경
  ```bash
  rm -rf .venv
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  ```
- [ ] PyTorch (CUDA 빌드). `cu130`은 예시, `nvidia-smi`의 CUDA 버전에 맞춘다
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu130
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```
  → `True`와 GPU 이름이 나와야 함. `False`면 NGC PyTorch 컨테이너 사용 (README 참고)
- [ ] 나머지 패키지
  ```bash
  pip install -r requirements.txt
  python -c "import transformers; print(transformers.__version__)"   # 4.57 이상
  ```

## 2. 설정

- [ ] `.env` 만들기
  ```bash
  cp .env.example .env
  mkdir -p /data/hf-cache        # .env의 HF_HOME과 같은 경로
  nano .env
  ```
- [ ] 확인할 값

  | 변수 | 값 |
  |---|---|
  | `QWEN_MODEL_ID` | 처음엔 `Qwen/Qwen3-VL-8B-Instruct` |
  | `HF_HOME` | 용량 넉넉한 디스크 |
  | `PRELOAD_MODEL` | `1` |
  | `AI_API_KEY` | 백엔드가 같은 서버면 비워도 됨, 다른 컴퓨터면 반드시 설정 |

## 3. 모델 받기

- [ ] 8B 다운로드 (약 17GB)
  ```bash
  set -a; source .env; set +a
  hf download Qwen/Qwen3-VL-8B-Instruct
  ```
  `hf: command not found`면 `pip install -U huggingface_hub`

## 4. 테스트와 실행

- [ ] 단위 테스트 (모델 없이 돌아감): `pytest -q` → 전부 통과
- [ ] 서버 실행 (tmux 안에서)
  ```bash
  tmux new -s ai
  cd ~/fgs-ai && source .venv/bin/activate
  uvicorn api.main:app --host 127.0.0.1 --port 8001 --env-file .env
  ```
  `Ctrl+B` → `D`로 빠져나오기, `tmux attach -t ai`로 다시 들어가기
- [ ] 로그에서 확인
  - `모델 로딩 완료: N초`
  - `GPU 메모리 사용량: N GB`
- [ ] 헬스 체크
  ```bash
  curl http://127.0.0.1:8001/health
  # {"status":"ok","model_loaded":true,"model":"Qwen/Qwen3-VL-8B-Instruct"}
  ```

## 5. 실제 보드로 생성해 보기

FGS-backend로 레이아웃과 이미지를 받아서 AI 서버에 넣는다.
백엔드가 같은 서버의 8000번에서 돌고 있다고 가정한다. 다른 곳이면 주소만 바꾸면 된다.

- [ ] 테스트용 피그마 보드 URL 준비 (실제 기획 보드 2~3개)
- [ ] 레이아웃과 이미지 받기
  ```bash
  URL='https://www.figma.com/board/<파일키>/<이름>'
  ENC=$(python -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$URL")
  curl -s "http://127.0.0.1:8000/api/figma/page?url=$ENC" -o layout.json
  curl -s "http://127.0.0.1:8000/api/figma/page/render?url=$ENC&scale=1" -o page.png
  ```
- [ ] AI 서버 호출
  ```bash
  python - <<'PY'
  import base64, json, time, urllib.request
  body = {
      "layout": json.load(open("layout.json", encoding="utf-8")),
      "image": base64.b64encode(open("page.png", "rb").read()).decode(),
  }
  req = urllib.request.Request(
      "http://127.0.0.1:8001/generate",
      data=json.dumps(body).encode(),
      headers={"Content-Type": "application/json"},  # AI_API_KEY 설정했으면 "X-API-Key" 추가
  )
  t = time.time()
  res = json.load(urllib.request.urlopen(req, timeout=900))
  print(json.dumps(res["usage"], ensure_ascii=False), f"{time.time() - t:.1f}s")
  open("result.md", "w", encoding="utf-8").write(res["markdown"])
  PY
  cat result.md
  ```
- [ ] 결과 확인
  - [ ] `# 파일이름`으로 시작하고 목차 5개(개요, 주요 기능, 요구사항, 화면 흐름, 미정 사항)가 다 있음
  - [ ] 보드의 글자를 제대로 읽음 (작은 글씨, 한글)
  - [ ] 보드에 없는 내용을 지어내지 않음
  - [ ] 끝까지 생성됨 (로그에 `최대 토큰` 경고 없음)
  - [ ] 걸린 시간: ____초
  - [ ] 생성 중 GPU 메모리 (`nvidia-smi`): ____GB

## 6. 튜닝

문제가 있으면 `.env`를 고치고 서버를 다시 띄운다.

| 증상 | 바꿀 것 |
|---|---|
| 글자를 못 읽음 | 백엔드 render `scale` 올리기 (2 등), `QWEN_MAX_PIXELS` 올리기 |
| GPU 메모리 부족 (503) | `QWEN_MAX_PIXELS` 내리기, 4B 모델로 변경 |
| 문서가 중간에 끊김 | `QWEN_MAX_NEW_TOKENS` 올리기 (4096) |
| 같은 문장 반복 | `QWEN_REPETITION_PENALTY` 1.1 |
| 매번 결과가 너무 다름 | `QWEN_TEMPERATURE` 0 (항상 같은 결과) |
| 너무 느림 | 4B 모델, `QWEN_MAX_PIXELS` 내리기 |

- [ ] 모델 크기 비교 (같은 보드로)

  | 모델 | 시간 | GPU 메모리 | 글자 인식 | 문서 품질 (5점) |
  |---|---|---|---|---|
  | Qwen3-VL-4B-Instruct | | | | |
  | Qwen3-VL-8B-Instruct | | | | |
  | Qwen3-VL-30B-A3B-Instruct (여유 있으면) | | | | |

- [ ] 사용할 모델 확정 → `.env`의 `QWEN_MODEL_ID`

## 7. 백엔드와 연결

- [ ] 백엔드가 **같은 서버**: 지금 설정(`--host 127.0.0.1`) 그대로, 백엔드에서 `http://127.0.0.1:8001/generate` 호출
- [ ] 백엔드가 **다른 컴퓨터**:
  ```bash
  # .env에 AI_API_KEY=<긴 랜덤 문자열> 설정 후
  uvicorn api.main:app --host 0.0.0.0 --port 8001 --env-file .env
  ufw allow from <백엔드 IP> to any port 8001
  ```
- [ ] 백엔드 HTTP 클라이언트 타임아웃을 넉넉히 (10분)

## 8. 시연용 상시 실행 (선택)

SSH를 끊거나 서버가 재부팅돼도 자동으로 뜨게 하려면 systemd로 등록한다.

- [ ] `/etc/systemd/system/fgs-ai.service`
  ```ini
  [Unit]
  Description=FGS AI server
  After=network.target

  [Service]
  WorkingDirectory=/root/fgs-ai
  ExecStart=/root/fgs-ai/.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8001 --env-file .env
  Restart=on-failure

  [Install]
  WantedBy=multi-user.target
  ```
- [ ] 등록과 확인
  ```bash
  systemctl daemon-reload
  systemctl enable --now fgs-ai
  journalctl -u fgs-ai -f        # 로그 보기
  ```

## 9. 코드가 바뀌었을 때

```bash
cd ~/fgs-ai && git pull
source .venv/bin/activate
pip install -r requirements.txt
diff .env.example .env           # 새로 생긴 변수가 있으면 .env에 추가
pytest -q
# tmux 창에서 Ctrl+C 후 다시 실행, 또는 systemctl restart fgs-ai
```

## 10. 반납 전

- [ ] 시연 날짜가 대여 기간 안인지 확인
- [ ] 5번 결과(`result.md`)와 6번 비교표를 내 PC로 복사 (보고서용)
  ```bash
  scp root@<서버주소>:~/fgs-ai/result.md .
  ```
- [ ] `.env`의 토큰 등 개인 정보 삭제
