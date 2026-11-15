"""FastAPI: POST /estimate (кадр), WS /stream (поток), GET /health, демо на /demo."""
import os
import time
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from .config import settings
from .metrics import (HTTP_LATENCY, HTTP_REQUESTS, WS_CONNECTIONS, WS_FRAMES,
                      metrics_response, set_backend_info)
from .pipeline import Pipeline
from .schemas import EstimateResponse

app = FastAPI(title="AgeLive", version="0.1.0")
pipe = Pipeline(settings)
set_backend_info(pipe.info())


@app.middleware("http")
async def metrics_middleware(request, call_next):
    """RPS / латентность / коды ответов. WebSocket сюда не попадает (см. /stream)."""
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        # шаблон маршрута, а не сырой путь: иначе мусорные URL плодят бесконечные серии
        endpoint = getattr(route, "path", None) or "unmatched"
        if endpoint != "/metrics":
            HTTP_REQUESTS.labels(request.method, endpoint, str(status)).inc()
            HTTP_LATENCY.labels(request.method, endpoint).observe(time.perf_counter() - start)


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("invalid image")
    return img


@app.get("/health")
def health():
    return {"ok": True, **pipe.info()}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return metrics_response()


@app.post("/estimate", response_model=EstimateResponse)
async def estimate(file: UploadFile = File(...),
                   threshold: int = Query(18, ge=1, le=99),
                   session_id: Optional[str] = Query(None, max_length=64)):
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "image too large")
    try:
        frame = decode(data)
    except ValueError:
        raise HTTPException(400, "invalid image")
    return await run_in_threadpool(pipe.process, frame, session_id, threshold)


@app.websocket("/stream")
async def stream(ws: WebSocket, threshold: int = 18):
    """Клиент шлёт JPEG-кадры (binary), получает JSON на каждый кадр."""
    await ws.accept()
    WS_CONNECTIONS.inc()
    sid = uuid.uuid4().hex
    try:
        while True:
            data = await ws.receive_bytes()
            try:
                frame = decode(data)
            except ValueError:
                WS_FRAMES.labels("invalid_image").inc()
                await ws.send_json({"status": "error", "reasons": ["invalid_image"]})
                continue
            WS_FRAMES.labels("ok").inc()
            res = await run_in_threadpool(pipe.process, frame, sid, threshold)
            await ws.send_json(res)
    except WebSocketDisconnect:
        pass
    finally:
        WS_CONNECTIONS.dec()
        pipe.sessions.drop(sid)      # данные сессии не храним


_web = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.isdir(_web):
    app.mount("/demo", StaticFiles(directory=_web, html=True), name="demo")
