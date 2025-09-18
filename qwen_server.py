"""OpenAI-compatible endpoint that proxies Qwen3 Coder."""
from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, AsyncIterator, Tuple

import httpx
import threading
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIStatusError, OpenAI
from pydantic import BaseModel, Field, ConfigDict

QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_OAUTH_TOKEN_ENDPOINT = "https://chat.qwen.ai/api/v1/oauth2/token"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CREDENTIAL_ENV = os.environ.get("QWEN_OAUTH_CREDENTIAL_PATH")
CREDENTIAL_PATH = Path(CREDENTIAL_ENV).expanduser() if CREDENTIAL_ENV else Path("~/.qwen/oauth_creds.json").expanduser()
MODEL_ID = "qwen3-coder-plus"
TOKEN_REFRESH_BUFFER_MS = 30_000
TOKEN_REFRESH_LOOKAHEAD_MS = 3_600_000
TOKEN_REFRESH_COOLDOWN_MS = 60_000
API_KEY = os.environ.get("OPENAI_GATEWAY_API_KEY")


@dataclass
class QwenCredentials:
    access_token: str
    refresh_token: str
    token_type: str
    expiry_date: int
    resource_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QwenCredentials":
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            token_type=data["token_type"],
            expiry_date=int(data["expiry_date"]),
            resource_url=data.get("resource_url"),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expiry_date": self.expiry_date,
        }
        if self.resource_url:
            data["resource_url"] = self.resource_url
        return data

    def is_valid(self) -> bool:
        return int(time.time() * 1000) < self.expiry_date - TOKEN_REFRESH_BUFFER_MS


class QwenSession:
    def __init__(self, cred_path: Path = CREDENTIAL_PATH) -> None:
        self.cred_path = cred_path
        self._creds: Optional[QwenCredentials] = None
        self._last_refresh_attempt_ms: int = 0

    def load_credentials(self) -> QwenCredentials:
        data = json.loads(self.cred_path.read_text(encoding="utf-8"))
        self._creds = QwenCredentials.from_dict(data)
        return self._creds

    def ensure_credentials(self) -> QwenCredentials:
        creds = self._creds or self.load_credentials()
        now_ms = int(time.time() * 1000)
        if now_ms >= creds.expiry_date - TOKEN_REFRESH_LOOKAHEAD_MS:
            if now_ms - self._last_refresh_attempt_ms > TOKEN_REFRESH_COOLDOWN_MS:
                creds = self.refresh_credentials(creds)
        if not creds.is_valid():
            creds = self.refresh_credentials(creds)
        self._creds = creds
        return creds

    def refresh_credentials(self, creds: Optional[QwenCredentials] = None) -> QwenCredentials:
        creds = creds or self._creds or self.load_credentials()

        def do_refresh() -> QwenCredentials:
            response = httpx.post(
                QWEN_OAUTH_TOKEN_ENDPOINT,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": creds.refresh_token,
                    "client_id": QWEN_OAUTH_CLIENT_ID,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                timeout=httpx.Timeout(600, connect=60),
            )
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(
                    f"Token refresh failed: {payload['error']}: {payload.get('error_description')}"
                )

            updated = QwenCredentials(
                access_token=payload["access_token"],
                refresh_token=payload.get("refresh_token", creds.refresh_token),
                token_type=payload["token_type"],
                expiry_date=int(time.time() * 1000) + int(payload["expires_in"]) * 1000,
                resource_url=creds.resource_url,
            )
            self.cred_path.parent.mkdir(parents=True, exist_ok=True)
            self.cred_path.write_text(json.dumps(updated.to_dict(), indent=2), encoding="utf-8")
            self._last_refresh_attempt_ms = int(time.time() * 1000)
            return updated

        return self._retry(do_refresh)

    def _build_client(self, creds: QwenCredentials) -> OpenAI:
        base_url = creds.resource_url or DEFAULT_BASE_URL
        if not base_url.startswith(("http://", "https://")):
            base_url = f"https://{base_url}"
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        return OpenAI(api_key=creds.access_token, base_url=base_url, timeout=600)

    def _retry(self, fn, *, attempts: int = 5, base_delay: float = 0.5, max_delay: float = 8.0):
        delay = base_delay
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception:
                traceback.print_exc()
                if attempt == attempts:
                    raise
                sleep_for = min(delay, max_delay) + random.uniform(0, 0.5)
                time.sleep(sleep_for)
                delay *= 2

    def create_chat_completion(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        model: str = MODEL_ID,
        retries: int = 4,
        **kwargs,
    ) -> Dict[str, Any]:
        extra_params = dict(kwargs)
        backoff = 0.5
        for attempt in range(1, retries + 1):
            creds = self.ensure_credentials()
            client = self._build_client(creds)

            try:
                api_params = {
                    "model": model,
                    "messages": list(messages),
                    "timeout": 600,
                }
                if "max_tokens" in extra_params:
                    api_params["max_completion_tokens"] = extra_params["max_tokens"]
                api_params.update(extra_params)

                response = client.chat.completions.create(**api_params)
                if hasattr(response, "model_dump"):
                    return response.model_dump(exclude_none=True)
                return response
            except TypeError as exc:
                if "unexpected keyword argument" in str(exc):
                    from fastapi import HTTPException

                    raise HTTPException(status_code=400, detail=f"Unsupported parameter: {str(exc)}")
                raise
            except APIStatusError as exc:
                if exc.status_code == 401:
                    traceback.print_exc()
                    self.refresh_credentials(creds)
                elif exc.status_code == 400:
                    from fastapi import HTTPException

                    raise HTTPException(status_code=400, detail=str(exc))
                else:
                    traceback.print_exc()
                    raise
            except Exception:
                traceback.print_exc()
                if attempt == retries:
                    raise

            time.sleep(min(backoff, 8.0) + random.uniform(0, 0.5))
            backoff *= 2

        raise RuntimeError("Failed to create chat completion after retries")

    def stream_chat_raw(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        model: str = MODEL_ID,
        retries: int = 4,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        extra_params = dict(kwargs)
        backoff = 0.5
        for attempt in range(1, retries + 1):
            creds = self.ensure_credentials()
            client = self._build_client(creds)

            try:
                api_params = {
                    "model": model,
                    "messages": list(messages),
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "timeout": 600,
                }
                if "max_tokens" in extra_params:
                    api_params["max_completion_tokens"] = extra_params["max_tokens"]
                api_params.update(extra_params)

                stream = client.chat.completions.create(**api_params)
                for chunk in stream:
                    if hasattr(chunk, "model_dump"):
                        yield chunk.model_dump(exclude_none=True)
                    else:
                        yield chunk
                return
            except TypeError as exc:
                if "unexpected keyword argument" in str(exc):
                    from fastapi import HTTPException

                    raise HTTPException(status_code=400, detail=f"Unsupported parameter: {str(exc)}")
                raise
            except APIStatusError as exc:
                if exc.status_code == 401:
                    traceback.print_exc()
                    self.refresh_credentials(creds)
                elif exc.status_code == 400:
                    from fastapi import HTTPException

                    raise HTTPException(status_code=400, detail=str(exc))
                else:
                    traceback.print_exc()
                    raise
            except Exception:
                traceback.print_exc()
                if attempt == retries:
                    raise

            time.sleep(min(backoff, 8.0) + random.uniform(0, 0.5))
            backoff *= 2


def append_thinking_instruction(messages: List[ChatMessage]) -> None:
    """Append the think instruction to the last user message if not already present."""
    for msg in reversed(messages):
        if msg.role.lower() == "user":
            content = msg.content or ""
            if THINK_PROMPT not in content:
                if content and not content.endswith("\n"):
                    content += "\n"
                content += THINK_PROMPT
                msg.content = content
            break


THINK_PROMPT = (
    "Please first write down your in-depth thinking process within <think>…, and then provide the final answer."
)

app = FastAPI(title="Qwen OpenAI-Compatible Gateway")


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = Field(default=None, alias="tool_call_id")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, alias="tool_calls")

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    enable_thinking: bool = Field(default=False, alias="enable_thinking")

    model_config = ConfigDict(extra="allow", populate_by_name=True)


def normalize_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not any(msg.role.lower() == "system" for msg in messages):
        normalized.append({"role": "system", "content": "You are a helpful assistant."})

    for msg in messages:
        payload = msg.model_dump(by_alias=True, exclude_none=True)
        if "content" not in payload:
            payload["content"] = ""
        normalized.append(payload)
    return normalized


def _authorize(authorization: Optional[str]) -> None:
    if API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})
        token = authorization[len("Bearer "):].strip()
        if token != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})


def _prepare_request(
    request: ChatRequest,
    authorization: Optional[str],
) -> Tuple[QwenSession, List[Dict[str, Any]], Dict[str, Any]]:
    _authorize(authorization)

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    messages = [ChatMessage(**msg.model_dump()) for msg in request.messages]
    if request.enable_thinking:
        append_thinking_instruction(messages)

    qwen_messages = normalize_messages(messages)

    request_dict = request.model_dump()
    forwarded_params = {
        k: v
        for k, v in request_dict.items()
        if k not in {"model", "messages", "stream", "enable_thinking", "extra_body"} and v is not None
    }

    return QwenSession(), qwen_messages, forwarded_params


def _build_stream_chunk(
    completion_id: str,
    created: int,
    model: str,
    *,
    delta: Optional[Dict[str, Any]] = None,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    choice: Dict[str, Any] = {"index": 0}
    if delta is not None:
        choice["delta"] = delta
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason

    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }


def _build_usage_chunk(
    completion_id: str,
    created: int,
    model: str,
    usage: Dict[str, Any],
) -> Dict[str, Any]:
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _format_sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _claude_stream_chunks(result: Dict[str, Any]) -> Iterator[str]:
    choices = result.get("choices", [])
    if not choices:
        yield "data: [DONE]\n\n"
        return

    choice = choices[0]
    message = choice.get("message", {})
    completion_id = result.get("id") or uuid.uuid4().hex
    created = result.get("created", int(time.time()))
    model = result.get("model", "")

    yield _format_sse(
        _build_stream_chunk(
            completion_id,
            created,
            model,
            delta={"role": "assistant"},
        )
    )

    if message.get("content"):
        yield _format_sse(
            _build_stream_chunk(
                completion_id,
                created,
                model,
                delta={"content": message["content"]},
            )
        )

    reasoning = message.get("reasoning_content")
    if reasoning:
        yield _format_sse(
            _build_stream_chunk(
                completion_id,
                created,
                model,
                delta={"reasoning_content": reasoning},
            )
        )

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        deltas: List[Dict[str, Any]] = []
        for index, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue
            deltas.append(
                {
                    "index": index,
                    "id": call.get("id"),
                    "type": call.get("type", "function"),
                    "function": call.get("function", {}),
                }
            )
        if deltas:
            yield _format_sse(
                _build_stream_chunk(
                    completion_id,
                    created,
                    model,
                    delta={"tool_calls": deltas},
                )
            )

    yield "data: [DONE]\n\n"


async def create_completion(
    request: ChatRequest,
    authorization: Optional[str] = Header(default=None),
):
    session, qwen_messages, forwarded_params = _prepare_request(request, authorization)

    if request.stream:
        async def stream_generator() -> AsyncIterator[str]:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()
            stop_token = object()

            def worker() -> None:
                try:
                    for chunk in session.stream_chat_raw(
                        qwen_messages,
                        model=request.model,
                        **forwarded_params,
                    ):
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                except Exception as exc:  # pylint: disable=broad-except
                    asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(stop_token), loop)

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            while True:
                item = await queue.get()
                if item is stop_token:
                    break
                if isinstance(item, Exception):
                    raise item
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            session.create_chat_completion,
            qwen_messages,
            model=request.model,
            **forwarded_params,
        ),
    )
    return JSONResponse(result)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def openai_completion(
    request: ChatRequest,
    authorization: Optional[str] = Header(default=None),
):
    return await create_completion(request, authorization)


@app.post("/claude/v1/chat/completions")
@app.post("/claude/chat/completions")
async def claude_completion(
    request: ChatRequest,
    authorization: Optional[str] = Header(default=None),
):
    session, qwen_messages, forwarded_params = _prepare_request(request, authorization)

    if request.stream:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                session.create_chat_completion,
                qwen_messages,
                model=request.model,
                **forwarded_params,
            ),
        )

        async def claude_stream() -> AsyncIterator[str]:
            for chunk in _claude_stream_chunks(result):
                yield chunk

        return StreamingResponse(
            claude_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            session.create_chat_completion,
            qwen_messages,
            model=request.model,
            **forwarded_params,
        ),
    )
    return JSONResponse(result)


@app.post("/oauth/qwen/refresh")
async def manual_refresh_qwen_credentials(
    authorization: Optional[str] = Header(default=None),
):
    _authorize(authorization)
    session = QwenSession()
    loop = asyncio.get_running_loop()
    try:
        creds = await loop.run_in_executor(None, session.refresh_credentials)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Failed to refresh credentials: {exc}") from exc

    now_ms = int(time.time() * 1000)
    return {
        "status": "ok",
        "expiry_date": creds.expiry_date,
        "expires_in_ms": max(creds.expiry_date - now_ms, 0),
        "token_type": creds.token_type,
        "resource_url": creds.resource_url or DEFAULT_BASE_URL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("qwen_server:app", host="127.0.0.1", port=54434, reload=False)
