"""OpenAI-compatible endpoint that proxies Qwen3 Coder."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, AsyncIterator

import httpx
import threading
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from openai import APIStatusError, OpenAI
from pydantic import BaseModel, Field, ValidationError


QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_OAUTH_TOKEN_ENDPOINT = "https://chat.qwen.ai/api/v1/oauth2/token"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CREDENTIAL_ENV = os.environ.get("QWEN_OAUTH_CREDENTIAL_PATH")
CREDENTIAL_PATH = Path(CREDENTIAL_ENV).expanduser() if CREDENTIAL_ENV else Path("~/.qwen/oauth_creds.json").expanduser()
MODEL_ID = "qwen3-coder-plus"
TOKEN_REFRESH_BUFFER_MS = 30_000
API_KEY = os.environ.get("OPENAI_GATEWAY_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def load_credentials(self) -> QwenCredentials:
        data = json.loads(self.cred_path.read_text(encoding="utf-8"))
        self._creds = QwenCredentials.from_dict(data)
        return self._creds

    def ensure_credentials(self) -> QwenCredentials:
        creds = self._creds or self.load_credentials()
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
                raise RuntimeError(f"Token refresh failed: {payload['error']}: {payload.get('error_description')}")

            updated = QwenCredentials(
                access_token=payload["access_token"],
                refresh_token=payload.get("refresh_token", creds.refresh_token),
                token_type=payload["token_type"],
                expiry_date=int(time.time() * 1000) + int(payload["expires_in"]) * 1000,
                resource_url=creds.resource_url,
            )
            self.cred_path.parent.mkdir(parents=True, exist_ok=True)
            self.cred_path.write_text(json.dumps(updated.to_dict(), indent=2), encoding="utf-8")
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

    def stream_chat(
        self,
        messages: Iterable[Dict[str, Any]],
        *,
        model: str = MODEL_ID,
        retries: int = 4,
        **kwargs  # Accept ALL OpenAI parameters transparently
    ) -> Generator[Dict[str, Any], None, None]:
        backoff = 0.5
        for attempt in range(1, retries + 1):
            creds = self.ensure_credentials()
            client = self._build_client(creds)

            try:
                # Prepare base parameters
                api_params = {
                    "model": model,
                    "messages": list(messages),
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "timeout": 600,
                }

                # Handle max_tokens parameter mapping (OpenAI uses max_tokens, but we need max_completion_tokens)
                if "max_tokens" in kwargs:
                    api_params["max_completion_tokens"] = kwargs.pop("max_tokens")

                # Forward all other parameters transparently
                api_params.update(kwargs)

                stream = client.chat.completions.create(**api_params)
                yield from self._consume_stream(stream)
                return
            except TypeError as exc:
                # Handle unsupported parameters - convert to HTTP 400 error
                if "unexpected keyword argument" in str(exc):
                    logger.error(f"Unsupported parameter error: {exc}")
                    from fastapi import HTTPException
                    raise HTTPException(status_code=400, detail=f"Unsupported parameter: {str(exc)}")
                else:
                    logger.error(f"TypeError in stream_chat: {exc}")
                    raise
            except APIStatusError as exc:
                if exc.status_code == 401:
                    # Only print traceback for auth errors (for debugging)
                    logger.error(f"Authentication error (401): {exc}")
                    traceback.print_exc()
                    self.refresh_credentials(creds)
                elif exc.status_code == 400:
                    # Forward upstream 400 errors (invalid parameters) as HTTP exceptions
                    logger.error(f"Upstream API 400 error: {exc}")
                    logger.error(f"Request parameters: model={model}, messages_count={len(messages)}, kwargs={list(kwargs.keys())}")
                    from fastapi import HTTPException
                    raise HTTPException(status_code=400, detail=str(exc))
                else:
                    # For other API errors, print traceback and propagate
                    logger.error(f"Upstream API error {exc.status_code}: {exc}")
                    traceback.print_exc()
                    raise
            except Exception as exc:
                logger.error(f"Unexpected error in stream_chat attempt {attempt}/{retries}: {exc}")
                traceback.print_exc()
                if attempt == retries:
                    raise

            time.sleep(min(backoff, 8.0) + random.uniform(0, 0.5))
            backoff *= 2

    def _consume_stream(self, stream) -> Generator[Dict[str, Any], None, None]:
        full_content = ""
        inside_think = False
        pending = ""
        tool_calls_accumulator = {}  # Track tool calls by index

        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            delta: Dict[str, Any] = {}
            if choices:
                choice = choices[0]
                delta = choice.delta or {}
                if hasattr(delta, "model_dump"):
                    delta = delta.model_dump(exclude_none=True)

                new_text = delta.get("content")
                if new_text:
                    if new_text.startswith(full_content):
                        addition = new_text[len(full_content) :]
                    else:
                        addition = new_text
                    full_content = new_text

                    if addition:
                        pending += addition
                        while pending:
                            if inside_think:
                                end_idx = pending.find("</think>")
                                if end_idx != -1:
                                    if end_idx > 0:
                                        yield {"type": "reasoning", "text": pending[:end_idx]}
                                    pending = pending[end_idx + len("</think>") :]
                                    yield {"type": "reasoning_end"}
                                    inside_think = False
                                    continue

                                safe_len = len(pending) - (len("</think>") - 1)
                                if safe_len > 0:
                                    yield {"type": "reasoning", "text": pending[:safe_len]}
                                    pending = pending[safe_len:]
                                break
                            else:
                                start_idx = pending.find("<think>")
                                if start_idx != -1:
                                    if start_idx > 0:
                                        yield {"type": "text", "text": pending[:start_idx]}
                                    pending = pending[start_idx + len("<think>") :]
                                    yield {"type": "reasoning_start"}
                                    inside_think = True
                                    continue

                                safe_len = len(pending) - (len("<think>") - 1)
                                if safe_len > 0:
                                    yield {"type": "text", "text": pending[:safe_len]}
                                    pending = pending[safe_len:]
                                break

                reasoning = delta.get("reasoning_content")
                if reasoning:
                    yield {"type": "reasoning", "text": reasoning}

                # Handle tool_calls - accumulate them properly
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tool_call_delta in tool_calls:
                        index = tool_call_delta.get("index", 0)

                        # Initialize tool call if not exists
                        if index not in tool_calls_accumulator:
                            tool_calls_accumulator[index] = {
                                "id": tool_call_delta.get("id", ""),
                                "type": tool_call_delta.get("type", "function"),
                                "function": {
                                    "name": tool_call_delta.get("function", {}).get("name", ""),
                                    "arguments": ""
                                }
                            }

                        # Update tool call with delta
                        if "id" in tool_call_delta and tool_call_delta["id"]:
                            tool_calls_accumulator[index]["id"] = tool_call_delta["id"]

                        if "type" in tool_call_delta:
                            tool_calls_accumulator[index]["type"] = tool_call_delta["type"]

                        if "function" in tool_call_delta:
                            func_delta = tool_call_delta["function"]
                            if "name" in func_delta:
                                tool_calls_accumulator[index]["function"]["name"] = func_delta["name"]
                            if "arguments" in func_delta:
                                tool_calls_accumulator[index]["function"]["arguments"] += func_delta["arguments"]

                    # Yield the current state of tool calls
                    current_tool_calls = list(tool_calls_accumulator.values())
                    yield {"type": "tool_calls", "tool_calls": current_tool_calls}

            usage = getattr(chunk, "usage", None)
            if usage:
                info = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
                yield {
                    "type": "usage",
                    "input_tokens": info.get("prompt_tokens", 0),
                    "output_tokens": info.get("completion_tokens", 0),
                }

        if pending:
            if inside_think:
                yield {"type": "reasoning", "text": pending}
                yield {"type": "reasoning_end"}
            else:
                yield {"type": "text", "text": pending}


THINK_PROMPT = (
    "Please first write down your in-depth thinking process within <think>…, and then provide the final answer."
)

app = FastAPI(title="Qwen OpenAI-Compatible Gateway")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors (422)"""
    logger.error(f"Request validation error: {exc}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")

    # Try to get request body for debugging
    try:
        body = await request.body()
        if body:
            logger.error(f"Request body: {body.decode('utf-8')[:500]}...")
    except Exception:
        pass

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    logger.error(f"Pydantic validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatRequest(BaseModel):
    # Core required fields
    model: str
    messages: List[ChatMessage]

    # Service-specific fields
    stream: bool = False
    enable_thinking: bool = Field(default=False, alias="enable_thinking")

    # All other OpenAI parameters accepted transparently via extra="allow"
    class Config:
        extra = "allow"
        allow_population_by_field_name = True


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def create_completion(request: ChatRequest, authorization: Optional[str] = Header(default=None)):
    """Handle OpenAI-style chat completion requests."""
    # Log request details for debugging
    logger.info(f"Received request: model={request.model}, messages={len(request.messages)}, stream={request.stream}")

    if API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            logger.warning("Missing or invalid authorization header")
            raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})
        token = authorization[len("Bearer "):].strip()
        if token != API_KEY:
            logger.warning("Invalid API key provided")
            raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})

    if not request.messages:
        logger.error("Empty messages array in request")
        raise HTTPException(status_code=400, detail="messages must not be empty")

    messages = [ChatMessage(**msg.model_dump()) for msg in request.messages]
    if request.enable_thinking:
        append_thinking_instruction(messages)

    qwen_messages = normalize_messages(messages)

    # Extract all parameters transparently (exclude service-specific ones)
    request_dict = request.model_dump()
    forwarded_params = {
        k: v for k, v in request_dict.items()
        if k not in {"model", "messages", "stream", "enable_thinking"} and v is not None
    }

    # Log forwarded parameters for debugging
    if forwarded_params:
        logger.info(f"Forwarding parameters: {list(forwarded_params.keys())}")
        # Log tools specifically if present
        if "tools" in forwarded_params:
            tools_count = len(forwarded_params["tools"]) if isinstance(forwarded_params["tools"], list) else 0
            logger.info(f"Tools parameter: {tools_count} functions defined")
        if "tool_choice" in forwarded_params:
            logger.info(f"Tool choice: {forwarded_params['tool_choice']}")

    try:
        if request.stream:
            generator = stream_response(
                qwen_messages,
                model=request.model,
                **forwarded_params  # Forward all parameters transparently
            )
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )

        result = await run_completion(
            qwen_messages,
            model=request.model,
            **forwarded_params  # Forward all parameters transparently
        )

        # Log successful response details
        if result and "choices" in result and result["choices"]:
            message = result["choices"][0].get("message", {})
            has_tool_calls = "tool_calls" in message and message["tool_calls"]
            has_content = "content" in message and message["content"]
            logger.info(f"Response generated: tool_calls={bool(has_tool_calls)}, content={bool(has_content)}")
            if has_tool_calls:
                tool_names = [tc["function"]["name"] for tc in message["tool_calls"]]
                logger.info(f"Tool calls: {tool_names}")

        return JSONResponse(result)
    except HTTPException as http_exc:
        # Log HTTP exceptions for debugging
        logger.error(f"HTTP {http_exc.status_code} error: {http_exc.detail}")
        raise
    except Exception as exc:
        # Convert any other exceptions to HTTP 500
        logger.error(f"Unexpected server error: {exc}")
        logger.error(f"Request details: model={request.model}, messages_count={len(request.messages)}, stream={request.stream}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")


def append_thinking_instruction(messages: List[ChatMessage]) -> None:
    """Append the think instruction to the last user message if not already present."""
    for msg in reversed(messages):
        if msg.role.lower() == "user":
            if THINK_PROMPT not in msg.content:
                if not msg.content.endswith("\n"):
                    msg.content += "\n"
                msg.content += THINK_PROMPT
            break


def normalize_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Ensure there's a system prompt and convert to dict format."""
    normalized: List[Dict[str, Any]] = []
    has_system = any(msg.role.lower() == "system" for msg in messages)
    if not has_system:
        normalized.append({"role": "system", "content": "You are a helpful assistant."})

    # Convert messages to dict format, preserving all fields
    for msg in messages:
        message_dict = {"role": msg.role}

        # Add content if present (can be None for assistant messages with tool_calls)
        if msg.content is not None:
            message_dict["content"] = msg.content

        # Add tool_calls if present (for assistant messages)
        if msg.tool_calls is not None:
            message_dict["tool_calls"] = msg.tool_calls

        # Add tool_call_id if present (for tool messages)
        if msg.tool_call_id is not None:
            message_dict["tool_call_id"] = msg.tool_call_id

        # Add name if present (for tool messages)
        if msg.name is not None:
            message_dict["name"] = msg.name

        normalized.append(message_dict)

    return normalized



def _run_completion_sync(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str],
    **all_params  # Accept all parameters transparently
) -> Dict[str, object]:
    """Execute a non-streaming completion and return OpenAI-style payload."""
    session = QwenSession()
    display_model = model or "qwen3-coder-plus"
    qwen_model = model if model in {"qwen3-coder-plus", "qwen3-coder-flash"} else "qwen3-coder-plus"

    # Forward all parameters transparently to QwenSession
    events = session.stream_chat(
        messages,
        model=qwen_model,
        **all_params  # Pass everything through
    )

    answer_parts: List[str] = []
    reasoning_parts: List[str] = []
    final_tool_calls: List[Dict[str, Any]] = []
    usage_data = {"prompt_tokens": 0, "completion_tokens": 0}

    for event in events:
        etype = event.get("type")
        if etype == "text":
            answer_parts.append(event.get("text", ""))
        elif etype == "reasoning":
            reasoning_parts.append(event.get("text", ""))
        elif etype == "tool_calls":
            # Keep the latest tool_calls (they are cumulative)
            final_tool_calls = event.get("tool_calls", [])
        elif etype == "usage":
            usage_data = {
                "prompt_tokens": event.get("inputTokens", event.get("input_tokens", 0)),
                "completion_tokens": event.get("outputTokens", event.get("output_tokens", 0)),
            }

    completion_id = uuid.uuid4().hex
    created = int(time.time())

    content = "".join(answer_parts)
    reasoning = "".join(reasoning_parts)

    # Build message object
    message = {
        "role": "assistant",
        "content": content,
    }

    # Add reasoning_content if present
    if reasoning:
        message["reasoning_content"] = reasoning

    # Add tool_calls if present
    if final_tool_calls:
        message["tool_calls"] = final_tool_calls

    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": display_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage_data["prompt_tokens"],
            "completion_tokens": usage_data["completion_tokens"],
            "total_tokens": usage_data["prompt_tokens"] + usage_data["completion_tokens"],
        },
        "system_fingerprint": "",
    }

    return response


async def run_completion(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str],
    **all_params  # Accept all parameters transparently
) -> Dict[str, object]:
    loop = asyncio.get_running_loop()
    func = partial(
        _run_completion_sync,
        messages,
        model=model,
        **all_params  # Forward all parameters
    )
    return await loop.run_in_executor(None, func)


def _stream_response_sync(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str],
    **all_params  # Accept all parameters transparently
) -> Iterator[str]:
    """Yield SSE-formatted streaming responses."""
    session = QwenSession()
    display_model = model or "qwen3-coder-plus"
    qwen_model = model if model in {"qwen3-coder-plus", "qwen3-coder-flash"} else "qwen3-coder-plus"

    # Forward all parameters transparently to QwenSession
    events = session.stream_chat(
        messages,
        model=qwen_model,
        **all_params  # Pass everything through
    )

    completion_id = uuid.uuid4().hex
    created = int(time.time())

    usage_event: Optional[Dict[str, int]] = None

    # Emit the initial role delta as OpenAI does
    initial_chunk = build_stream_chunk(
        completion_id,
        created,
        display_model,
        delta={"role": "assistant"},
    )
    yield format_sse(initial_chunk)

    for event in events:
            etype = event.get("type")
            if etype == "reasoning_start" or etype == "reasoning_end":
                continue

            if etype == "reasoning":
                text = event.get("text", "")
                chunk = build_stream_chunk(
                    completion_id,
                    created,
                    display_model,
                    delta={"reasoning_content": text},
                )
                yield format_sse(chunk)
            elif etype == "text":
                text = event.get("text", "")
                chunk = build_stream_chunk(
                    completion_id,
                    created,
                    display_model,
                    delta={"content": text},
                )
                yield format_sse(chunk)
            elif etype == "tool_calls":
                tool_calls = event.get("tool_calls", [])
                chunk = build_stream_chunk(
                    completion_id,
                    created,
                    display_model,
                    delta={"tool_calls": tool_calls},
                )
                yield format_sse(chunk)
            elif etype == "usage":
                usage_event = {
                    "prompt_tokens": event.get("inputTokens", event.get("input_tokens", 0)),
                    "completion_tokens": event.get("outputTokens", event.get("output_tokens", 0)),
                }

    final_chunk = build_stream_chunk(
        completion_id,
        created,
        display_model,
        delta={},
        finish_reason="stop",
    )
    yield format_sse(final_chunk)

    if usage_event is not None:
        usage_payload = build_usage_chunk(
            completion_id,
            created,
            display_model,
            usage_event["prompt_tokens"],
            usage_event["completion_tokens"],
        )
        yield format_sse(usage_payload)

    yield "data: [DONE]\n\n"


async def stream_response(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str],
    **all_params  # Accept all parameters transparently
) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop_token = object()

    def worker() -> None:
        try:
            for chunk in _stream_response_sync(
                messages,
                model=model,
                **all_params  # Forward all parameters
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
        yield item

def build_stream_chunk(
    completion_id: str,
    created: int,
    model: str,
    *,
    delta: Dict[str, str],
    finish_reason: Optional[str] = None,
) -> Dict[str, object]:
    """Construct a streaming chunk payload."""
    choice: Dict[str, object] = {
        "index": 0,
        "delta": delta,
    }
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason

    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }


def build_usage_chunk(
    completion_id: str,
    created: int,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Dict[str, object]:
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


def format_sse(payload: Dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("qwen_server:app", host="127.0.0.1", port=54434, reload=False)
