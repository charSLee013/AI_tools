# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
This repository contains AI tools, primarily focused on providing an OpenAI-compatible endpoint that proxies Qwen3 Coder. The main component is a FastAPI server that handles authentication with Qwen's OAuth system and translates between OpenAI and Qwen APIs.

## Key Components
1. `qwen_server.py` - Main FastAPI server that provides OpenAI-compatible endpoints
2. Test files for verifying server functionality
3. Various translation tools using different AI services
4. Safetensors analysis and modification tools

## Common Development Tasks
### Running the Server
```bash
python -m uvicorn qwen_server:app --host 127.0.0.1 --port 54434
```

### Running Tests
```bash
# Run comprehensive server tests
python test_qwen_server.py

# Run specific fix verification
python test_fix.py

# Run end-to-end HTTP tests
python test_real_fix.py

# Run curl-based tests
./test_curl.sh
```

## Code Architecture
The main server (`qwen_server.py`) follows these key patterns:
1. OAuth credential management with automatic refresh
2. OpenAI-compatible API with transparent parameter forwarding
3. Support for both streaming and non-streaming responses
4. Special handling for Qwen's thinking mode and tool calling
5. Proper error handling and propagation from upstream APIs

## Parameter Forwarding
The server transparently forwards all OpenAI parameters to the upstream Qwen API, including:
- Core parameters: temperature, max_tokens, top_p, frequency_penalty, presence_penalty
- Tool calling parameters: tools, tool_choice
- Streaming parameters: stream, stream_options
- Special Qwen features: enable_thinking

## Testing
The test suite (`test_qwen_server.py`) provides comprehensive coverage of:
- Basic functionality (streaming/non-streaming)
- Parameter validation and forwarding
- Error handling
- Tool calling flows
- Thinking mode support
- Edge cases like null content handling