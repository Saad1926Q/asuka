import asyncio
import json
from typing import Any

import httpx
import pytest

from asuka.rollout.sglang import (
    SGLangClient,
    SGLangError,
    SGLangResponseError,
    parse_generation_response,
)


def response_payload() -> dict[str, Any]:
    return {
        "text": " answer",
        "meta_info": {
            "output_token_logprobs": [
                [-0.2, 42, " answer"],
                [-0.4, 7, ""],
            ],
            "finish_reason": {"type": "stop"},
        },
    }


def test_parse_generation_response_extracts_tokens_and_log_probs() -> None:
    result = parse_generation_response(response_payload())

    assert result.text == " answer"
    assert result.token_ids == [42, 7]
    assert result.log_probs == [-0.2, -0.4]
    assert result.meta_info["finish_reason"] == {"type": "stop"}


def test_sglang_client_sends_generation_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_payload())

    async def run() -> None:
        async with SGLangClient(
            "http://sglang.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            result = await client.generate(
                [1, 2, 3],
                max_new_tokens=8,
                temperature=0.7,
                top_p=0.9,
                top_k=20,
                stop=["\\n"],
            )

        assert result.token_ids == [42, 7]
        assert len(requests) == 1
        assert requests[0].url.path == "/generate"
        assert json.loads(requests[0].content) == {
            "input_ids": [1, 2, 3],
            "sampling_params": {
                "max_new_tokens": 8,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 20,
                "stop": ["\\n"],
            },
            "return_logprob": True,
        }

    asyncio.run(run())


def test_parse_generation_response_rejects_missing_log_probs() -> None:
    payload = {"text": "answer", "meta_info": {}}

    with pytest.raises(SGLangResponseError, match="output_token_logprobs"):
        parse_generation_response(payload)


def test_parse_generation_response_rejects_malformed_log_prob_entry() -> None:
    payload = {
        "text": "answer",
        "meta_info": {"output_token_logprobs": [[-0.2]]},
    }

    with pytest.raises(SGLangResponseError, match="entry 0"):
        parse_generation_response(payload)


def test_sglang_client_wraps_http_errors() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    async def run() -> None:
        async with SGLangClient(
            "http://sglang.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            with pytest.raises(SGLangError, match="request failed"):
                await client.generate([1], max_new_tokens=1)

    asyncio.run(run())
