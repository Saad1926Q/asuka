"""Async HTTP integration with an external SGLang generation server."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx


class SGLangError(RuntimeError):
    """Base error for SGLang requests and response handling."""


class SGLangResponseError(SGLangError, ValueError):
    """Raised when SGLang returns a response with an invalid shape."""


@dataclass(slots=True)
class GenerationResult:
    """Generated response data returned by the SGLang adapter."""

    text: str
    token_ids: list[int]
    log_probs: list[float]
    meta_info: dict[str, Any]


def parse_generation_response(payload: Mapping[str, Any]) -> GenerationResult:
    """Parses SGLang's text and per-token log-probability response.

    Args:
        payload: JSON object returned by SGLang's ``/generate`` endpoint.

    Returns:
        Generated text, response token IDs, response log-probabilities, and
        the untouched SGLang metadata.
    """

    text = payload.get("text")

    meta_info = payload.get("meta_info")

    if not isinstance(text, str):
        raise SGLangResponseError("SGLang response is missing string field 'text'")

    if not isinstance(meta_info, Mapping):
        raise SGLangResponseError("SGLang response is missing object field 'meta_info'")

    token_logprobs = meta_info.get("output_token_logprobs")

    if not isinstance(token_logprobs, Sequence) or isinstance(token_logprobs, (str, bytes)):
        raise SGLangResponseError(
            "SGLang meta_info is missing sequence field 'output_token_logprobs'"
        )

    token_ids: list[int] = []
    log_probs: list[float] = []

    for token_index, item in enumerate(token_logprobs):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2:
            raise SGLangResponseError(
                f"SGLang log-probability entry {token_index} must contain "
                "log-probability and token ID"
            )

        try:
            log_probs.append(float(item[0]))
            token_ids.append(int(item[1]))

        except (TypeError, ValueError) as error:
            raise SGLangResponseError(
                f"SGLang log-probability entry {token_index} contains non-numeric values"
            ) from error

    return GenerationResult(
        text=text,
        token_ids=token_ids,
        log_probs=log_probs,
        meta_info=dict(meta_info),
    )


class SGLangClient:
    """Calls an externally running SGLang generation server."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 120.0,
        endpoint: str = "/generate",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Creates a client for one SGLang HTTP endpoint.

        Args:
            base_url: Server URL, such as ``http://localhost:30000``.
            timeout: Request timeout in seconds.
            endpoint: Generation route exposed by the server.
            transport: Optional HTTPX transport, primarily useful for tests.
        """

        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

        self._endpoint = endpoint

    async def generate(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generates one response and returns its tokens and log-probabilities.

        Args:
            prompt_tokens: Token IDs for the prompt sent to SGLang.
            max_new_tokens: Maximum number of response tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling limit; ``-1`` disables the limit.
            stop: Optional strings that terminate generation.
        """

        sampling_params: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        if stop is not None:
            sampling_params["stop"] = stop

        try:
            response = await self._client.post(
                self._endpoint,
                json={
                    "input_ids": prompt_tokens,
                    "sampling_params": sampling_params,
                    "return_logprob": True,
                },
            )

            response.raise_for_status()

            payload = response.json()

        except httpx.HTTPError as error:
            raise SGLangError(f"SGLang request failed: {error}") from error

        except ValueError as error:
            raise SGLangResponseError("SGLang returned invalid JSON") from error

        if not isinstance(payload, Mapping):
            raise SGLangResponseError("SGLang response must be a JSON object")

        return parse_generation_response(payload)

    async def aclose(self) -> None:
        """Closes the underlying HTTP connection pool."""

        await self._client.aclose()

    async def __aenter__(self) -> SGLangClient:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()
