import httpx
import json
import logging
from typing import Any, List, Mapping, Optional, Dict, Iterator, AsyncIterator

from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk, Generation, LLMResult

from config.settings import (
    VLLM_ENDPOINT_URL,
    VLLM_MODEL_IDENTIFIER,
    VLLM_API_KEY,
    VLLM_REQUEST_TIMEOUT,
    # Defaults for generation parameters - can be overridden
    VLLM_DEFAULT_TEMPERATURE,
    VLLM_DEFAULT_MAX_TOKENS,
    VLLM_DEFAULT_TOP_P,
    VLLM_DEFAULT_TOP_K,
)

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Custom Exception Classes for VLLM Client ---
class VLLMError(Exception):
    """Base exception for VLLM client errors."""
    pass

class VLLMConnectionError(VLLMError):
    """Error connecting to the VLLM endpoint."""
    pass

class VLLMTimeoutError(VLLMConnectionError):
    """Request to VLLM timed out."""
    pass

class VLLMAPIError(VLLMError):
    """Error returned by the VLLM API (e.g., bad request, auth error)."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text

class VLLMResponseError(VLLMError):
    """Error parsing the response from VLLM."""
    pass

# --- VLLM Client Implementation ---
class VLLMClient(LLM):
    """
    Langchain LLM class interacting with a vLLM server's OpenAI-compatible API.
    """
    endpoint_url: str = VLLM_ENDPOINT_URL
    model_identifier: str = VLLM_MODEL_IDENTIFIER
    api_key: Optional[str] = VLLM_API_KEY
    timeout: int = VLLM_REQUEST_TIMEOUT

    # Default generation parameters (can be overridden at runtime)
    temperature: float = VLLM_DEFAULT_TEMPERATURE
    max_tokens: int = VLLM_DEFAULT_MAX_TOKENS
    top_p: float = VLLM_DEFAULT_TOP_P
    top_k: int = VLLM_DEFAULT_TOP_K # -1 means disable top_k
    stop: Optional[List[str]] = None

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "vllm_client_openai_compatible"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters for Langchain callbacks and caching."""
        return {
            "endpoint_url": self.endpoint_url,
            "model_identifier": self.model_identifier,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": self.stop,
            # Include other configured parameters here
        }

    def _prepare_request_payload(self, prompt: str, stop: Optional[List[str]], stream: bool, **kwargs: Any) -> Dict[str, Any]:
        """Helper to construct the JSON payload for the vLLM API call."""
        merged_stop = stop if stop is not None else self.stop

        payload = {
            "model": self.model_identifier,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "stop": merged_stop,
            "stream": stream,
        }
        # Filter out None values, as some APIs might not handle them correctly
        return {k: v for k, v in payload.items() if v is not None}

    def _prepare_headers(self) -> Dict[str, str]:
        """Helper to construct request headers, including Authorization."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key and self.api_key.lower() != "empty":
            logger.debug("Adding Authorization header for VLLM request.")
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _process_response(self, response_data: Dict[str, Any]) -> str:
        """Helper to extract text from a non-streaming vLLM response."""
        if not isinstance(response_data, dict):
             raise VLLMResponseError(f"Expected JSON object (dict) but received {type(response_data)}")

        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            logger.warning(f"No 'choices' found in vLLM response or it's not a list: {response_data}")
            raise VLLMResponseError(f"Invalid response format: 'choices' missing or empty. Response: {response_data}")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
             raise VLLMResponseError(f"Expected dict in 'choices' list, got {type(first_choice)}")

        # Handle standard completion format
        text = first_choice.get("text")
        if isinstance(text, str):
            return text.strip()

        # Handle OpenAI chat completion format (delta for streaming is handled separately)
        message = first_choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()

        logger.warning(f"Could not extract text/content from vLLM choice: {first_choice}")
        raise VLLMResponseError(f"Could not extract text from choice: {first_choice}")

    # --- Synchronous Methods --- #

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous, non-streaming call to the vLLM endpoint."""
        payload = self._prepare_request_payload(prompt, stop, stream=False, **kwargs)
        headers = self._prepare_headers()

        logger.info(f"Sending request to vLLM endpoint: {self.endpoint_url}")
        logger.debug(f"Request payload (non-streaming): {payload}")

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx
                response_data = response.json()
                logger.debug(f"Received successful non-streaming response: {response_data}")
                return self._process_response(response_data)

        except httpx.TimeoutException as e:
            logger.error(f"Request to vLLM timed out after {self.timeout} seconds.", exc_info=True)
            raise VLLMTimeoutError(f"Request timed out after {self.timeout}s") from e
        except httpx.RequestError as e:
            # Includes connection errors, DNS errors, etc.
            logger.error(f"Failed to connect to vLLM endpoint '{self.endpoint_url}': {e}", exc_info=True)
            raise VLLMConnectionError(f"Connection failed to endpoint {self.endpoint_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            response_text = e.response.text
            logger.error(f"vLLM request failed with status {status_code}. Response: {response_text[:500]}...", exc_info=True)
            raise VLLMAPIError(
                f"HTTP {status_code} error calling VLLM: {response_text[:200]}...",
                status_code=status_code,
                response_text=response_text
            ) from e
        except json.JSONDecodeError as e:
            logger.error(f"Could not decode JSON response from vLLM. Response: {response.text[:500]}...", exc_info=True)
            raise VLLMResponseError(f"Invalid JSON response received: {e}. Response text: {response.text[:200]}...") from e
        except VLLMResponseError as e:
             # Re-raise errors from _process_response
             logger.error(f"Error processing VLLM response: {e}", exc_info=True)
             raise
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"An unexpected error occurred during vLLM call: {e}", exc_info=True)
            raise VLLMError(f"Unexpected error: {e}") from e

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Synchronous, streaming call to the vLLM endpoint."""
        payload = self._prepare_request_payload(prompt, stop, stream=True, **kwargs)
        headers = self._prepare_headers()

        logger.info(f"Sending streaming request to vLLM endpoint: {self.endpoint_url}")
        logger.debug(f"Request payload (streaming): {payload}")

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", self.endpoint_url, headers=headers, json=payload) as response:
                    response.raise_for_status() # Check status before starting iteration
                    logger.debug("Established stream connection successfully.")
                    for line in response.iter_lines():
                        if line and line.startswith("data:"):
                            data_str = line[len("data:"):].strip()
                            if data_str == "[DONE]":
                                logger.debug("Received [DONE] marker, ending stream.")
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                                text_chunk = delta.get("content", "")
                                if text_chunk: # Only yield if there's content
                                    chunk = GenerationChunk(text=text_chunk)
                                    yield chunk
                                    if run_manager:
                                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                            except json.JSONDecodeError:
                                logger.warning(f"Could not decode JSON chunk: {data_str}")
                            except Exception as e:
                                 logger.warning(f"Error processing stream chunk {data_str}: {e}", exc_info=False)
                    logger.info("Finished receiving stream from vLLM.")

        except httpx.TimeoutException as e:
            logger.error(f"Request to vLLM timed out during streaming after {self.timeout} seconds.", exc_info=True)
            raise VLLMTimeoutError(f"Stream request timed out after {self.timeout}s") from e
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to vLLM endpoint for streaming '{self.endpoint_url}': {e}", exc_info=True)
            raise VLLMConnectionError(f"Stream connection failed to endpoint {self.endpoint_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            # Try to read body for error details, might be difficult with stream context
            response_text = ""
            try:
                pass # Avoid reading potentially large body here
            except Exception:
                pass
            logger.error(f"vLLM stream request failed with status {status_code}.", exc_info=True)
            raise VLLMAPIError(
                f"HTTP {status_code} error starting VLLM stream.",
                status_code=status_code,
                # response_text=response_text
            ) from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during vLLM stream: {e}", exc_info=True)
            raise VLLMError(f"Unexpected stream error: {e}") from e

    # --- Asynchronous Methods --- #

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronous, non-streaming call to the vLLM endpoint."""
        payload = self._prepare_request_payload(prompt, stop, stream=False, **kwargs)
        headers = self._prepare_headers()

        logger.info(f"Sending async request to vLLM endpoint: {self.endpoint_url}")
        logger.debug(f"Async request payload (non-streaming): {payload}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                response_data = response.json()
                logger.debug(f"Received successful async non-streaming response: {response_data}")
                result = self._process_response(response_data)
                if run_manager: # Handle potential callback
                    await run_manager.on_llm_end(LLMResult(generations=[[Generation(text=result)]]))
                return result

        except httpx.TimeoutException as e:
            logger.error(f"Async request to vLLM timed out after {self.timeout} seconds.", exc_info=True)
            raise VLLMTimeoutError(f"Async request timed out after {self.timeout}s") from e
        except httpx.RequestError as e:
            logger.error(f"Failed to connect async to vLLM endpoint '{self.endpoint_url}': {e}", exc_info=True)
            raise VLLMConnectionError(f"Async connection failed to endpoint {self.endpoint_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            response_text = e.response.text
            logger.error(f"Async vLLM request failed with status {status_code}. Response: {response_text[:500]}...", exc_info=True)
            raise VLLMAPIError(
                f"Async HTTP {status_code} error calling VLLM: {response_text[:200]}...",
                status_code=status_code,
                response_text=response_text
            ) from e
        except json.JSONDecodeError as e:
            logger.error(f"Could not decode async JSON response from vLLM. Response: {response.text[:500]}...", exc_info=True)
            raise VLLMResponseError(f"Invalid async JSON response received: {e}. Response text: {response.text[:200]}...") from e
        except VLLMResponseError as e:
             logger.error(f"Error processing async VLLM response: {e}", exc_info=True)
             raise
        except Exception as e:
            logger.error(f"An unexpected async error occurred during vLLM call: {e}", exc_info=True)
            raise VLLMError(f"Unexpected async error: {e}") from e

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """Asynchronous, streaming call to the vLLM endpoint."""
        payload = self._prepare_request_payload(prompt, stop, stream=True, **kwargs)
        headers = self._prepare_headers()

        logger.info(f"Sending async streaming request to vLLM endpoint: {self.endpoint_url}")
        logger.debug(f"Async request payload (streaming): {payload}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.endpoint_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    logger.debug("Established async stream connection successfully.")
                    async for line in response.aiter_lines():
                        if line and line.startswith("data:"):
                            data_str = line[len("data:"):].strip()
                            if data_str == "[DONE]":
                                logger.debug("Received async [DONE] marker, ending stream.")
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                                text_chunk = delta.get("content", "")
                                if text_chunk:
                                    chunk = GenerationChunk(text=text_chunk)
                                    yield chunk
                                    if run_manager:
                                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                            except json.JSONDecodeError:
                                logger.warning(f"Could not decode async JSON chunk: {data_str}")
                            except Exception as e:
                                 logger.warning(f"Error processing async stream chunk {data_str}: {e}", exc_info=False)
                    logger.info("Finished receiving async stream from vLLM.")

        except httpx.TimeoutException as e:
            logger.error(f"Async request to vLLM timed out during streaming after {self.timeout} seconds.", exc_info=True)
            raise VLLMTimeoutError(f"Async stream request timed out after {self.timeout}s") from e
        except httpx.RequestError as e:
            logger.error(f"Failed to connect async to vLLM endpoint for streaming '{self.endpoint_url}': {e}", exc_info=True)
            raise VLLMConnectionError(f"Async stream connection failed to endpoint {self.endpoint_url}: {e}") from e
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            # response_text = await e.response.aread().decode() # Attempt to read async body
            logger.error(f"Async vLLM stream request failed with status {status_code}.", exc_info=True)
            raise VLLMAPIError(
                f"Async HTTP {status_code} error starting VLLM stream.",
                status_code=status_code,
                # response_text=response_text
            ) from e
        except Exception as e:
            logger.error(f"An unexpected async error occurred during vLLM stream: {e}", exc_info=True)
            raise VLLMError(f"Unexpected async stream error: {e}") from e

# Example Usage / Test Block
if __name__ == "__main__":
    import asyncio
    # Configure logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Running VLLMClient Self-Test ---")
    # Ensure your vLLM server (or a compatible mock) is running at VLLM_ENDPOINT_URL

    # Use default settings from config, but can override here
    llm_client = VLLMClient(
        temperature=0.01,
        max_tokens=4096,
        top_p=0.01,
        api_key=None # Set API key here or ensure it's in settings/env if needed
    )
    test_prompt = "Explain the concept of Retrieval-Augmented Generation (RAG) in about 50 words."

    # --- Test Synchronous Invoke --- #
    try:
        logger.info(f"\n--- Testing Synchronous Invoke --- ")
        logger.info(f"Prompt: '{test_prompt}'")
        sync_response = llm_client.invoke(test_prompt, stop=["\n\n"])
        logger.info(f"Sync Response:\n{sync_response}")
        assert isinstance(sync_response, str) and len(sync_response) > 10
    except VLLMError as e:
        logger.error(f"Sync invoke failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during sync invoke test: {e}", exc_info=True)

    # --- Test Asynchronous Invoke --- #
    async def run_async_invoke():
        try:
            logger.info(f"\n--- Testing Asynchronous Invoke --- ")
            logger.info(f"Prompt: '{test_prompt}'")
            async_response = await llm_client.ainvoke(test_prompt, temperature=0.2)
            logger.info(f"Async Response:\n{async_response}")
            assert isinstance(async_response, str) and len(async_response) > 10
        except VLLMError as e:
            logger.error(f"Async invoke failed: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during async invoke test: {e}", exc_info=True)
    asyncio.run(run_async_invoke())

    # --- Test Synchronous Stream --- #
    try:
        logger.info(f"\n--- Testing Synchronous Stream --- ")
        logger.info(f"Prompt: '{test_prompt}'")
        streamed_response = ""
        logger.info("Streamed Response Chunks:")
        for chunk in llm_client.stream(test_prompt, max_tokens=100):
            logger.info(f"  Chunk: {chunk.text}")
            streamed_response += chunk.text
        logger.info(f"Full Streamed Response:\n{streamed_response}")
        assert isinstance(streamed_response, str) and len(streamed_response) > 10
    except VLLMError as e:
        logger.error(f"Sync stream failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during sync stream test: {e}", exc_info=True)

    # --- Test Asynchronous Stream --- #
    async def run_async_stream():
        try:
            logger.info(f"\n--- Testing Asynchronous Stream --- ")
            logger.info(f"Prompt: '{test_prompt}'")
            async_streamed_response = ""
            logger.info("Async Streamed Response Chunks:")
            async for chunk in llm_client.astream(test_prompt, max_tokens=100):
                logger.info(f"  Async Chunk: {chunk.text}")
                async_streamed_response += chunk.text
            logger.info(f"Full Async Streamed Response:\n{async_streamed_response}")
            assert isinstance(async_streamed_response, str) and len(async_streamed_response) > 10
        except VLLMError as e:
            logger.error(f"Async stream failed: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error during async stream test: {e}", exc_info=True)
    asyncio.run(run_async_stream())

    logger.info("\n--- VLLMClient Self-Test Completed ---")