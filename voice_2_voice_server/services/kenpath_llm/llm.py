from loguru import logger
from pipecat.services.openai.llm import (
    OpenAILLMService,
    OpenAIAssistantContextAggregator,
    OpenAIContextAggregatorPair,
)
from pipecat.frames.frames import LLMTextFrame, TTSSpeakFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from services.bhashini.stt import BhashiniKenpathUserContextAggregator
import aiohttp
import asyncio
import codecs
import json
import jwt
import time
from typing import AsyncIterator, Iterator, List, Optional
from pathlib import Path
import uuid
import os


DEFAULT_VISTAAR_PROD_URL = "https://voice-prod.mahapocra.gov.in"
DEFAULT_VISTAAR_DEV_URL = "https://vistaar-dev.mahapocra.gov.in"
DEFAULT_BHARAT_VISTAAR_API_URL = "https://chat-vistaar.da.gov.in"
DEFAULT_BHARAT_VISTAAR_KEY_PATH = "services/kenpath_llm/prod_private_key_bh.pem"

ENGLISH_LANGUAGE_LABELS = frozenset(
    {"English", "English (India)", "English (United States)"}
)


def parse_bharat_vistaar_voice_delta(content: str) -> tuple[str, bool]:
    """Extract spoken text from Bharat Vistaar SSE delta.content.

    Production returns JSON objects like
    ``{"audio": "...", "end_interaction": false, "language": "en"}``.
    The API doc shows plain text deltas; accept both shapes.
    """
    text = (content or "").strip()
    if not text:
        return "", False
    if not text.startswith("{"):
        return text, False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text, False
    if not isinstance(payload, dict):
        return text, False
    audio = str(payload.get("audio") or "").strip()
    end_interaction = bool(payload.get("end_interaction"))
    return audio, end_interaction


def normalize_vistaar_environment(environment: Optional[str]) -> str:
    """Normalize vistaar environment to 'prod' or 'dev'."""
    if environment and str(environment).strip().lower() == "dev":
        return "dev"
    return "prod"


def normalize_kenpath_backend(backend: Optional[str]) -> str:
    if backend and str(backend).strip().lower() == "bharatvistaar":
        return "bharatvistaar"
    return "vistaar"


def resolve_vistaar_base_url(environment: Optional[str]) -> str:
    """Resolve Vistaar base URL from environment and deployment env vars."""
    env = normalize_vistaar_environment(environment)
    if env == "dev":
        return os.environ.get("KENPATH_VISTAAR_API_URL_DEV", DEFAULT_VISTAAR_DEV_URL)
    return (
        os.environ.get("KENPATH_VISTAAR_API_URL_PROD")
        or os.environ.get("KENPATH_VISTAAR_API_URL")
        or DEFAULT_VISTAAR_PROD_URL
    )


def resolve_voice_bhili_url(environment: Optional[str]) -> str:
    """Resolve Voice Bhili URL from vistaar environment (same prod/dev as streaming API)."""
    env = normalize_vistaar_environment(environment)
    if env == "dev":
        override = os.environ.get("KENPATH_VOICE_BHILI_URL_DEV")
    else:
        override = os.environ.get("KENPATH_VOICE_BHILI_URL_PROD")
    if override:
        return override
    legacy = os.environ.get("KENPATH_VOICE_BHILI_URL")
    if legacy:
        return legacy
    return f"{resolve_vistaar_base_url(environment)}/api/voice-bhili"


def resolve_bharat_vistaar_language(language: Optional[str]) -> str:
    """Map agent language to Bharat Vistaar X-Language header (en or hi)."""
    lang_lower = (language or "").strip().lower()
    if lang_lower in ("english", "en") or language in ENGLISH_LANGUAGE_LABELS:
        return "en"
    if lang_lower == "hindi":
        return "hi"
    return "hi"


class KenpathLLM(OpenAILLMService):
    def __init__(
        self,
        vistaar_session_id: Optional[str] = None,
        language: Optional[str] = None,
        vistaar_environment: Optional[str] = None,
        kenpath_backend: Optional[str] = None,
        hold_messages: Optional[list[str]] = None,
        response_timeout: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.response_timeout = max(0.05, float(response_timeout))
        self._vistaar_session_id = vistaar_session_id
        self._bhashini_fast_turn = False
        self._kenpath_backend = normalize_kenpath_backend(kenpath_backend)
        self._call_user_id = vistaar_session_id or str(uuid.uuid4())

        self._private_key: Optional[str] = None
        self._bharat_private_key: Optional[str] = None
        self._jwt_phone = os.environ.get("KENPATH_JWT_PHONE", "+91-9036722772")
        self._vistaar_environment = normalize_vistaar_environment(vistaar_environment)
        self._base_url = resolve_vistaar_base_url(self._vistaar_environment)
        self._bharat_api_url = os.environ.get(
            "BHARAT_VISTAAR_API_URL", DEFAULT_BHARAT_VISTAAR_API_URL
        ).rstrip("/")
        self._bharat_language = resolve_bharat_vistaar_language(language)

        if self._kenpath_backend == "vistaar":
            self._private_key = Path(os.environ["KENPATH_JWT_PRIVATE_KEY_PATH"]).read_text()

        self._session: Optional[aiohttp.ClientSession] = None
        self._voice_bhili_url = resolve_voice_bhili_url(self._vistaar_environment)

        lang_lower = (language or "").strip().lower()
        if self._kenpath_backend == "bharatvistaar":
            self._use_voice_bhili = False
            self._source_lang = self._bharat_language
            self._target_lang = self._bharat_language
        elif lang_lower == "bhb":
            self._use_voice_bhili = True
            self._source_lang = "bhb"
            self._target_lang = "bhb"
        else:
            self._use_voice_bhili = False
            if lang_lower == "hindi":
                self._source_lang = "hi"
                self._target_lang = "hi"
            else:
                self._source_lang = "mr"
                self._target_lang = "mr"

        self.hold_messages = [
            str(msg).strip() for msg in (hold_messages or []) if str(msg).strip()
        ]
        self.hold_message_index = 0

        if self._kenpath_backend == "bharatvistaar":
            logger.info(
                "KenpathLLM initialized | Bharat Vistaar | timeout={}s | hold_messages={} | "
                "url={} | X-Language={}",
                self.response_timeout,
                len(self.hold_messages),
                self._bharat_api_url,
                self._bharat_language,
            )
        elif self._use_voice_bhili:
            logger.info(
                "KenpathLLM initialized | Voice Bhili | env={} | timeout={}s | "
                "hold_messages={} | url={}",
                self._vistaar_environment,
                self.response_timeout,
                len(self.hold_messages),
                self._voice_bhili_url,
            )
        else:
            logger.info(
                "KenpathLLM initialized | Vistaar | env={} | timeout={}s | "
                "hold_messages={} | url={} | lang={}",
                self._vistaar_environment,
                self.response_timeout,
                len(self.hold_messages),
                self._base_url,
                self._source_lang,
            )
        if self._vistaar_session_id:
            logger.info("Vistaar session ID for this call: {}", self._vistaar_session_id)

    def enable_bhashini_fast_turn(self) -> None:
        """Use Bhashini final transcript as the sole LLM turn trigger (Bhashini+Kenpath only)."""
        self._bhashini_fast_turn = True
        self._user_aggregator_params = LLMUserAggregatorParams(
            aggregation_timeout=0.05
        )
        logger.info(
            "Kenpath: enabled fast turn — LLM starts on Bhashini final transcript"
        )

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> OpenAIContextAggregatorPair:
        if self._bhashini_fast_turn:
            context.set_llm_adapter(self.get_llm_adapter())
            user = BhashiniKenpathUserContextAggregator(context, params=user_params)
            assistant = OpenAIAssistantContextAggregator(
                context, params=assistant_params
            )
            return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)
        return super().create_context_aggregator(
            context,
            user_params=user_params,
            assistant_params=assistant_params,
        )

    def _load_bharat_private_key(self) -> str:
        if self._bharat_private_key is None:
            key_path = os.environ.get(
                "BHARAT_VISTAAR_JWT_PRIVATE_KEY_PATH", DEFAULT_BHARAT_VISTAAR_KEY_PATH
            ).strip()
            if not key_path:
                raise ValueError(
                    "Bharat Vistaar requires BHARAT_VISTAAR_JWT_PRIVATE_KEY_PATH in .env."
                )
            self._bharat_private_key = Path(key_path).read_text()
        return self._bharat_private_key

    def _generate_jwt(self) -> str:
        """Generate JWT for legacy Vistaar /api/voice/ (iss: voice-provider)."""
        now = int(time.time())
        payload = {
            "sub": self._jwt_phone,
            "iss": "voice-provider",
            "iat": now,
            "exp": now + 3600,
        }
        return jwt.encode(payload, self._private_key, algorithm="RS256")

    def _bharat_call_id(self) -> str:
        """Call-scoped ID for Bharat Vistaar session, user, and tenant fields."""
        return self._vistaar_session_id or self._call_user_id

    def _generate_bharat_vistaar_jwt(self) -> str:
        """Generate JWT for Bharat Vistaar chat completions (iss: samvaad)."""
        now = int(time.time())
        call_id = self._bharat_call_id()
        payload = {
            "user_id": call_id,
            "tenant_id": call_id,
            "iss": "samvaad",
            "iat": now,
            "exp": now + 3600,
        }
        return jwt.encode(
            payload, self._load_bharat_private_key(), algorithm="RS256"
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120)
            )
        return self._session

    def _get_hold_message(self) -> str:
        msg = self.hold_messages[self.hold_message_index]
        self.hold_message_index = (self.hold_message_index + 1) % len(self.hold_messages)
        logger.debug("Hold message: '{}'", msg)
        return msg

    def _build_chat_messages(
        self, context: OpenAILLMContext | LLMContext
    ) -> List[dict]:
        messages: List[dict] = []
        for msg in context.get_messages():
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("system", "user", "assistant") and str(content).strip():
                messages.append({"role": role, "content": str(content)})
        return messages

    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        messages = self._build_chat_messages(context)
        user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        if not user_message:
            logger.warning("No user message found")
            return

        logger.info("Processing: '{}...'", user_message[:50])

        first_chunk_arrived = asyncio.Event()
        start_time = time.perf_counter()

        async def hold_message_timer():
            try:
                await asyncio.wait_for(
                    first_chunk_arrived.wait(),
                    timeout=self.response_timeout,
                )
                logger.debug("LLM responded before timeout")
            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - start_time
                hold_msg = self._get_hold_message()
                logger.info("Timeout after {:.2f}s - playing: '{}'", elapsed, hold_msg)
                await self.push_frame(TTSSpeakFrame(hold_msg))

        timer_task = None
        if self.hold_messages:
            timer_task = asyncio.create_task(hold_message_timer())

        await self.start_ttfb_metrics()

        try:
            first_chunk = True
            chunk_count = 0

            if self._kenpath_backend == "bharatvistaar":
                stream = self._stream_bharat_vistaar_chat(messages)
            elif self._use_voice_bhili:
                stream = self._iter_voice_bhili_text(user_message)
            else:
                stream = self._stream_vistaar_completions(user_message)

            async for chunk in stream:
                if first_chunk:
                    first_chunk = False
                    elapsed = time.perf_counter() - start_time
                    logger.info("First chunk received at {:.2f}s", elapsed)
                    first_chunk_arrived.set()
                    await self.stop_ttfb_metrics()

                await self.push_frame(LLMTextFrame(text=chunk))
                chunk_count += 1

            logger.info("Completed - {} chunks streamed", chunk_count)

        except Exception as e:
            logger.error("Error: {}", e)
            first_chunk_arrived.set()
            raise

        finally:
            if timer_task is not None and not timer_task.done():
                timer_task.cancel()
                try:
                    await timer_task
                except asyncio.CancelledError:
                    pass

    def _yield_word_chunks_from_text(self, text: str) -> Iterator[str]:
        buffer = text
        while " " in buffer or "\n" in buffer:
            space_idx = buffer.find(" ")
            newline_idx = buffer.find("\n")

            if space_idx == -1 and newline_idx == -1:
                break
            elif space_idx == -1:
                split_idx = newline_idx
            elif newline_idx == -1:
                split_idx = space_idx
            else:
                split_idx = min(space_idx, newline_idx)

            word = buffer[:split_idx].strip()
            buffer = buffer[split_idx + 1 :]

            if word:
                yield word + " "

        if buffer.strip():
            yield buffer.strip()

    async def _yield_words_from_delta_buffer(
        self, delta: str, pending: str
    ) -> AsyncIterator[str]:
        combined = pending + delta
        if not combined:
            return
        if " " not in combined and "\n" not in combined:
            yield "", combined
            return

        remainder = combined
        for chunk in self._yield_word_chunks_from_text(combined):
            yield chunk, ""
            remainder = ""
        if remainder and (" " not in combined and "\n" not in combined):
            yield "", remainder

    def _bharat_vistaar_audio_suffix(
        self, audio_text: str, last_audio: str
    ) -> tuple[str, str]:
        """Return only the new spoken suffix from cumulative Bharat Vistaar audio."""
        if not audio_text:
            return "", last_audio
        if not last_audio:
            return audio_text, audio_text
        if audio_text.startswith(last_audio):
            return audio_text[len(last_audio) :], audio_text
        if audio_text == last_audio:
            return "", last_audio
        logger.warning(
            "Bharat Vistaar audio reset mid-stream | previous_len={} new_len={}",
            len(last_audio),
            len(audio_text),
        )
        return audio_text, audio_text

    async def _stream_bharat_vistaar_chat(
        self, messages: List[dict]
    ) -> AsyncIterator[str]:
        """POST Bharat Vistaar chat completions with SSE streaming."""
        call_id = self._bharat_call_id()
        url = f"{self._bharat_api_url}/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._generate_bharat_vistaar_jwt()}",
            "X-Tenant-ID": call_id,
            "X-User-ID": call_id,
            "X-Session-ID": call_id,
            "X-Language": self._bharat_language,
            "Accept": "text/event-stream",
        }
        body = {
            "model": "bharatvistaar-voice",
            "messages": messages,
            "stream": True,
        }

        logger.info(
            "Bharat Vistaar API | call_id={} | X-Language={} | messages={}",
            call_id,
            self._bharat_language,
            len(messages),
        )

        session = await self._get_session()
        async with session.post(url, json=body, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(
                    "Bharat Vistaar API error {}: {}",
                    response.status,
                    error_text,
                )
                raise Exception(f"Bharat Vistaar API Error {response.status}")

            line_buffer = ""
            buffer = ""
            last_audio = ""
            final_audio = ""

            async for chunk in response.content.iter_any():
                line_buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        if buffer.strip():
                            yield buffer.strip()
                        if final_audio:
                            logger.info(
                                "Bharat Vistaar final spoken text: {}",
                                final_audio[:200],
                            )
                        return

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("Bharat Vistaar skipped non-JSON SSE line: {}", data[:80])
                        continue

                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    content = ((choices[0].get("delta") or {}).get("content"))
                    if not content:
                        continue

                    audio_text, end_interaction = parse_bharat_vistaar_voice_delta(
                        content
                    )
                    if end_interaction:
                        logger.info("Bharat Vistaar end_interaction=true")
                    if not audio_text:
                        continue

                    final_audio = audio_text
                    new_spoken, last_audio = self._bharat_vistaar_audio_suffix(
                        audio_text, last_audio
                    )
                    if not new_spoken:
                        continue

                    buffer += new_spoken
                    while " " in buffer or "\n" in buffer:
                        space_idx = buffer.find(" ")
                        newline_idx = buffer.find("\n")
                        if space_idx == -1 and newline_idx == -1:
                            break
                        elif space_idx == -1:
                            split_idx = newline_idx
                        elif newline_idx == -1:
                            split_idx = space_idx
                        else:
                            split_idx = min(space_idx, newline_idx)

                        word = buffer[:split_idx].strip()
                        buffer = buffer[split_idx + 1 :]
                        if word:
                            yield word + " "

            if buffer.strip():
                yield buffer.strip()
            if final_audio:
                logger.info(
                    "Bharat Vistaar final spoken text: {}",
                    final_audio[:200],
                )

    async def _iter_voice_bhili_text(self, query: str):
        session_id = self._vistaar_session_id or str(uuid.uuid4())
        params = {
            "query": query,
            "session_id": session_id,
            "source_lang": self._source_lang,
            "target_lang": self._target_lang,
        }
        headers = {"Accept": "application/json"}
        if self._vistaar_environment == "prod":
            headers["Authorization"] = f"Bearer {self._generate_jwt()}"

        logger.info(
            "Voice Bhili API | env={} | session_id={} | query={}...",
            self._vistaar_environment,
            session_id,
            query[:50],
        )

        session = await self._get_session()
        async with session.get(
            self._voice_bhili_url, params=params, headers=headers
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error("Voice Bhili API error {}: {}", response.status, error_text)
                raise Exception(f"Voice Bhili API Error {response.status}")

            data = await response.json()
            text = ""
            if isinstance(data, dict):
                text = data.get("response") or ""
            if not (text or "").strip():
                logger.warning("Voice Bhili returned empty response")
                return

            for chunk in self._yield_word_chunks_from_text(text):
                yield chunk

    async def _stream_vistaar_completions(
        self,
        query: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        url = f"{self._base_url}/api/voice/"
        session_id = session_id or self._vistaar_session_id or str(uuid.uuid4())
        source_lang = source_lang if source_lang is not None else self._source_lang
        target_lang = target_lang if target_lang is not None else self._target_lang

        params = {
            "query": query,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "session_id": session_id,
        }

        headers = {
            "Authorization": f"Bearer {self._generate_jwt()}",
        }

        logger.info(
            "Vistaar API request | session_id={} | query={}...",
            session_id,
            query[:50],
        )

        session = await self._get_session()

        async with session.get(url, params=params, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error("API error {}: {}", response.status, error_text)
                raise Exception(f"Vistaar API Error {response.status}")

            buffer = ""
            decoder = codecs.getincrementaldecoder("utf-8")("replace")

            async for data in response.content.iter_any():
                buffer += decoder.decode(data, final=False)

                while " " in buffer or "\n" in buffer:
                    space_idx = buffer.find(" ")
                    newline_idx = buffer.find("\n")

                    if space_idx == -1 and newline_idx == -1:
                        break
                    elif space_idx == -1:
                        split_idx = newline_idx
                    elif newline_idx == -1:
                        split_idx = space_idx
                    else:
                        split_idx = min(space_idx, newline_idx)

                    word = buffer[:split_idx].strip()
                    buffer = buffer[split_idx + 1 :]

                    if word:
                        yield word + " "

            buffer += decoder.decode(b"", final=True)
            if buffer.strip():
                yield buffer.strip()

    async def cleanup(self):
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("aiohttp session closed")
