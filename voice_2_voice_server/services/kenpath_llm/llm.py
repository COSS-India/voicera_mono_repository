from loguru import logger
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.frames.frames import LLMTextFrame, TTSSpeakFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_context import LLMContext
import aiohttp
import asyncio
import codecs
import jwt
import time
from typing import Iterator, Optional
from pathlib import Path
import uuid
import os


# Hold messages and lang codes per language
KENPATH_HINDI_HOLD_MESSAGES = [
    "कृपया रुकिए, मैं जानकारी खोज रही हूँ",
    "एक क्षण रुकिए, मैं जांच कर रही हूँ",
    "कृपया प्रतीक्षा करें, मैं उत्तर खोज रही हूँ",
    "थोड़ा समय दें, मैं जानकारी प्राप्त कर रही हूँ",
]
KENPATH_MARATHI_HOLD_MESSAGES = [
    "कृपया थांबा, मी माहिती शोधत आहे",
    "एक क्षण थांबा, मी तपासत आहे",
    "कृपया प्रतीक्षा करा, मी उत्तर शोधत आहे",
    "थोडा वेळ द्या, मी माहिती मिळवत आहे",
]
KENPATH_BHILI_HOLD_MESSAGES = [
    "जाराक ऊबिरा, आय माहिती होदी रियोहं",
    "एकूच घेडी ऊबिरा, आय तपासी रियोहं",
    "जाराक वाट वेरा, आय उत्तर होदी दिहुव",
    "बेन घेडी, आय माहिती मिलवुहू",
]

DEFAULT_VISTAAR_PROD_URL = "https://voice-prod.mahapocra.gov.in"
DEFAULT_VISTAAR_DEV_URL = "https://vistaar-dev.mahapocra.gov.in"


def normalize_vistaar_environment(environment: Optional[str]) -> str:
    """Normalize vistaar environment to 'prod' or 'dev'."""
    if environment and str(environment).strip().lower() == "dev":
        return "dev"
    return "prod"


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


class KenpathLLM(OpenAILLMService):
    def __init__(
        self,
        vistaar_session_id: Optional[str] = None,
        language: Optional[str] = None,
        vistaar_environment: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.response_timeout = 0.3  # seconds
        self._vistaar_session_id = vistaar_session_id

        # JWT config
        self._private_key = Path(os.environ["KENPATH_JWT_PRIVATE_KEY_PATH"]).read_text()
        self._jwt_phone = os.environ.get("KENPATH_JWT_PHONE", "+91-9036722772")
        self._vistaar_environment = normalize_vistaar_environment(vistaar_environment)
        self._base_url = resolve_vistaar_base_url(self._vistaar_environment)

        # Shared aiohttp session (created lazily)
        self._session: Optional[aiohttp.ClientSession] = None

        # Voice Bhili (dev): GET JSON, no JWT — see KENPATH_VOICE_BHILI_URL
        self._voice_bhili_url = os.environ.get(
            "KENPATH_VOICE_BHILI_URL",
            "https://vistaar-dev.mahapocra.gov.in/api/voice-bhili",
        )

        # Language: bhb -> Voice Bhili API; Hindi -> hi; else Marathi mr
        lang_lower = (language or "").strip().lower()
        if lang_lower == "bhb":
            self._use_voice_bhili = True
            self.hold_messages = list(KENPATH_BHILI_HOLD_MESSAGES)
            self._source_lang = "bhb"
            self._target_lang = "bhb"
        else:
            self._use_voice_bhili = False
            if lang_lower == "hindi":
                self.hold_messages = list(KENPATH_HINDI_HOLD_MESSAGES)
                self._source_lang = "hi"
                self._target_lang = "hi"
            else:
                self.hold_messages = list(KENPATH_MARATHI_HOLD_MESSAGES)
                self._source_lang = "mr"
                self._target_lang = "mr"

        self.hold_message_index = 0

        if self._use_voice_bhili:
            logger.info(
                f"🤖 KenpathLLM initialized | Voice Bhili | timeout={self.response_timeout}s | url={self._voice_bhili_url}"
            )
        else:
            logger.info(
                f"🤖 KenpathLLM initialized | env={self._vistaar_environment} | timeout={self.response_timeout}s | url={self._base_url} | lang={self._source_lang}"
            )
        if self._vistaar_session_id:
            logger.info(f"📞 Vistaar session ID for this call: {self._vistaar_session_id}")

    def _generate_jwt(self) -> str:
        """Generate a fresh JWT token (local operation, ~microseconds)."""
        now = int(time.time())
        payload = {
            "sub": self._jwt_phone,
            "iss": "voice-provider",
            "iat": now,
            "exp": now + 3600,
        }
        return jwt.encode(payload, self._private_key, algorithm="RS256")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a shared aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self._session

    def _get_hold_message(self) -> str:
        """Get current hold message and rotate to next."""
        msg = self.hold_messages[self.hold_message_index]
        self.hold_message_index = (self.hold_message_index + 1) % len(self.hold_messages)
        logger.debug(f"🔄 Hold message: '{msg}'")
        return msg

    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        """Main processing with hold message on timeout."""

        # Extract user message
        messages = context.get_messages()
        user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        if not user_message:
            logger.warning("⚠️ No user message found")
            return

        logger.info(f"💬 Processing: '{user_message[:50]}...'")

        # Simple flag to track if first chunk arrived
        first_chunk_arrived = asyncio.Event()
        start_time = time.perf_counter()

        async def hold_message_timer():
            """Wait for timeout, then play hold message if no response yet."""
            try:
                await asyncio.wait_for(
                    first_chunk_arrived.wait(),
                    timeout=self.response_timeout,
                )
                logger.debug("✅ LLM responded before timeout")

            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - start_time
                hold_msg = self._get_hold_message()
                logger.info(f"⏳ Timeout after {elapsed:.2f}s - playing: '{hold_msg}'")
                await self.push_frame(TTSSpeakFrame(hold_msg))

        # Start the timer task
        timer_task = asyncio.create_task(hold_message_timer())

        await self.start_ttfb_metrics()

        try:
            first_chunk = True
            chunk_count = 0

            if self._use_voice_bhili:
                stream = self._iter_voice_bhili_text(user_message)
            else:
                stream = self._stream_vistaar_completions(user_message)

            async for chunk in stream:

                if first_chunk:
                    first_chunk = False
                    elapsed = time.perf_counter() - start_time
                    logger.info(f"🚀 First chunk received at {elapsed:.2f}s")
                    first_chunk_arrived.set()
                    await self.stop_ttfb_metrics()

                await self.push_frame(LLMTextFrame(text=chunk))
                chunk_count += 1

            logger.info(f"✅ Completed - {chunk_count} chunks streamed")

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            first_chunk_arrived.set()  # Prevent hold message on error
            raise

        finally:
            if not timer_task.done():
                timer_task.cancel()
                try:
                    await timer_task
                except asyncio.CancelledError:
                    pass

    def _yield_word_chunks_from_text(self, text: str) -> Iterator[str]:
        """Split plain text on whitespace/newlines; same chunk shape as streaming path."""
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

    async def _iter_voice_bhili_text(self, query: str):
        """GET Voice Bhili API; JSON {response: ...}; yield word chunks for TTS."""
        session_id = self._vistaar_session_id or str(uuid.uuid4())
        params = {
            "query": query,
            "session_id": session_id,
            "source_lang": self._source_lang,
            "target_lang": self._target_lang,
        }
        headers = {"Accept": "application/json"}

        logger.info(
            f"📡 Voice Bhili API | session_id={session_id} | query={query[:50]}..."
        )

        session = await self._get_session()
        async with session.get(
            self._voice_bhili_url, params=params, headers=headers
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"❌ Voice Bhili API error {response.status}: {error_text}")
                raise Exception(f"Voice Bhili API Error {response.status}")

            data = await response.json()
            text = ""
            if isinstance(data, dict):
                text = data.get("response") or ""
            if not (text or "").strip():
                logger.warning("⚠️ Voice Bhili returned empty response")
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
        """Stream words from Vistaar production API with JWT auth."""
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

        logger.info(f"📡 Vistaar API request | session_id={session_id} | query={query[:50]}...")

        session = await self._get_session()

        async with session.get(url, params=params, headers=headers) as response:

            if response.status != 200:
                error_text = await response.text()
                logger.error(f"❌ API error {response.status}: {error_text}")
                raise Exception(f"Vistaar API Error {response.status}")

            logger.debug("✅ Connected, streaming...")

            buffer = ""
            decoder = codecs.getincrementaldecoder("utf-8")("replace")

            async for data in response.content.iter_any():
                buffer += decoder.decode(data, final=False)

                # Extract complete words
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
                    buffer = buffer[split_idx + 1:]

                    if word:
                        yield word + " "

            # Flush decoder and remaining buffer
            buffer += decoder.decode(b"", final=True)
            if buffer.strip():
                yield buffer.strip()

            logger.debug("✅ Stream complete")

    async def cleanup(self):
        """Close shared session on shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("🧹 aiohttp session closed")