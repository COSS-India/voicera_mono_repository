from loguru import logger
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.frames.frames import LLMTextFrame, TTSSpeakFrame

# Pipecat 0.0.101: Use universal LLMContext (OpenAILLMContext is deprecated)
from pipecat.processors.aggregators.llm_context import LLMContext
import aiohttp
import asyncio
from typing import Optional


class KenpathLLM(OpenAILLMService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.response_timeout = 1  # Seconds before playing hold message

        # ✅ NEW: 4 different hold messages that rotate
        self.hold_messages = [
            "कृपया थांबा, मी माहिती शोधत आहे",  # Message 1: Please wait, I'm searching for information
            "एक क्षण थांबा, मी तपासत आहे",  # Message 2: Wait a moment, I'm checking
            "कृपया प्रतीक्षा करा, मी उत्तर शोधत आहे",  # Message 3: Please wait, I'm searching for the answer
            "थोडा वेळ द्या, मी माहिती मिळवत आहे",
        ]
        self.hold_message_index = 0  # Track which message to play next

        logger.info(f"🤖 KenpathLLM initialized with {self.response_timeout}s timeout")
        logger.info(f"📢 Loaded {len(self.hold_messages)} rotating hold messages")

    def _get_hold_message(self):
        """
        Get next hold message and rotate to the next one.
        Cycles through all 4 messages: 0 → 1 → 2 → 3 → 0 → 1...
        """
        msg = self.hold_messages[self.hold_message_index]

        # Move to next message (with wraparound)
        self.hold_message_index = (self.hold_message_index + 1) % len(
            self.hold_messages
        )

        logger.debug(f"🔄 Selected hold message #{self.hold_message_index}: '{msg}'")
        logger.debug(
            f"📌 Next message will be #{self.hold_message_index + 1 if self.hold_message_index < len(self.hold_messages) - 1 else 0}"
        )

        return msg

    async def _process_context(self, context: LLMContext):
        """
        Override to use Vistaar API with hold message on timeout.

        Flow:
        1. User stops speaking
        2. Start 3-second timer
        3. If LLM responds within 3s: Cancel timer, play response
        4. If LLM takes >3s: Play hold message (rotates through 4 messages), then play response when ready
        """
        logger.debug(
            f"{self}: Generating chat from Vistaar API context {context.get_messages_for_logging()}"
        )

        # Extract user message from context
        messages = context.get_messages()
        user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break

        if not user_message:
            logger.warning("⚠️ No user message found in context")
            return

        logger.info(f"💬 Processing user message: '{user_message[:50]}...'")

        # Create hold message task that runs in background
        hold_message_task = None

        async def play_hold_message_after_delay():
            """
            Waits for timeout duration, then plays hold message.
            Cancellable if response arrives quickly.
            """
            try:
                logger.info(f"⏱️ Starting {self.response_timeout}s timeout timer")
                await asyncio.sleep(self.response_timeout)

                # Timeout reached - play hold message (rotates through 4 messages)
                hold_msg = self._get_hold_message()
                logger.info(f"⏳ TIMEOUT! Queueing hold message: '{hold_msg}'")

                # Push TTSSpeakFrame - Pipecat's TTS will queue this
                # If LLM response arrives while this is playing, it will queue naturally
                await self.push_frame(TTSSpeakFrame(hold_msg))

                logger.info("✅ Hold message sent to TTS queue")
                return True

            except asyncio.CancelledError:
                logger.info(
                    "✅ Hold message cancelled - response arrived within timeout"
                )
                return False

        # Start the background timer task
        hold_message_task = asyncio.create_task(play_hold_message_after_delay())

        try:
            # Track if we've received first chunk (to cancel hold message)
            first_chunk = True
            chunk_count = 0

            # Stream response from Vistaar API
            async for chunk in self._stream_vistaar_completions(user_message):
                # On first chunk, cancel hold message if still pending
                if first_chunk:
                    first_chunk = False

                    if hold_message_task and not hold_message_task.done():
                        logger.info(
                            "🚀 First LLM chunk received - cancelling hold message"
                        )
                        hold_message_task.cancel()
                        try:
                            await hold_message_task
                        except asyncio.CancelledError:
                            pass  # Expected cancellation
                    else:
                        logger.info(
                            "🚀 First LLM chunk received (hold message already played/playing)"
                        )

                # Push LLM response chunk
                # If hold message was queued, TTS will finish it first, then play this
                await self.push_frame(LLMTextFrame(text=chunk))
                chunk_count += 1

            logger.info(f"✅ Streamed {chunk_count} chunks from LLM")

        except Exception as e:
            logger.error(f"❌ Error during LLM streaming: {type(e).__name__}: {e}")

            # Cancel hold message on error
            if hold_message_task and not hold_message_task.done():
                hold_message_task.cancel()
                try:
                    await hold_message_task
                except asyncio.CancelledError:
                    pass
            raise

        finally:
            # Ensure hold message task is always cleaned up
            if hold_message_task and not hold_message_task.done():
                logger.debug("🧹 Cleaning up hold message task")
                hold_message_task.cancel()
                try:
                    await hold_message_task
                except asyncio.CancelledError:
                    pass

            logger.debug("✅ _process_context completed")

    async def _stream_vistaar_completions(
        self,
        query: str,
        base_url: str = "https://vistaar-dev.mahapocra.gov.in",
        source_lang: str = "mr",
        target_lang: str = "mr",
        session_id: Optional[str] = None,
    ):
        """
        Stream completions from Vistaar API word by word.

        Args:
            query: User's query text
            base_url: Vistaar API base URL
            source_lang: Source language code
            target_lang: Target language code
            session_id: Session identifier for conversation continuity

        Yields:
            str: Words from the LLM response with trailing space
        """
        import uuid

        url = f"{base_url}/api/voice/"  # ✅ WITH trailing slash
        session_id = session_id or str(uuid.uuid4())

        # ✅ WITH underscores (matching working curl command)
        params = {
            "query": query,
            "source_lang": source_lang,  # ✅ underscore
            "target_lang": target_lang,  # ✅ underscore
            "session_id": session_id,  # ✅ underscore
        }

        logger.info(f"📡 Calling Vistaar API for session {session_id}")

        logger.debug(f"Parameters: {params}")

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"❌ Vistaar API error {response.status}: {error_text}"
                    )
                    raise Exception(
                        f"Vistaar API Error {response.status}: {error_text}"
                    )

                logger.info("✅ Connected to Vistaar API, streaming response...")

                buffer = ""
                word_count = 0

                async for data in response.content.iter_any():
                    try:
                        decoded_chunk = data.decode("utf-8")
                        buffer += decoded_chunk

                        # Split on spaces and newlines
                        while " " in buffer or "\n" in buffer:
                            space_idx = buffer.find(" ")
                            newline_idx = buffer.find("\n")

                            if space_idx == -1:
                                split_idx = newline_idx
                            elif newline_idx == -1:
                                split_idx = space_idx
                            else:
                                split_idx = min(space_idx, newline_idx)

                            if split_idx == -1:
                                break

                            word = buffer[:split_idx].strip()
                            buffer = buffer[split_idx + 1 :]

                            if word:
                                word_count += 1
                                if word_count == 1:
                                    logger.info(f"📝 First word received: '{word}'")
                                elif word_count % 10 == 0:
                                    logger.debug(f"📝 Streamed {word_count} words...")

                                yield word + " "

                    except UnicodeDecodeError:
                        logger.warning("⚠️ Unicode decode error in chunk, skipping")
                        continue

                # Yield any remaining content in buffer
                if buffer.strip():
                    word_count += 1
                    logger.debug(f"📝 Final chunk: '{buffer.strip()}'")
                    yield buffer.strip()

                logger.info(
                    f"✅ Vistaar API streaming complete. Total words: {word_count}"
                )
