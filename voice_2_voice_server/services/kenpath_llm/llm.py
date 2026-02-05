from loguru import logger
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.frames.frames import LLMTextFrame, TTSSpeakFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_context import LLMContext
import aiohttp
import asyncio
from typing import Optional
import time
import uuid


class KenpathLLM(OpenAILLMService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.response_timeout = 1.0  # seconds
        
        self.hold_messages = [
            "कृपया थांबा, मी माहिती शोधत आहे",
            "एक क्षण थांबा, मी तपासत आहे",
            "कृपया प्रतीक्षा करा, मी उत्तर शोधत आहे",
            "थोडा वेळ द्या, मी माहिती मिळवत आहे"
        ]
        self.hold_message_index = 0
        
        logger.info(f"🤖 KenpathLLM initialized with {self.response_timeout}s timeout")

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
                # Wait for either: first chunk arrives OR timeout
                await asyncio.wait_for(
                    first_chunk_arrived.wait(),
                    timeout=self.response_timeout
                )
                # First chunk arrived before timeout - do nothing
                logger.debug("✅ LLM responded before timeout")
                
            except asyncio.TimeoutError:
                # Timeout reached - play hold message
                elapsed = time.perf_counter() - start_time
                hold_msg = self._get_hold_message()
                logger.info(f"⏳ Timeout after {elapsed:.2f}s - playing: '{hold_msg}'")
                
                # Push to TTS - Pipecat will queue LLM response after this
                await self.push_frame(TTSSpeakFrame(hold_msg))
        
        # Start the timer task
        timer_task = asyncio.create_task(hold_message_timer())
        
        try:
            first_chunk = True
            chunk_count = 0
            
            # Stream from Vistaar API
            async for chunk in self._stream_vistaar_completions(user_message):
                
                # Signal first chunk arrived (cancels hold message if still waiting)
                if first_chunk:
                    first_chunk = False
                    elapsed = time.perf_counter() - start_time
                    logger.info(f"🚀 First chunk received at {elapsed:.2f}s")
                    first_chunk_arrived.set()
                
                # Push LLM response - if hold message is playing, this queues behind it
                await self.push_frame(LLMTextFrame(text=chunk))
                chunk_count += 1
            
            logger.info(f"✅ Completed - {chunk_count} chunks streamed")
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            first_chunk_arrived.set()  # Prevent hold message on error
            raise
            
        finally:
            # Clean up timer task
            if not timer_task.done():
                timer_task.cancel()
                try:
                    await timer_task
                except asyncio.CancelledError:
                    pass

    async def _stream_vistaar_completions(
        self,
        query: str,
        base_url: str = "https://vistaar-dev.mahapocra.gov.in",
        source_lang: str = "mr",
        target_lang: str = "mr",
        session_id: Optional[str] = None
    ):
        """Stream words from Vistaar API."""
        
        url = f"{base_url}/api/voice/"
        session_id = session_id or str(uuid.uuid4())
        
        params = {
            "query": query,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "session_id": session_id
        }
        
        logger.info(f"📡 Calling Vistaar API (session: {session_id[:8]}...)")
        
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=timeout) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ API error {response.status}: {error_text}")
                    raise Exception(f"Vistaar API Error {response.status}")
                
                logger.debug("✅ Connected, streaming...")
                
                buffer = ""
                
                async for data in response.content.iter_any():
                    try:
                        buffer += data.decode('utf-8')
                        
                        # Extract complete words
                        while ' ' in buffer or '\n' in buffer:
                            # Find first delimiter
                            space_idx = buffer.find(' ')
                            newline_idx = buffer.find('\n')
                            
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
                                
                    except UnicodeDecodeError:
                        continue
                
                # Yield remaining buffer
                if buffer.strip():
                    yield buffer.strip()
                
                logger.debug("✅ Stream complete")