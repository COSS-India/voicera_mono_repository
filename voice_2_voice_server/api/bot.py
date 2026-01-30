"""Voice bot pipeline implementation using Pipecat."""

import os
import json
import time
import traceback
from datetime import datetime

from loguru import logger
from dotenv import load_dotenv



from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator, Aggregation, AggregationType
from typing import Any
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from storage.minio_client import MinIOStorage
from serializer.vobiz_serializer import VobizFrameSerializer
from .services import (
    create_llm_service,
    create_stt_service,
    create_tts_service,
    ServiceCreationError,
)
# Import the new filter
from services.audio.greeting_interruption_filter import GreetingInterruptionFilter
from .call_recording_utils import submit_call_recording



load_dotenv(override=False)


# Monkey-patch SOXRStreamAudioResampler to reduce latency from ~200ms to near-zero
# by switching from "VHQ" (Very High Quality) to "Quick" quality.
try:
    from pipecat.audio.resamplers.soxr_stream_resampler import SOXRStreamAudioResampler
    import soxr
    import time
    
    def patched_initialize(self, in_rate: float, out_rate: float):
        self._in_rate = in_rate
        self._out_rate = out_rate
        self._last_resample_time = time.time()
        # "QQ" = Quick Quality (Cubic/Linear), minimal buffer
        # "VHQ" = Very High Quality (Sinc), large FIR filter buffer
        self._soxr_stream = soxr.ResampleStream(
            in_rate=in_rate, out_rate=out_rate, num_channels=1, quality="QQ", dtype="int16"
        )
    
    SOXRStreamAudioResampler._initialize = patched_initialize
    logger.info("Monkey-patched SOXRStreamAudioResampler for low latency (Quick quality)")
except Exception as e:
    logger.warning(f"Failed to patch SOXRStreamAudioResampler: {e}")



def _get_sample_rate() -> int:
    """Get the audio sample rate from environment."""
    return int(os.getenv("SAMPLE_RATE", "8000"))


class FastPunctuationAggregator(BaseTextAggregator):
    """Fast aggregator that sends text immediately on punctuation - no lookahead/NLTK."""
    
    def __init__(self):
        self._text = ""
    
    @property
    def text(self):
        return Aggregation(text=self._text.strip(), type=AggregationType.SENTENCE)
    
    async def aggregate(self, text: str):
        for char in text:
            self._text += char
            if char in '.!?,':
                if self._text.strip():
                    yield Aggregation(self._text.strip(), AggregationType.SENTENCE)
                    self._text = ""
    
    async def flush(self):
        if self._text.strip():
            result = self._text.strip()
            self._text = ""
            return Aggregation(result, AggregationType.SENTENCE)
        return None
    
    async def handle_interruption(self):
        self._text = ""
    
    async def reset(self):
        self._text = ""


async def run_bot(
    transport: FastAPIWebsocketTransport,
    agent_config: dict,
    audiobuffer: AudioBufferProcessor,
    transcript: TranscriptProcessor,
    handle_sigint: bool = False,
    vad_analyzer: Any = None
) -> None:
    """Run the voice bot pipeline with the given configuration.
    
    Args:
        transport: WebSocket transport for audio I/O
        agent_config: Agent configuration dictionary
        audiobuffer: Audio buffer processor for recording
        transcript: Transcript processor for saving transcripts
        handle_sigint: Whether to handle SIGINT for graceful shutdown
    """
    start_time = time.monotonic()
    sample_rate = _get_sample_rate()
    
    logger.debug(f"Agent config: {json.dumps(agent_config, indent=2, default=str)}")
    
    try:
        llm_config = agent_config.get("llm_model", {})
        stt_config = agent_config.get("stt_model", {})
        tts_config = agent_config.get("tts_model", {})
        
        language = agent_config.get("language")
        if language:
            if not stt_config.get("language"):
                stt_config["language"] = language
            if not tts_config.get("language"):
                tts_config["language"] = language
     
        llm = create_llm_service(llm_config)
        stt = create_stt_service(stt_config, sample_rate, vad_analyzer=vad_analyzer)
        tts = create_tts_service(tts_config, sample_rate)
        
        # Use fast aggregator (no lookahead/NLTK) for lower latency
        tts._aggregate_sentences = True
        tts._text_aggregator = FastPunctuationAggregator()

        system_prompt = agent_config.get("system_prompt", None)
        context = OpenAILLMContext([{"role": "system", "content": system_prompt}])
        
        # Use stored user aggregator params if available (for OpenAI services)
        user_params = getattr(llm, "_user_aggregator_params", None)
        if user_params:
            context_aggregator = llm.create_context_aggregator(context, user_params=user_params)
        else:
            context_aggregator = llm.create_context_aggregator(context)
        
        greeting_filter = GreetingInterruptionFilter()
        
        pipeline = Pipeline([
            transport.input(),
            greeting_filter,
            stt,
            transcript.user(),
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            transcript.assistant(),
            audiobuffer,
            context_aggregator.assistant(),
        ])
        
        task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=True),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            await audiobuffer.start_recording()
            greeting = agent_config.get("greeting_message", '')
            if len(greeting.strip()) > 1:
                logger.info(f"greeting: {greeting}")
                greeting_filter.start_greeting()
                await task.queue_frames([TTSSpeakFrame(greeting)])
        
        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()
            
        
        runner = PipelineRunner(handle_sigint=handle_sigint)
        await runner.run(task)
        
    except ServiceCreationError as e:
        logger.error(f"Service creation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        duration = time.monotonic() - start_time
        logger.info(f"Call ended after {duration:.1f}s")


async def bot(
    websocket_client,
    stream_sid: str,
    call_sid: str,
    agent_type: str,
    agent_config: dict
) -> None:
    """Main bot entry point - sets up transport and runs the pipeline."""
    sample_rate = _get_sample_rate()
    session_timeout = agent_config.get("session_timeout_minutes", 10) * 60

    import time
    original_send = websocket_client.send_text
    async def timed_send(data):
        if "playAudio" in str(data)[:50]:
            logger.info(f"📤 WS SEND: {len(data)} bytes at {time.perf_counter()*1000:.0f}ms")
        return await original_send(data)
    websocket_client.send_text = timed_send
    
    # Track call start time
    call_start_time = time.monotonic()
    start_time_utc = datetime.utcnow().isoformat()
    
    # Initialize MinIO storage
    storage = MinIOStorage.from_env()
    
    serializer = VobizFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        params=VobizFrameSerializer.InputParams(
            vobiz_sample_rate=sample_rate,
            sample_rate=sample_rate
        )
    )
    
    vad_analyzer = SileroVADAnalyzer(
        sample_rate=sample_rate,
        params=VADParams(
            stop_secs=0.2,
            min_volume=0.5,
            confidence=0.4,
        )
    )
    vad_analyzer._smoothing_factor = 0.1  # Faster volume change response
    import pipecat.transports.base_input
    pipecat.transports.base_input.AUDIO_INPUT_TIMEOUT_SECS = 0.1

    import pipecat.transports.base_output
    pipecat.transports.base_output.BOT_VAD_STOP_SECS = 0.2
    
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=vad_analyzer,
            serializer=serializer,
            audio_in_passthrough=True,
            session_timeout=session_timeout,
            audio_out_10ms_chunks=2,  # ADD THIS LINE - reduces from 4 to 1
        ),
    )
    transport.output()._send_interval = 0


    original_start = transport.output().start
    async def optimized_start(self, frame):
        await original_start(frame)
        self._send_interval = 0  # Override to send immediately
    transport.output().start = optimized_start.__get__(transport.output(), type(transport.output()))
    
    # Optimize: Send first audio chunk immediately, bypassing queue
    # Hook into set_transport_ready to patch media senders after they're created
    original_set_transport_ready = transport.output().set_transport_ready
    async def patched_set_transport_ready(self, frame):
        await original_set_transport_ready(frame)
        # Patch all media senders to use immediate first chunk sending
        for sender in self._media_senders.values():
            # Store state directly on the sender instance
            sender._first_chunk_sent = False
            
            async def immediate_first_chunk_handle_audio_frame(sender_self, frame):
                """Send first chunk immediately, then use normal buffering for subsequent chunks."""
                if not sender_self._params.audio_out_enabled:
                    return
                
                # Resample if needed
                resampled = await sender_self._resampler.resample(
                    frame.audio, frame.sample_rate, sender_self._sample_rate
                )
                
                cls = type(frame)
                sender_self._audio_buffer.extend(resampled)
                
                # Send first chunk immediately when we have enough data
                if not sender_self._first_chunk_sent and len(sender_self._audio_buffer) >= sender_self._audio_chunk_size:
                    first_chunk = cls(
                        bytes(sender_self._audio_buffer[:sender_self._audio_chunk_size]),
                        sample_rate=sender_self._sample_rate,
                        num_channels=frame.num_channels,
                    )
                    first_chunk.transport_destination = sender_self._destination
                    
                    # Send directly to transport, bypassing queue for minimal latency
                    logger.info(f"🚀 Sending first chunk immediately: {len(first_chunk.audio)} bytes (bypassing queue)")
                    # Reset _next_send_time to past to ensure zero sleep delay
                    output_transport = sender_self._transport
                    if hasattr(output_transport, '_next_send_time'):
                        import time
                        # Set to slightly in the past to guarantee sleep_duration = 0
                        output_transport._next_send_time = time.monotonic() - 0.001
                    # Send the frame (will have zero sleep delay)
                    await output_transport.write_audio_frame(first_chunk)
                    sender_self._first_chunk_sent = True
                    
                    # Remove sent chunk from buffer
                    sender_self._audio_buffer = sender_self._audio_buffer[sender_self._audio_chunk_size:]
                
                # Continue with normal buffering for remaining/next chunks
                while len(sender_self._audio_buffer) >= sender_self._audio_chunk_size:
                    chunk = cls(
                        bytes(sender_self._audio_buffer[:sender_self._audio_chunk_size]),
                        sample_rate=sender_self._sample_rate,
                        num_channels=frame.num_channels,
                    )
                    chunk.transport_destination = sender_self._destination
                    await sender_self._audio_queue.put(chunk)
                    sender_self._audio_buffer = sender_self._audio_buffer[sender_self._audio_chunk_size:]
            
            sender.handle_audio_frame = immediate_first_chunk_handle_audio_frame.__get__(
                sender, type(sender)
            )
            logger.debug(f"✅ Patched media sender for immediate first chunk sending")
    
    transport.output().set_transport_ready = patched_set_transport_ready.__get__(
        transport.output(), type(transport.output())
    )
    
    # Reset first_chunk_sent flag for each new TTS response
    from pipecat.frames.frames import TTSStartedFrame
    original_process_frame = transport.output().process_frame
    async def patched_process_frame(self, frame, direction):
        # Reset flag when new TTS session starts
        if isinstance(frame, TTSStartedFrame):
            for sender in self._media_senders.values():
                if hasattr(sender, '_first_chunk_sent'):
                    sender._first_chunk_sent = False
                    logger.debug(f"🔄 Reset first_chunk_sent flag for new TTS response")
        
        # Call original process_frame
        await original_process_frame(frame, direction)
    
    transport.output().process_frame = patched_process_frame.__get__(
        transport.output(), type(transport.output())
    )
    
    # Create audio buffer processor
    audiobuffer = AudioBufferProcessor()
    
    # Accumulate audio chunks and transcript lines in memory (deferred storage)
    # Using a dict to avoid nonlocal issues
    call_data = {
        "audio_chunks": [],
        "audio_sample_rate": None,
        "audio_num_channels": None,
        "transcript_lines": []
    }
    
    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        # Accumulate audio chunks in memory (no I/O during call)
        call_data["audio_chunks"].append(audio)
        # Store sample rate and channels from first chunk (should be constant)
        if call_data["audio_sample_rate"] is None:
            call_data["audio_sample_rate"] = sample_rate
            call_data["audio_num_channels"] = num_channels
        total_bytes = sum(len(c) for c in call_data["audio_chunks"])
        logger.debug(f"Accumulated audio chunk: {len(audio)} bytes (total: {total_bytes} bytes)")
    
    # Create transcript processor
    transcript = TranscriptProcessor()
    
    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        # Accumulate transcript lines in memory (no I/O during call)
        for message in frame.messages:
            timestamp = f"[{message.timestamp}] " if message.timestamp else ""
            line = f"{timestamp}{message.role}: {message.content}"
            logger.info(f"Transcript: {line}")
            call_data["transcript_lines"].append(line)
    
    try:
        await run_bot(transport, agent_config, audiobuffer, transcript, handle_sigint=False, vad_analyzer=vad_analyzer)
    finally:
        logger.info(f"Saving call data for {call_sid}...")
        if call_data["audio_chunks"] and call_data["audio_sample_rate"] and call_data["audio_num_channels"]:
            try:
                await storage.save_recording_from_chunks(
                    call_sid, 
                    call_data["audio_chunks"], 
                    call_data["audio_sample_rate"], 
                    call_data["audio_num_channels"]
                )
                total_bytes = sum(len(c) for c in call_data["audio_chunks"])
                logger.info(f" Saved {len(call_data['audio_chunks'])} audio chunks ({total_bytes} bytes)")
            except Exception as e:
                logger.error(f"Failed to save audio recording: {e}")
        else:
            logger.warning(f"No audio data to save for {call_sid}")

        if call_data["transcript_lines"]:
            try:
                await storage.save_transcript_from_lines(call_sid, call_data["transcript_lines"])
                logger.info(f" Saved {len(call_data['transcript_lines'])} transcript lines")
            except Exception as e:
                logger.error(f" Failed to save transcript: {e}")
        else:
            logger.warning(f"No transcript data to save for {call_sid}")
        
        await submit_call_recording(
            call_sid=call_sid,
            agent_type=agent_type,
            agent_config=agent_config,
            storage=storage,
            call_start_time=call_start_time
        )