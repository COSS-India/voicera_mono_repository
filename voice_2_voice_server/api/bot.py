"""Voice bot pipeline implementation using Pipecat."""

import os
import json
import time
import traceback
from datetime import datetime

from loguru import logger
from dotenv import load_dotenv



from pipecat.frames.frames import TTSSpeakFrame, TTSStartedFrame
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
from serializer.ubona_serializer import UbonaFrameSerializer
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


def patch_immediate_first_chunk(transport):
    """Patch transport to send first audio chunk immediately with zero delay."""
    output = transport.output()
    output._send_interval = 0
    output._first_chunk_sent = False
    
    _orig_write = output.write_audio_frame
    async def _write_immediate(frame):
        if not output._first_chunk_sent:
            output._first_chunk_sent = True
            output._next_send_time = time.monotonic() - 0.001
            logger.info(f"🚀 Sending first chunk immediately: {len(frame.audio)} bytes (bypassing queue)")
        await _orig_write(frame)
    output.write_audio_frame = _write_immediate
    
    _orig_process = output.process_frame
    async def _reset_on_tts(frame, direction):
        if isinstance(frame, TTSStartedFrame):
            output._first_chunk_sent = False
            logger.debug(f"🔄 Reset first_chunk_sent flag for new TTS response")
        await _orig_process(frame, direction)
    output.process_frame = _reset_on_tts


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
            #logger.info(f"📤 WS SEND: {len(data)} bytes at {time.perf_counter()*1000:.0f}ms")
            pass
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
            start_secs=0.1,
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

    # Optimized first audio chunk sending
    patch_immediate_first_chunk(transport)
    
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

async def ubona_bot(
    websocket_client,
    stream_id: str,
    call_id: str,
    agent_type: str,
    agent_config: dict
) -> None:
    """Ubona bot entry point - sets up transport and runs the pipeline."""
    sample_rate = 8000  # Ubona only supports 8kHz PCMU
    session_timeout = agent_config.get("session_timeout_minutes", 10) * 60

    call_start_time = time.monotonic()
    storage = MinIOStorage.from_env()

    serializer = UbonaFrameSerializer(
        stream_id=stream_id,
        call_id=call_id,
        params=UbonaFrameSerializer.InputParams(sample_rate=sample_rate),
    )

    vad_analyzer = SileroVADAnalyzer(
        sample_rate=sample_rate,
        params=VADParams(stop_secs=0.2, min_volume=0.5, confidence=0.4, start_secs=0.1),
    )
    vad_analyzer._smoothing_factor = 0.1

    import pipecat.transports.base_input
    pipecat.transports.base_input.AUDIO_INPUT_TIMEOUT_SECS = 0.1
    import pipecat.transports.base_output
    pipecat.transports.base_output.BOT_VAD_STOP_SECS = 0.2

    # Wrapper to handle ping/pong inline
    class PingPongWrapper:
        def __init__(self, ws):
            self._ws = ws
        async def receive_text(self):
            while True:
                data = await self._ws.receive_text()
                try:
                    msg = json.loads(data)
                    if msg.get("event") == "ping":
                        # Spec: pong must contain the same ts as ping for round-trip
                        ping_ts = msg.get("ts", int(time.time() * 1000))
                        await self._ws.send_text(json.dumps({"event": "pong", "ts": ping_ts}))
                        continue
                except:
                    pass
                return data
        def __getattr__(self, name):
            return getattr(self._ws, name)

    transport = FastAPIWebsocketTransport(
        websocket=PingPongWrapper(websocket_client),
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=vad_analyzer,
            serializer=serializer,
            audio_in_passthrough=True,
            session_timeout=session_timeout,
            audio_out_10ms_chunks=2,
        ),
    )

    patch_immediate_first_chunk(transport)

    audiobuffer = AudioBufferProcessor()
    transcript = TranscriptProcessor()
    call_data = {"audio_chunks": [], "audio_sample_rate": None, "audio_num_channels": None, "transcript_lines": []}

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        call_data["audio_chunks"].append(audio)
        if call_data["audio_sample_rate"] is None:
            call_data["audio_sample_rate"], call_data["audio_num_channels"] = sample_rate, num_channels

    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        for message in frame.messages:
            ts = f"[{message.timestamp}] " if message.timestamp else ""
            call_data["transcript_lines"].append(f"{ts}{message.role}: {message.content}")

    try:
        await run_bot(transport, agent_config, audiobuffer, transcript, vad_analyzer=vad_analyzer)
    finally:
        logger.info(f"Saving call data for {call_id}...")
        if call_data["audio_chunks"] and call_data["audio_sample_rate"]:
            try:
                await storage.save_recording_from_chunks(call_id, call_data["audio_chunks"], call_data["audio_sample_rate"], call_data["audio_num_channels"])
                logger.info(f"Saved {len(call_data['audio_chunks'])} audio chunks")
            except Exception as e:
                logger.error(f"Failed to save audio: {e}")

        if call_data["transcript_lines"]:
            try:
                await storage.save_transcript_from_lines(call_id, call_data["transcript_lines"])
                logger.info(f"Saved {len(call_data['transcript_lines'])} transcript lines")
            except Exception as e:
                logger.error(f"Failed to save transcript: {e}")

        await submit_call_recording(call_sid=call_id, agent_type=agent_type, agent_config=agent_config, storage=storage, call_start_time=call_start_time)
