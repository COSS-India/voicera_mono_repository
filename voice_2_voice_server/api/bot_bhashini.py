"""Voice bot pipeline implementation using Pipecat."""

import os
import json
import time
import traceback

from loguru import logger
from dotenv import load_dotenv



from pipecat.frames.frames import (
    InterruptionFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator, Aggregation, AggregationType
from typing import Any, Optional, Callable, Awaitable
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.runner.utils import parse_telephony_websocket
from storage.minio_client import MinIOStorage
from serializer.vobiz_serializer import VobizFrameSerializer
from serializer.ubona_serializer import UbonaFrameSerializer
from .services import (
    create_llm_service,
    create_stt_service,
    create_tts_service,
    ServiceCreationError,
)
from .latency_utils import build_latency_summary, record_latency_metric
# Import the new filter
from services.audio.greeting_interruption_filter import GreetingInterruptionFilter
from services.audio.marathi_idle_prompt_filter import MarathiIdlePromptFilter
from services.vllm_qwen import ensure_no_think_suffix
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


class BargeInInterruptionProcessor(FrameProcessor):
    """Smart barge-in: interrupt bot only on real human speech, never on noise.

    This applies identically during the greeting AND during normal conversation.

    Three-layer noise filter
    ────────────────────────
    Layer 1 — SileroVAD (neural, transport level)
        Runs a trained speech/non-speech classifier on every 30 ms audio window.
        confidence=0.7 means the audio must be 70%+ speech-like before
        UserStartedSpeakingFrame is emitted.
        • Cough   → typically scores 0.1–0.3  → SILENT, no frame ✓
        • Dog bark → typically scores 0.1–0.25 → SILENT, no frame ✓
        • Background noise → scores ~0.0       → SILENT, no frame ✓
        • Human speech     → scores 0.8–1.0    → UserStartedSpeakingFrame ✓

    Layer 2 — Speaking guard (_user_speaking flag)
        BargeInInterruptionProcessor only arms itself when it sees
        UserStartedSpeakingFrame (i.e. Silero approved the audio).
        If Silero was silent, _user_speaking stays False and no barge-in
        can fire — even if Bhashini's internal energy VAD happened to
        open a WebSocket and send the cough audio to the ASR server.

    Layer 3 — Transcript gate + minimum length
        Even with both layers above passed, we wait for Bhashini to return
        a real interim transcript with ≥ 3 characters.  A very loud cough
        that somehow slips Silero might produce 1–2 garbage characters;
        this gate discards those silently.  Real speech produces ≥ 3 chars.

    Result (both greeting and conversation):
        Human speaks   → all 3 layers pass → bot stops, listens   ✓
        Human coughs   → Layer 1 rejects   → bot keeps speaking   ✓
        Dog barks      → Layer 1 rejects   → bot keeps speaking   ✓
        Background noise → Layer 1 rejects → bot keeps speaking   ✓
    """

    # Minimum transcript character count to treat as real speech.
    # Raised from 1 to avoid single-character ASR artefacts from loud coughs.
    MIN_TEXT_CHARS = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._user_speaking: bool = False
        self._interrupted: bool = False

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            # Layer 1 passed: Silero confirmed speech-like audio.
            self._user_speaking = True
            self._interrupted = False
            logger.debug("Silero: speech detected — armed, waiting for transcript")

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Speech segment ended without a valid transcript → disarm quietly.
            if self._user_speaking and not self._interrupted:
                logger.debug("Silero: speech ended with no valid transcript — no barge-in")
            self._user_speaking = False
            self._interrupted = False

        elif isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            text = frame.text.strip()
            # Layers 2 + 3: Silero must have armed us AND text must be long enough.
            if self._user_speaking and not self._interrupted and len(text) >= self.MIN_TEXT_CHARS:
                self._interrupted = True
                logger.debug(
                    "Barge-in confirmed (Silero + transcript '{}') — interrupting bot",
                    text[:80],
                )
                await self.push_frame(InterruptionFrame(), direction)

        await self.push_frame(frame, direction)


def patch_immediate_first_chunk(transport, timing_state: Optional[dict] = None):
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
            if timing_state is not None:
                now = time.monotonic()
                timing_state["first_tts_audio_at"] = now
                last_user = timing_state.get("last_user_transcript_at")
                tts_started = timing_state.get("tts_started_at")
                if last_user is not None:
                    record_latency_metric(
                        timing_state,
                        service="orchestrator",
                        metric="user_transcript_to_first_tts_audio_ms",
                        value_ms=(now - last_user) * 1000.0,
                        stage="first_tts_audio",
                    )
                    logger.info(
                        "Latency | user_transcript_to_first_tts_audio_ms={:.1f}",
                        (now - last_user) * 1000.0,
                    )
                if tts_started is not None:
                    record_latency_metric(
                        timing_state,
                        service="orchestrator",
                        metric="tts_started_to_first_audio_ms",
                        value_ms=(now - tts_started) * 1000.0,
                        stage="first_tts_audio",
                    )
                    logger.info(
                        "Latency | tts_started_to_first_audio_ms={:.1f}",
                        (now - tts_started) * 1000.0,
                    )
        await _orig_write(frame)
    output.write_audio_frame = _write_immediate
    
    _orig_process = output.process_frame
    async def _reset_on_tts(frame, direction):
        if isinstance(frame, TTSStartedFrame):
            output._first_chunk_sent = False
            logger.debug(f"🔄 Reset first_chunk_sent flag for new TTS response")
            if timing_state is not None:
                timing_state["tts_started_at"] = time.monotonic()
                last_user = timing_state.get("last_user_transcript_at")
                if last_user is not None:
                    record_latency_metric(
                        timing_state,
                        service="orchestrator",
                        metric="user_transcript_to_tts_start_ms",
                        value_ms=(timing_state["tts_started_at"] - last_user) * 1000.0,
                        stage="tts_start",
                    )
                    logger.info(
                        "Latency | user_transcript_to_tts_start_ms={:.1f}",
                        (timing_state["tts_started_at"] - last_user) * 1000.0,
                    )
        await _orig_process(frame, direction)
    output.process_frame = _reset_on_tts


async def run_bot(
    transport: FastAPIWebsocketTransport,
    agent_config: dict,
    audiobuffer: AudioBufferProcessor,
    transcript: TranscriptProcessor,
    handle_sigint: bool = False,
    vad_analyzer: Any = None,
    vistaar_session_id: Optional[str] = None,
    timing_state: Optional[dict] = None,
    latency_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
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
    if timing_state is not None:
        timing_state["run_bot_started_at"] = start_time
    
    logger.debug(f"Agent config: {json.dumps(agent_config, indent=2, default=str)}")
    
    try:
        llm_config = dict(agent_config.get("llm_model", {}) or {})
        stt_config = agent_config.get("stt_model", {})
        tts_config = agent_config.get("tts_model", {})
        llm_provider_name = str(llm_config.get("name") or "").strip().lower()
        if llm_provider_name == "openai":
            llm_config["knowledge_base_enabled"] = bool(
                agent_config.get("knowledge_base_enabled", False)
            )
            llm_config["knowledge_document_ids"] = list(
                agent_config.get("knowledge_document_ids") or []
            )
            llm_config["knowledge_top_k"] = 10
        
        language = agent_config.get("language")
        if language:
            if not stt_config.get("language"):
                stt_config["language"] = language
            if not tts_config.get("language"):
                tts_config["language"] = language

        org_id = agent_config.get("org_id")
        stt_provider_name = str(stt_config.get("name") or "").strip().lower()
        
        llm = create_llm_service(
            llm_config,
            vistaar_session_id=vistaar_session_id,
            language=agent_config.get("language"),
            org_id=org_id,
            telemetry_callback=latency_callback,
        )
        stt = create_stt_service(
            stt_config,
            sample_rate,
            vad_analyzer=vad_analyzer,
            org_id=org_id,
            telemetry_callback=latency_callback,
        )
        if stt_provider_name == "bhashini" and llm_provider_name == "kenpath":
            enable_fast_turn = getattr(llm, "enable_bhashini_fast_turn", None)
            if callable(enable_fast_turn):
                enable_fast_turn()
        tts = create_tts_service(
            tts_config,
            sample_rate,
            org_id=org_id,
            telemetry_callback=latency_callback,
        )
        if timing_state is not None:
            timing_state["services_ready_at"] = time.monotonic()
            logger.info(
                "Latency | service_initialization_ms={:.1f}",
                (timing_state["services_ready_at"] - timing_state["run_bot_started_at"]) * 1000.0,
            )
        
        # Use fast aggregator (no lookahead/NLTK) for lower latency
        tts._aggregate_sentences = True
        tts._text_aggregator = FastPunctuationAggregator()

        system_prompt = agent_config.get("system_prompt", None)
        if llm_provider_name in ("qwen", "localqwen", "vllm"):
            system_prompt = ensure_no_think_suffix(system_prompt or "")
        context = OpenAILLMContext([{"role": "system", "content": system_prompt}])
        
        # Use stored user aggregator params if available (for OpenAI services)
        user_params = getattr(llm, "_user_aggregator_params", None)
        if user_params:
            context_aggregator = llm.create_context_aggregator(context, user_params=user_params)
        else:
            context_aggregator = llm.create_context_aggregator(context)
        
        # GreetingInterruptionFilter removed: Silero VAD (confidence=0.7) +
        # transcript gate in BargeInInterruptionProcessor already reject noise
        # (coughs, barks, background) without blocking real human speech.
        # Keeping the filter would prevent legitimate user barge-in on greeting.
        language_normalized = str(language or "").strip().lower()
        marathi_idle_prompt_enabled = (
            language_normalized == "marathi"
            or language_normalized == "mr"
            or language_normalized.startswith("mr-")
        )
        marathi_idle_prompt_filter = (
            MarathiIdlePromptFilter(timeout_secs=10.0) if marathi_idle_prompt_enabled else None
        )
        if marathi_idle_prompt_enabled:
            logger.info("Marathi idle prompt enabled (10s silence after bot speech)")

        pipeline_processors = [
            transport.input(),
            stt,
            BargeInInterruptionProcessor(),
            transcript.user(),
            context_aggregator.user(),
            llm,
            tts,
        ]
        if marathi_idle_prompt_filter:
            pipeline_processors.append(marathi_idle_prompt_filter)
        pipeline_processors.extend([
            transcript.assistant(),
            audiobuffer,
            transport.output(),
            context_aggregator.assistant(),
        ])

        pipeline = Pipeline(pipeline_processors)
        if timing_state is not None:
            timing_state["pipeline_built_at"] = time.monotonic()
            logger.info(
                "Latency | pipeline_build_ms={:.1f}",
                (timing_state["pipeline_built_at"] - timing_state["services_ready_at"]) * 1000.0,
            )
        
        task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=True),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            if timing_state is not None:
                timing_state["client_connected_at"] = time.monotonic()
                logger.info(
                    "Latency | client_connected_after_run_bot_ms={:.1f}",
                    (timing_state["client_connected_at"] - timing_state["run_bot_started_at"]) * 1000.0,
                )
            await audiobuffer.start_recording()
            greeting = agent_config.get("greeting_message", '')
            if len(greeting.strip()) > 1:
                logger.info(f"greeting: {greeting}")
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
        if timing_state is not None:
            timing_state["run_bot_finished_at"] = time.monotonic()
            logger.info(
                "Latency | run_bot_total_ms={:.1f}",
                (timing_state["run_bot_finished_at"] - timing_state["run_bot_started_at"]) * 1000.0,
            )


async def bot(
    websocket_client,
    stream_sid: Optional[str],
    call_sid: Optional[str],
    agent_type: str,
    agent_config: dict,
    provider: str = "vobiz",
    transcript_callback: Optional[Callable[[str, str, Optional[str]], Awaitable[None]]] = None,
    metrics_callback: Optional[Callable[[dict], Awaitable[None]]] = None,
) -> str:
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
    timing_state = {
        "call_start_at": call_start_time,
    }

    async def emit_latency_metric(entry: dict) -> None:
        if timing_state is not None:
            record_latency_metric(
                timing_state,
                service=str(entry.get("service") or "unknown"),
                metric=str(entry.get("metric") or "unknown"),
                value_ms=float(entry.get("value_ms") or 0.0),
                stage=entry.get("stage"),
                details=entry.get("details") or {},
            )
        if metrics_callback:
            try:
                await metrics_callback(entry)
            except Exception as callback_error:
                logger.debug(f"Latency callback failed: {callback_error}")
    
    # Initialize MinIO storage
    storage = MinIOStorage.from_env()

    normalized_provider = (provider or "vobiz").strip().lower()
    if normalized_provider == "plivo":
        await websocket_client.accept()
        _, telephony_call_data = await parse_telephony_websocket(websocket_client)
        stream_sid = (
            stream_sid
            or telephony_call_data.get("stream_id")
            or telephony_call_data.get("streamId")
            or "unknown"
        )
        call_sid = (
            call_sid
            or telephony_call_data.get("call_id")
            or telephony_call_data.get("callId")
            or "unknown"
        )
        serializer = PlivoFrameSerializer(
            stream_id=stream_sid,
            call_id=call_sid,
            params=PlivoFrameSerializer.InputParams(
                plivo_sample_rate=sample_rate,
                sample_rate=sample_rate,
                auto_hang_up=False,
            ),
        )
    else:
        stream_sid = stream_sid or "unknown"
        call_sid = call_sid or "unknown"
        serializer = VobizFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            params=VobizFrameSerializer.InputParams(
                vobiz_sample_rate=sample_rate,
                sample_rate=sample_rate
            )
        )
    stt_provider_name = str((agent_config.get("stt_model") or {}).get("name") or "").strip().lower()

    if stt_provider_name == "bhashini":
        # Use Silero VAD (neural speech classifier) on the transport even for
        # Bhashini.  Silero correctly ignores coughs, dog barks, background
        # noise and only fires UserStartedSpeakingFrame for real human speech.
        # Bhashini's internal energy VAD still manages WebSocket open/close
        # timing but will NOT emit duplicate speaking frames (suppress_vad_frames
        # is set True in services.py when vad_analyzer is not None).
        vad_analyzer = SileroVADAnalyzer(
            sample_rate=sample_rate,
            params=VADParams(
                stop_secs=0.2,     # 500 ms of silence ends the speech segment
                min_volume=0.6,    # ignore very quiet background hiss
                confidence=0.7,    # neural confidence threshold (0–1); coughs/barks
                                   # typically score < 0.4, real speech > 0.7
                start_secs=0.2,    # require 200 ms of sustained speech onset
            ),
        )
        vad_analyzer._smoothing_factor = 0.15
        logger.info(
            "Bhashini: using SileroVAD on transport for barge-in "
            "(confidence=0.7, min_volume=0.6) — coughs/barks will be ignored"
        )
    else:
        vad_analyzer = SileroVADAnalyzer(
            sample_rate=sample_rate,
            params=VADParams(
                stop_secs=0.4,
                min_volume=0.5,
                confidence=0.4,
                start_secs=0.1,
            ),
        )
        vad_analyzer._smoothing_factor = 0.1  # Faster volume change response

    import pipecat.transports.base_input
    # Give the transport more breathing room so short audio gaps do not
    # force premature stop/start transitions.
    pipecat.transports.base_input.AUDIO_INPUT_TIMEOUT_SECS = 0.25

    import pipecat.transports.base_output
    # Keep assistant speech grouped together across brief TTS chunk gaps.
    pipecat.transports.base_output.BOT_VAD_STOP_SECS = 0.6
    
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
    patch_immediate_first_chunk(transport, timing_state=timing_state)
    
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
            if message.content:
                now = time.monotonic()
                if message.role == "user":
                    timing_state["last_user_transcript_at"] = now
                    logger.info(
                        "Latency | user_transcript_received_ms={:.1f}",
                        (now - call_start_time) * 1000.0,
                    )
                elif message.role == "assistant":
                    timing_state["last_assistant_transcript_at"] = now
                    logger.info(
                        "Latency | assistant_transcript_received_ms={:.1f}",
                        (now - call_start_time) * 1000.0,
                    )
            if transcript_callback and message.content:
                try:
                    await transcript_callback(message.role, message.content, message.timestamp)
                except Exception as callback_error:
                    logger.debug(f"Transcript callback failed: {callback_error}")
    
    try:
        await run_bot(
            transport,
            agent_config,
            audiobuffer,
            transcript,
            handle_sigint=False,
            vad_analyzer=vad_analyzer,
            vistaar_session_id=call_sid,
            timing_state=timing_state,
            latency_callback=emit_latency_metric,
        )
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

        latency_summary = build_latency_summary(timing_state)
        if timing_state is not None:
            timing_state["latency_summary"] = latency_summary
        if metrics_callback:
            try:
                await metrics_callback(
                    {
                        "event": "latency_summary",
                        "call_sid": call_sid,
                        "summary": latency_summary,
                    }
                )
            except Exception as callback_error:
                logger.debug(f"Latency summary callback failed: {callback_error}")
        
        await submit_call_recording(
            call_sid=call_sid,
            agent_type=agent_type,
            agent_config=agent_config,
            storage=storage,
            call_start_time=call_start_time,
            latency_summary=latency_summary,
        )
    return call_sid

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
    # Give the transport more breathing room so short audio gaps do not
    # force premature stop/start transitions.
    pipecat.transports.base_input.AUDIO_INPUT_TIMEOUT_SECS = 0.25
    import pipecat.transports.base_output
    # Keep assistant speech grouped together across brief TTS chunk gaps.
    pipecat.transports.base_output.BOT_VAD_STOP_SECS = 0.6

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
        await run_bot(transport, agent_config, audiobuffer, transcript, vad_analyzer=vad_analyzer, vistaar_session_id=call_id)
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