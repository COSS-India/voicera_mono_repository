"""Voice bot pipeline implementation using Pipecat."""

import json
import time
import traceback

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
from typing import Any, Optional, Callable, Awaitable
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.runner.utils import parse_telephony_websocket
from storage.minio_client import MinIOStorage
from serializer.vobiz_serializer import VobizFrameSerializer
from .services import (
    create_llm_service,
    create_stt_service,
    create_tts_service,
    ServiceCreationError,
)
from utils.audio.greeting_interruption_filter import create_greeting_filters
from utils.audio.user_online_detection_filter import UserOnlineDetectionFilter
from utils.call_goodbye import GoodbyeHangupProcessor
from utils.call_management import UserSilenceHangupProcessor
from utils.pipelines import run_alert_bot
from services.vllm_qwen import ensure_no_think_suffix
from utils.bot_utils import (
    BargeInInterruptionProcessor,
    FastPunctuationAggregator,
    get_call_timeout_seconds,
    get_alert_call_timeout_seconds,
    get_hold_message_timeout_seconds,
    get_hold_messages,
    get_ignore_user_speech_before_greeting,
    get_interruption_min_words,
    get_sample_rate,
    get_user_online_detection_enabled,
    get_user_online_detection_message,
    get_user_online_detection_seconds,
    get_user_silence_hangup_seconds,
    is_non_conversational,
    patch_immediate_first_chunk,
)
from utils.call_recording_utils import submit_call_recording
from utils.vobiz_recording import start_vobiz_call_recording, wait_and_download_vobiz_recording
from utils.metrics import CallMetricsObserver



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


async def run_bot(
    transport: FastAPIWebsocketTransport,
    agent_config: dict,
    transcript: TranscriptProcessor,
    audiobuffer: Optional[AudioBufferProcessor] = None,
    handle_sigint: bool = False,
    vad_analyzer: Any = None,
    vistaar_session_id: Optional[str] = None,
    sample_rate: Optional[int] = None,
    on_client_connected_hook: Optional[Callable[[], Awaitable[None]]] = None,
) -> Optional[CallMetricsObserver]:
    """Run the voice bot pipeline with the given configuration.
    
    Args:
        transport: WebSocket transport for audio I/O
        agent_config: Agent configuration dictionary
        audiobuffer: Audio buffer processor for recording
        transcript: Transcript processor for saving transcripts
        handle_sigint: Whether to handle SIGINT for graceful shutdown
    """
    start_time = time.monotonic()
    sample_rate = sample_rate or get_sample_rate()
    
    logger.debug(f"Agent config: {json.dumps(agent_config, indent=2, default=str)}")

    if is_non_conversational(agent_config):
        return await run_alert_bot(
            transport,
            agent_config,
            transcript,
            audiobuffer=audiobuffer,
            handle_sigint=handle_sigint,
            sample_rate=sample_rate,
            on_client_connected_hook=on_client_connected_hook,
        )
    
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
     
        llm = create_llm_service(
            llm_config,
            vistaar_session_id=vistaar_session_id,
            language=agent_config.get("language"),
            org_id=org_id,
            hold_messages=get_hold_messages(agent_config),
            hold_message_timeout_seconds=get_hold_message_timeout_seconds(
                agent_config
            ),
        )
        stt = create_stt_service(stt_config, sample_rate, vad_analyzer=vad_analyzer, org_id=org_id)
        tts = create_tts_service(tts_config, sample_rate, org_id=org_id)

        stt_provider_name = str(stt_config.get("name") or "").strip().lower()
        if stt_provider_name == "bhashini" and llm_provider_name == "kenpath":
            enable_fast_turn = getattr(llm, "enable_bhashini_fast_turn", None)
            if callable(enable_fast_turn):
                enable_fast_turn()
        
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
        
        _, greeting_blocker, greeting_completer = create_greeting_filters()
        ignore_user_speech_before_greeting = get_ignore_user_speech_before_greeting(agent_config)
        interruption_min_words = get_interruption_min_words(agent_config)
        logger.info(
            "Interruption settings | ignore_before_greeting={} min_words={}",
            ignore_user_speech_before_greeting,
            interruption_min_words,
        )
        task_ref: dict[str, Optional[PipelineTask]] = {"task": None}

        async def schedule_call_end() -> None:
            pipeline_task = task_ref["task"]
            if pipeline_task is not None:
                await pipeline_task.stop_when_done()

        goodbye_processor = GoodbyeHangupProcessor(schedule_call_end)

        user_online_detection_message = get_user_online_detection_message(agent_config)
        user_online_detection_enabled = (
            get_user_online_detection_enabled(agent_config)
            and bool(user_online_detection_message)
        )
        user_online_detection_seconds = get_user_online_detection_seconds(agent_config)
        user_online_detection_filter = (
            UserOnlineDetectionFilter(
                prompt_text=user_online_detection_message,
                timeout_secs=user_online_detection_seconds,
                suppress_idle_when=goodbye_processor.should_suppress_idle,
            )
            if user_online_detection_enabled
            else None
        )
        if user_online_detection_enabled:
            logger.info(
                "User online detection enabled ({}s silence after bot speech)",
                user_online_detection_seconds,
            )

        user_silence_hangup_secs = get_user_silence_hangup_seconds(agent_config)
        user_silence_hangup_filter = (
            UserSilenceHangupProcessor(
                timeout_secs=float(user_silence_hangup_secs),
                schedule_call_end=schedule_call_end,
                suppress_idle_when=goodbye_processor.should_suppress_idle,
            )
            if user_silence_hangup_secs > 0
            else None
        )
        if user_silence_hangup_filter:
            logger.info(
                "User silence hangup enabled ({}s after bot speech)",
                user_silence_hangup_secs,
            )

        pipeline_processors = [
            transport.input(),
            stt,
            greeting_blocker,
            BargeInInterruptionProcessor(min_words=interruption_min_words),
            transcript.user(),
            context_aggregator.user(),
            llm,
            goodbye_processor,
            tts,
            greeting_completer,
        ]
        if user_online_detection_filter:
            pipeline_processors.append(user_online_detection_filter)
        if user_silence_hangup_filter:
            pipeline_processors.append(user_silence_hangup_filter)
        pipeline_processors.append(transcript.assistant())
        if audiobuffer is not None:
            pipeline_processors.append(audiobuffer)
        pipeline_processors.extend([
            transport.output(),
            context_aggregator.assistant(),
        ])

        pipeline = Pipeline(pipeline_processors)

        metrics_observer = CallMetricsObserver(
            stt_processor_name=stt.name,
            llm_processor_name=llm.name,
            tts_processor_name=tts.name,
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=True, enable_metrics=True),
            observers=[metrics_observer],
        )
        task_ref["task"] = task

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            if audiobuffer is not None:
                await audiobuffer.start_recording()
            if on_client_connected_hook is not None:
                await on_client_connected_hook()
            greeting = agent_config.get("greeting_message", '')
            if len(greeting.strip()) > 1:
                logger.info(f"greeting: {greeting}")
                if ignore_user_speech_before_greeting:
                    greeting_blocker.start_greeting()
                await task.queue_frames([TTSSpeakFrame(greeting)])
        
        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()
            
        
        runner = PipelineRunner(handle_sigint=handle_sigint)
        await runner.run(task)
        return metrics_observer

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
    stream_sid: Optional[str],
    call_sid: Optional[str],
    agent_type: str,
    agent_config: dict,
    provider: str = "vobiz",
    transcript_callback: Optional[Callable[[str, str, Optional[str]], Awaitable[None]]] = None,
    sample_rate: Optional[int] = None,
) -> str:
    """Main bot entry point - sets up transport and runs the pipeline."""
    sample_rate = sample_rate or get_sample_rate()
    session_timeout = (
        get_alert_call_timeout_seconds(agent_config)
        if is_non_conversational(agent_config)
        else get_call_timeout_seconds(agent_config)
    )

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
    
    # Initialize MinIO storage
    storage = MinIOStorage.from_env()

    normalized_provider = (provider or "vobiz").strip().lower()
    if normalized_provider == "plivo":
        await websocket_client.accept()
        _, telephony_call_data = await parse_telephony_websocket(websocket_client)
        stream_sid = stream_sid or telephony_call_data.get("stream_id") or "unknown"
        call_sid = call_sid or telephony_call_data.get("call_id") or "unknown"
     
        serializer = VobizFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            params=VobizFrameSerializer.InputParams(
                vobiz_sample_rate=sample_rate,
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
       
        vad_analyzer = SileroVADAnalyzer(
            sample_rate=sample_rate,
            params=VADParams(
                stop_secs=0.2,
                min_volume=0.6,    # ignore very quiet background hiss
                confidence=0.7,    # coughs/barks typically score < 0.4, real speech > 0.7
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
                min_volume=0.4,
                confidence=0.3,
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
    
    use_vobiz_native_recording = normalized_provider == "vobiz"
    audiobuffer = None if use_vobiz_native_recording else AudioBufferProcessor()

    # Accumulate audio chunks and transcript lines in memory (deferred storage)
    call_data = {
        "audio_chunks": [],
        "audio_sample_rate": None,
        "audio_num_channels": None,
        "transcript_lines": [],
        "vobiz_recording_id": None,
    }

    if audiobuffer is not None:
        @audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            call_data["audio_chunks"].append(audio)
            if call_data["audio_sample_rate"] is None:
                call_data["audio_sample_rate"] = sample_rate
                call_data["audio_num_channels"] = num_channels
            total_bytes = sum(len(c) for c in call_data["audio_chunks"])
            logger.debug(f"Accumulated audio chunk: {len(audio)} bytes (total: {total_bytes} bytes)")

    on_client_connected_hook = None
    if use_vobiz_native_recording:
        org_id = agent_config.get("org_id")

        async def start_vobiz_recording():
            if not org_id:
                logger.warning(f"No org_id for Vobiz recording on call {call_sid}")
                return
            recording_id = await start_vobiz_call_recording(
                call_sid, org_id, session_timeout
            )
            call_data["vobiz_recording_id"] = recording_id

        on_client_connected_hook = start_vobiz_recording
    
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
            if transcript_callback and message.content:
                try:
                    await transcript_callback(message.role, message.content, message.timestamp)
                except Exception as callback_error:
                    logger.debug(f"Transcript callback failed: {callback_error}")
    
    metrics_observer = None
    try:
        metrics_observer = await run_bot(
            transport,
            agent_config,
            transcript,
            audiobuffer=audiobuffer,
            handle_sigint=False,
            vad_analyzer=vad_analyzer,
            vistaar_session_id=call_sid,
            sample_rate=sample_rate,
            on_client_connected_hook=on_client_connected_hook,
        )
    finally:
        logger.info(f"Saving call data for {call_sid}...")
        recording_url = None
        if use_vobiz_native_recording:
            recording_id = call_data.get("vobiz_recording_id")
            org_id = agent_config.get("org_id")
            if recording_id and org_id:
                try:
                    audio_bytes = await wait_and_download_vobiz_recording(
                        recording_id, org_id
                    )
                    if audio_bytes:
                        await storage.save_recording_bytes(call_sid, audio_bytes, "mp3")
                        recording_url = f"minio://recordings/{call_sid}.mp3"
                    else:
                        logger.warning(f"Vobiz recording download failed for {call_sid}")
                except Exception as e:
                    logger.error(f"Failed to ingest Vobiz recording for {call_sid}: {e}")
            else:
                logger.warning(f"No Vobiz recording_id to fetch for {call_sid}")
        elif call_data["audio_chunks"] and call_data["audio_sample_rate"] and call_data["audio_num_channels"]:
            try:
                await storage.save_recording_from_chunks(
                    call_sid,
                    call_data["audio_chunks"],
                    call_data["audio_sample_rate"],
                    call_data["audio_num_channels"],
                )
                total_bytes = sum(len(c) for c in call_data["audio_chunks"])
                logger.info(f" Saved {len(call_data['audio_chunks'])} audio chunks ({total_bytes} bytes)")
                recording_url = f"minio://recordings/{call_sid}.wav"
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
        
        latency_metrics = metrics_observer.to_dict() if metrics_observer else None
        await submit_call_recording(
            call_sid=call_sid,
            agent_type=agent_type,
            agent_config=agent_config,
            storage=storage,
            call_start_time=call_start_time,
            latency_metrics=latency_metrics,
            recording_url=recording_url,
            omit_recording_url=use_vobiz_native_recording and not recording_url,
        )
    return call_sid
