"""Voice bot pipeline implementation using Pipecat.

Upgraded to Pipecat 0.0.101 with:
- LLMContext + LLMContextAggregatorPair (universal context aggregator)
- Smart Turn v3 (ML-based end-of-turn detection)
- User Turn Strategies (configurable interruption/turn-taking)
- VAD moved from transport to LLMUserAggregatorParams (new recommended pattern)
"""

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
from pipecat.pipeline.task import PipelineTask

# New universal context (replaces deprecated OpenAILLMContext)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStoppedMessage,
)

from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

# VAD
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# Smart Turn v3 - ML-based end-of-turn detection
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

# User Turn Strategies (replaces PipelineParams.allow_interruptions)
from pipecat.turns.user_start import (
    VADUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# Mute strategy: suppresses all user input until bot finishes first speech (greeting)
# This replaces the old GreetingInterruptionFilter which can't work with the new
# turn management architecture where turns are managed inside LLMUserAggregator
from pipecat.turns.user_mute.mute_until_first_bot_complete_user_mute_strategy import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
)


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

from .call_recording_utils import submit_call_recording


load_dotenv(override=False)


# NOTE: SOXR resampler monkey-patch REMOVED.
# Previously patched from VHQ (Very High Quality) to QQ (Quick) to reduce latency.
# With Smart Turn v3 handling end-of-turn detection, the ~200ms resampler latency
# is negligible compared to Smart Turn inference (~100ms) and network latency (50-200ms).
# VHQ uses a proper sinc anti-aliasing filter — QQ's cubic interpolation was causing
# audible "shh" artifacts (aliasing) especially on sibilant sounds.
# Using Pipecat's default VHQ for clean audio quality.


def _get_sample_rate() -> int:
    """Get the audio sample rate from environment."""
    return int(os.getenv("SAMPLE_RATE", "8000"))


async def run_bot(
    transport: FastAPIWebsocketTransport,
    agent_config: dict,
    audiobuffer: AudioBufferProcessor,
    call_data: dict,
    handle_sigint: bool = False,
    vad_analyzer: SileroVADAnalyzer = None,
) -> None:
    """Run the voice bot pipeline with the given configuration.

    Args:
        transport: WebSocket transport for audio I/O
        agent_config: Agent configuration dictionary
        audiobuffer: Audio buffer processor for recording
        call_data: Shared dict for accumulating transcript lines
        handle_sigint: Whether to handle SIGINT for graceful shutdown
        vad_analyzer: SileroVADAnalyzer instance (now passed to LLMUserAggregatorParams)
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

        system_prompt = agent_config.get("system_prompt", None)

        # --- Pipecat 0.0.101: Universal LLMContext + LLMContextAggregatorPair ---
        # Replaces: OpenAILLMContext + llm.create_context_aggregator()
        context = LLMContext([{"role": "system", "content": system_prompt}])

        # Read Smart Turn config from env vars (tuneable without rebuild)
        # stop_secs: silence fallback — if ML model keeps saying "incomplete",
        # force COMPLETE after this many seconds of silence. Lower = more responsive
        # but may cut off mid-thought. Default 1.0s is good for telephony.
        # (Pipecat default is 3.0 but that's too long for phone calls with background noise)
        smart_turn_stop_secs = float(os.getenv("SMART_TURN_STOP_SECS", "1.0"))
        smart_turn_pre_speech_ms = float(os.getenv("SMART_TURN_PRE_SPEECH_MS", "500.0"))
        smart_turn_max_duration = float(
            os.getenv("SMART_TURN_MAX_DURATION_SECS", "8.0")
        )
        enable_smart_turn = os.getenv("ENABLE_SMART_TURN", "true").lower() in (
            "true",
            "1",
            "yes",
        )

        # Build stop strategies
        if enable_smart_turn:
            logger.info(
                f"Smart Turn v3 ENABLED: stop_secs={smart_turn_stop_secs}, "
                f"pre_speech_ms={smart_turn_pre_speech_ms}, "
                f"max_duration={smart_turn_max_duration}"
            )
            stop_strategies = [
                TurnAnalyzerUserTurnStopStrategy(
                    turn_analyzer=LocalSmartTurnAnalyzerV3(
                        sample_rate=sample_rate,
                        params=SmartTurnParams(
                            stop_secs=smart_turn_stop_secs,
                            pre_speech_ms=smart_turn_pre_speech_ms,
                            max_duration_secs=smart_turn_max_duration,
                        ),
                    )
                )
            ]
        else:
            logger.info("Smart Turn DISABLED, using transcription-based turn detection")
            stop_strategies = (
                None  # Falls back to default TranscriptionUserTurnStopStrategy
            )

        # Build start strategies — VAD for fast turn detection, transcription for interruptions
        # VADUserTurnStartStrategy: detects speech start via VAD (fast, ~50ms)
        #   enable_interruptions=False: VAD alone cannot interrupt the bot.
        #   This prevents background conversations/noise from killing bot responses.
        # TranscriptionUserTurnStartStrategy: detects speech via STT transcription
        #   enable_interruptions=True: only actual transcribed speech can interrupt.
        #   Background noise rarely produces coherent transcription, so this is robust.
        enable_interruptions = os.getenv("ENABLE_INTERRUPTIONS", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        start_strategies = [
            VADUserTurnStartStrategy(enable_interruptions=False),
            TranscriptionUserTurnStartStrategy(
                enable_interruptions=enable_interruptions
            ),
        ]

        # Build user turn strategies
        user_turn_strategies_kwargs = {"start": start_strategies}
        if stop_strategies:
            user_turn_strategies_kwargs["stop"] = stop_strategies

        # user_turn_stop_timeout: fallback timeout when neither Smart Turn nor
        # transcription triggers a turn-end. Default is 5.0s which is too slow
        # for telephony — short utterances like "Yes" may not give Smart Turn
        # enough audio to analyze, so this timeout is the only thing that fires.
        user_turn_stop_timeout = float(os.getenv("USER_TURN_STOP_TIMEOUT", "1.5"))

        # Mute strategy: suppress all user input (audio, VAD, transcriptions)
        # until the bot finishes its first speech (the greeting).
        # This replaces the old GreetingInterruptionFilter which blocked frames
        # in the pipeline but couldn't prevent the LLMUserAggregator's internal
        # turn controller from starting a turn during the greeting.
        greeting_message = agent_config.get("greeting_message", "")
        user_mute_strategies = []
        if len(greeting_message.strip()) > 1:
            user_mute_strategies.append(MuteUntilFirstBotCompleteUserMuteStrategy())
            logger.info(
                "Greeting mute strategy enabled: user muted until greeting completes"
            )

        # Build the context aggregator with VAD + Smart Turn + strategies
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=vad_analyzer,
                user_turn_strategies=UserTurnStrategies(**user_turn_strategies_kwargs),
                user_turn_stop_timeout=user_turn_stop_timeout,
                user_mute_strategies=user_mute_strategies,
            ),
        )

        logger.info(
            f"Context aggregator: LLMContextAggregatorPair with "
            f"interruptions={enable_interruptions}, "
            f"smart_turn={enable_smart_turn}, "
            f"mute_until_greeting={len(user_mute_strategies) > 0}"
        )

        # Transcript capture via aggregator turn events (replaces deprecated TranscriptProcessor).
        # These fire once per complete turn with the full assembled text,
        # instead of once per STT segment which caused fragmented transcripts.
        @context_aggregator.user().event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(
            aggregator, strategy, message: UserTurnStoppedMessage
        ):
            if message.content.strip():
                timestamp = f"[{message.timestamp}] " if message.timestamp else ""
                line = f"{timestamp}user: {message.content}"
                logger.info(f"Transcript: {line}")
                call_data["transcript_lines"].append(line)

        @context_aggregator.assistant().event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(
            aggregator, message: AssistantTurnStoppedMessage
        ):
            if message.content.strip():
                timestamp = f"[{message.timestamp}] " if message.timestamp else ""
                line = f"{timestamp}assistant: {message.content}"
                logger.info(f"Transcript: {line}")
                call_data["transcript_lines"].append(line)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                audiobuffer,
                context_aggregator.assistant(),
            ]
        )

        # Pipecat 0.0.101: allow_interruptions is deprecated on PipelineParams.
        # Interruptions are now controlled via UserTurnStrategies start strategy
        # (enable_interruptions=True on VADUserTurnStartStrategy above).
        task = PipelineTask(pipeline)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            await audiobuffer.start_recording()
            greeting = agent_config.get("greeting_message", "")
            if len(greeting.strip()) > 1:
                logger.info(f"greeting: {greeting}")
                # MuteUntilFirstBotCompleteUserMuteStrategy handles greeting
                # protection — user input is muted until this TTS finishes
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
    agent_config: dict,
) -> None:
    """Main bot entry point - sets up transport and runs the pipeline."""
    sample_rate = _get_sample_rate()
    session_timeout = agent_config.get("session_timeout_minutes", 10) * 60

    # Track call start time
    call_start_time = time.monotonic()
    start_time_utc = datetime.utcnow().isoformat()

    # Initialize MinIO storage
    storage = MinIOStorage.from_env()

    serializer = VobizFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        params=VobizFrameSerializer.InputParams(
            vobiz_sample_rate=sample_rate, sample_rate=sample_rate
        ),
    )

    # =========================================================================
    # VAD Configuration (env-configurable, tuneable without rebuild)
    # =========================================================================
    # VAD stop_secs controls how long silence must last before VAD fires
    # "user stopped speaking". With Smart Turn enabled, this triggers the ML
    # model to analyze whether the turn is complete.
    #
    # Too low (0.2): every thinking pause triggers the ML model, which may
    # predict COMPLETE on natural mid-sentence pauses → fragmented turns.
    # Too high (1.0+): sluggish response — user waits too long after finishing.
    # 0.5s is a good balance: skips brief thinking pauses, still responsive.
    #
    # Without Smart Turn, stop_secs is the ONLY end-of-turn signal, so 0.8s
    # gives the user more breathing room.
    enable_smart_turn = os.getenv("ENABLE_SMART_TURN", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    default_stop_secs = "0.5" if enable_smart_turn else "0.8"

    vad_confidence = float(os.getenv("VAD_CONFIDENCE", "0.7"))
    vad_min_volume = float(os.getenv("VAD_MIN_VOLUME", "0.6"))
    vad_start_secs = float(os.getenv("VAD_START_SECS", "0.2"))
    vad_stop_secs = float(os.getenv("VAD_STOP_SECS", default_stop_secs))

    logger.info(
        f"VAD config: confidence={vad_confidence}, min_volume={vad_min_volume}, "
        f"start_secs={vad_start_secs}, stop_secs={vad_stop_secs}"
    )

    vad_analyzer = SileroVADAnalyzer(
        sample_rate=sample_rate,
        params=VADParams(
            confidence=vad_confidence,
            min_volume=vad_min_volume,
            start_secs=vad_start_secs,
            stop_secs=vad_stop_secs,
        ),
    )
    # NOTE: Removed aggressive monkey-patches from 0.0.98:
    #   - vad_analyzer._smoothing_factor = 0.1 (was causing noise sensitivity, default 0.2 is better)
    #   - AUDIO_INPUT_TIMEOUT_SECS = 0.1 (was causing false "user stopped" events, default 1.0 is correct)
    #   - BOT_VAD_STOP_SECS = 0.2 (was too aggressive, default 0.35 is fine)

    # BOT_VAD_STOP_SECS: how long bot silence before "bot stopped speaking" event.
    # Slightly above default (0.35) to give TTS breathing room between sentences.
    bot_vad_stop_secs = float(os.getenv("BOT_VAD_STOP_SECS", "0.5"))
    import pipecat.transports.base_output

    pipecat.transports.base_output.BOT_VAD_STOP_SECS = bot_vad_stop_secs
    logger.info(f"BOT_VAD_STOP_SECS set to {bot_vad_stop_secs}")

    # =========================================================================
    # Transport Configuration
    # =========================================================================
    # NOTE: vad_analyzer is NO LONGER passed to the transport.
    # In Pipecat 0.0.101, VAD is passed to LLMUserAggregatorParams instead
    # (handled in run_bot). The transport just handles audio I/O.
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
            audio_in_passthrough=True,
            session_timeout=session_timeout,
            audio_out_10ms_chunks=2,
        ),
    )

    # Create audio buffer processor
    audiobuffer = AudioBufferProcessor()

    # Accumulate audio chunks and transcript lines in memory (deferred storage)
    # Using a dict to avoid nonlocal issues
    call_data = {
        "audio_chunks": [],
        "audio_sample_rate": None,
        "audio_num_channels": None,
        "transcript_lines": [],
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
        logger.debug(
            f"Accumulated audio chunk: {len(audio)} bytes (total: {total_bytes} bytes)"
        )

    try:
        await run_bot(
            transport,
            agent_config,
            audiobuffer,
            call_data,
            handle_sigint=False,
            vad_analyzer=vad_analyzer,
        )
    finally:
        logger.info(f"Saving call data for {call_sid}...")
        if (
            call_data["audio_chunks"]
            and call_data["audio_sample_rate"]
            and call_data["audio_num_channels"]
        ):
            try:
                await storage.save_recording_from_chunks(
                    call_sid,
                    call_data["audio_chunks"],
                    call_data["audio_sample_rate"],
                    call_data["audio_num_channels"],
                )
                total_bytes = sum(len(c) for c in call_data["audio_chunks"])
                logger.info(
                    f" Saved {len(call_data['audio_chunks'])} audio chunks ({total_bytes} bytes)"
                )
            except Exception as e:
                logger.error(f"Failed to save audio recording: {e}")
        else:
            logger.warning(f"No audio data to save for {call_sid}")

        if call_data["transcript_lines"]:
            try:
                await storage.save_transcript_from_lines(
                    call_sid, call_data["transcript_lines"]
                )
                logger.info(
                    f" Saved {len(call_data['transcript_lines'])} transcript lines"
                )
            except Exception as e:
                logger.error(f" Failed to save transcript: {e}")
        else:
            logger.warning(f"No transcript data to save for {call_sid}")

        await submit_call_recording(
            call_sid=call_sid,
            agent_type=agent_type,
            agent_config=agent_config,
            storage=storage,
            call_start_time=call_start_time,
        )
