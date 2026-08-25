"""TTS-only pipeline for non-conversational alert agents."""

import time
import traceback
from typing import Awaitable, Callable, Optional

from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport

from api.services import ServiceCreationError, create_tts_service
from utils.bot_utils import FastPunctuationAggregator, get_sample_rate
from utils.call_management import AlertHangupProcessor
from utils.metrics import CallMetricsObserver


async def run_alert_bot(
    transport: FastAPIWebsocketTransport,
    agent_config: dict,
    transcript: TranscriptProcessor,
    audiobuffer: Optional[AudioBufferProcessor] = None,
    handle_sigint: bool = False,
    sample_rate: Optional[int] = None,
    on_client_connected_hook: Optional[Callable[[], Awaitable[None]]] = None,
) -> Optional[CallMetricsObserver]:
    """Run a TTS-only alert pipeline (no STT/LLM/interruptions)."""
    start_time = time.monotonic()
    sample_rate = sample_rate or get_sample_rate()
    logger.info("Running non-conversational alert pipeline")

    try:
        tts_config = dict(agent_config.get("tts_model", {}) or {})
        language = agent_config.get("language")
        if language and not tts_config.get("language"):
            tts_config["language"] = language

        tts = create_tts_service(
            tts_config, sample_rate, org_id=agent_config.get("org_id")
        )
        tts._aggregate_sentences = True
        tts._text_aggregator = FastPunctuationAggregator()

        task_ref: dict[str, Optional[PipelineTask]] = {"task": None}

        async def schedule_call_end() -> None:
            if task_ref["task"] is not None:
                await task_ref["task"].stop_when_done()

        pipeline_processors = [
            transport.input(),
            tts,
            AlertHangupProcessor(schedule_call_end),
            transcript.assistant(),
        ]
        if audiobuffer is not None:
            pipeline_processors.append(audiobuffer)
        pipeline_processors.append(transport.output())

        metrics_observer = CallMetricsObserver(
            stt_processor_name="none",
            llm_processor_name="none",
            tts_processor_name=tts.name,
        )
        task = PipelineTask(
            Pipeline(pipeline_processors),
            params=PipelineParams(allow_interruptions=False, enable_metrics=True),
            observers=[metrics_observer],
        )
        task_ref["task"] = task

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected (alert agent)")
            if audiobuffer is not None:
                await audiobuffer.start_recording()
            if on_client_connected_hook is not None:
                await on_client_connected_hook()
            alert_message = str(agent_config.get("greeting_message") or "").strip()
            if alert_message:
                logger.info(f"Playing alert message: {alert_message[:80]}...")
                await task.queue_frames([TTSSpeakFrame(alert_message)])
            else:
                logger.warning("No alert message configured — ending call")
                await schedule_call_end()

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        await PipelineRunner(handle_sigint=handle_sigint).run(task)
        return metrics_observer

    except ServiceCreationError as e:
        logger.error(f"Service creation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Alert pipeline error: {type(e).__name__}: {e}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        logger.info(f"Alert call ended after {time.monotonic() - start_time:.1f}s")
