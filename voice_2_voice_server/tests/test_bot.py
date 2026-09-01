import pytest
import json
import time
from contextlib import ExitStack
from unittest.mock import MagicMock, AsyncMock, patch, ANY

from pipecat.frames.frames import TTSSpeakFrame
from api.bot import run_bot, bot as voice_bot
from api.services import ServiceCreationError


class TestRunBot:
    @pytest.mark.asyncio
    @patch("api.bot.is_non_conversational")
    @patch("api.bot.run_alert_bot")
    async def test_run_bot_non_conversational(
        self, mock_run_alert_bot, mock_is_non_conversational
    ):
        mock_is_non_conversational.return_value = True
        mock_observer = MagicMock()
        mock_run_alert_bot.return_value = mock_observer

        transport = MagicMock()
        agent_config = {"interaction_mode": "non_conversational"}
        transcript = MagicMock()

        result = await run_bot(transport, agent_config, transcript)

        assert result == mock_observer
        mock_run_alert_bot.assert_called_once_with(
            transport,
            agent_config,
            transcript,
            audiobuffer=None,
            handle_sigint=False,
            sample_rate=8000,
            on_client_connected_hook=None,
        )

    @pytest.mark.asyncio
    @patch("api.bot.is_non_conversational")
    @patch("api.bot.create_llm_service")
    @patch("api.bot.create_stt_service")
    @patch("api.bot.create_tts_service")
    @patch("api.bot.Pipeline")
    @patch("api.bot.PipelineTask")
    @patch("api.bot.PipelineRunner")
    @patch("api.bot.CallMetricsObserver")
    async def test_run_bot_bhashini_kenpath(
        self,
        mock_observer,
        mock_runner,
        mock_task,
        mock_pipeline_class,
        mock_create_tts,
        mock_create_stt,
        mock_create_llm,
        mock_is_non_conversational,
    ):
        mock_is_non_conversational.return_value = False
        mock_llm = MagicMock()
        mock_llm.name = "kenpath"
        mock_llm.enable_bhashini_fast_turn = MagicMock()
        mock_create_llm.return_value = mock_llm

        mock_stt = MagicMock()
        mock_stt.name = "bhashini"
        mock_create_stt.return_value = mock_stt

        mock_tts = MagicMock()
        mock_tts.name = "mock_tts"
        mock_create_tts.return_value = mock_tts

        transport = MagicMock()
        agent_config = {
            "llm_model": {"name": "kenpath"},
            "stt_model": {"name": "bhashini"},
        }
        transcript = MagicMock()
        
        mock_runner.return_value.run = AsyncMock()

        await run_bot(transport, agent_config, transcript)
        mock_llm.enable_bhashini_fast_turn.assert_called_once()

    @pytest.mark.asyncio
    @patch("api.bot.is_non_conversational")
    @patch("api.bot.create_llm_service")
    @patch("api.bot.create_stt_service")
    @patch("api.bot.create_tts_service")
    @patch("api.bot.ensure_no_think_suffix")
    @patch("api.bot.Pipeline")
    @patch("api.bot.PipelineTask")
    @patch("api.bot.PipelineRunner")
    @patch("api.bot.CallMetricsObserver")
    async def test_run_bot_qwen_suffix(
        self,
        mock_observer,
        mock_runner,
        mock_task,
        mock_pipeline_class,
        mock_ensure_no_think,
        mock_create_tts,
        mock_create_stt,
        mock_create_llm,
        mock_is_non_conversational,
    ):
        mock_is_non_conversational.return_value = False
        mock_create_llm.return_value = MagicMock()
        mock_create_stt.return_value = MagicMock()
        mock_create_tts.return_value = MagicMock()
        mock_ensure_no_think.return_value = "cleaned system prompt"

        transport = MagicMock()
        agent_config = {
            "llm_model": {"name": "qwen"},
            "system_prompt": "think prompt",
        }
        transcript = MagicMock()
        
        mock_runner.return_value.run = AsyncMock()

        await run_bot(transport, agent_config, transcript)
        mock_ensure_no_think.assert_called_once_with("think prompt")

    @pytest.mark.asyncio
    @patch("api.bot.is_non_conversational")
    @patch("api.bot.create_llm_service")
    @patch("api.bot.create_stt_service")
    @patch("api.bot.create_tts_service")
    @patch("api.bot.Pipeline")
    @patch("api.bot.PipelineTask")
    @patch("api.bot.PipelineRunner")
    @patch("api.bot.CallMetricsObserver")
    async def test_run_bot_openai_llm_with_user_params(
        self,
        mock_observer,
        mock_runner,
        mock_task,
        mock_pipeline_class,
        mock_create_tts,
        mock_create_stt,
        mock_create_llm,
        mock_is_non_conversational,
    ):
        mock_is_non_conversational.return_value = False
        mock_llm = MagicMock()
        mock_llm._user_aggregator_params = {"param1": "val1"}
        mock_create_llm.return_value = mock_llm
        mock_create_stt.return_value = MagicMock()
        mock_create_tts.return_value = MagicMock()

        transport = MagicMock()
        agent_config = {"llm_model": {"name": "openai"}}
        transcript = MagicMock()
        
        mock_runner.return_value.run = AsyncMock()

        await run_bot(transport, agent_config, transcript)
        mock_llm.create_context_aggregator.assert_called_once_with(
            ANY, user_params={"param1": "val1"}
        )

    @pytest.mark.asyncio
    @patch("api.bot.create_llm_service")
    async def test_run_bot_service_creation_error(self, mock_create_llm):
        mock_create_llm.side_effect = ServiceCreationError("LLM failed")
        with pytest.raises(ServiceCreationError):
            await run_bot(MagicMock(), {}, MagicMock())

    @pytest.mark.asyncio
    @patch("api.bot.create_llm_service")
    async def test_run_bot_generic_exception(self, mock_create_llm):
        mock_create_llm.side_effect = ValueError("Some standard value error")
        mock_runner = AsyncMock()
        with patch("api.bot.PipelineRunner", return_value=mock_runner):
            mock_runner.run = AsyncMock()
            with pytest.raises(ValueError):
                await run_bot(MagicMock(), {}, MagicMock())


class TestRunBotConversational:
    """Focused tests for the conversational pipeline path in run_bot.

    Each test covers a single concern.  All shared mock infrastructure lives in
    the bot_mocks fixture and _run helper — no test method carries a @patch stack.

    Why no @patch decorators here:
        All 11 required patches are applied once inside _run via ExitStack.
        Mock objects retain their call history after the patches are removed,
        so every assertion made after _run() returns is still valid.
    """

    # Complete config so every utility helper (get_hold_messages, etc.) reads
    # from the dict without needing a patch.  No `interaction_mode` key means
    # is_non_conversational() returns False — the conversational path runs naturally.
    AGENT_CONFIG = {
        "llm_model": {"name": "openai", "args": {"model": "gpt-4"}},
        "stt_model": {"name": "deepgram"},
        "tts_model": {"name": "elevenlabs"},
        "greeting_message": "Hello there!",
        "org_id": "org_123",
        "language": "English",
        "hold_messages": ["Please hold."],
        "hold_message_timeout_seconds": 0.3,
        "interruption_min_words": 1,
        "ignore_user_speech_before_greeting": True,
        "user_online_detection_enabled": True,
        "user_online_detection_message": "Are you there?",
        "user_online_detection_seconds": 5.0,
        "user_silence_hangup_seconds": 10,
    }

    @pytest.fixture
    def bot_mocks(self):
        """All shared mock objects.  Tests pull what they need by key."""
        mock_llm = MagicMock()
        mock_llm.name = "mock_llm"
        mock_stt = MagicMock()
        mock_stt.name = "mock_stt"
        mock_tts = MagicMock()
        mock_tts.name = "mock_tts"

        mock_context_aggregator = MagicMock()
        mock_llm.create_context_aggregator.return_value = mock_context_aggregator

        mock_greeting_blocker = MagicMock()
        mock_greeting_completer = MagicMock()
        mock_goodbye = MagicMock()
        mock_online = MagicMock()
        mock_silence = MagicMock()

        mock_task = MagicMock()
        mock_task.queue_frames = AsyncMock()
        mock_task.cancel = AsyncMock()
        mock_task.stop_when_done = AsyncMock()

        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock()

        mock_observer = MagicMock()

        registered_handlers = {}
        mock_transport = MagicMock()
        mock_transport.input.return_value = MagicMock()
        mock_transport.output.return_value = MagicMock()

        def capture_handler(event_name):
            def decorator(fn):
                registered_handlers[event_name] = fn
                return fn
            return decorator

        mock_transport.event_handler.side_effect = capture_handler

        return {
            "mock_llm": mock_llm,
            "mock_stt": mock_stt,
            "mock_tts": mock_tts,
            "mock_context_aggregator": mock_context_aggregator,
            "mock_greeting_blocker": mock_greeting_blocker,
            "mock_greeting_completer": mock_greeting_completer,
            "mock_goodbye": mock_goodbye,
            "mock_online": mock_online,
            "mock_silence": mock_silence,
            "mock_task": mock_task,
            "mock_runner": mock_runner,
            "mock_observer": mock_observer,
            "mock_transport": mock_transport,
            "registered_handlers": registered_handlers,
        }

    async def _run(self, mocks, agent_config=None, **kwargs):
        """Run run_bot inside a fully mocked environment.

        Spy mock references (create_llm_service, Pipeline, PipelineTask, etc.)
        are written back into `mocks` so callers can assert call_args after
        this method returns — mock objects retain history after patches are removed.
        """
        kwargs.setdefault("sample_rate", 8000)
        with ExitStack() as stack:
            mocks["mock_create_llm"] = stack.enter_context(
                patch("api.bot.create_llm_service", return_value=mocks["mock_llm"])
            )
            mocks["mock_create_stt"] = stack.enter_context(
                patch("api.bot.create_stt_service", return_value=mocks["mock_stt"])
            )
            mocks["mock_create_tts"] = stack.enter_context(
                patch("api.bot.create_tts_service", return_value=mocks["mock_tts"])
            )
            stack.enter_context(patch(
                "api.bot.create_greeting_filters",
                return_value=(None, mocks["mock_greeting_blocker"], mocks["mock_greeting_completer"]),
            ))
            mocks["mock_goodbye_cls"] = stack.enter_context(
                patch("api.bot.GoodbyeHangupProcessor", return_value=mocks["mock_goodbye"])
            )
            stack.enter_context(patch("api.bot.UserOnlineDetectionFilter", return_value=mocks["mock_online"]))
            stack.enter_context(patch("api.bot.UserSilenceHangupProcessor", return_value=mocks["mock_silence"]))
            mocks["mock_pipeline_cls"] = stack.enter_context(patch("api.bot.Pipeline"))
            mocks["mock_task_cls"] = stack.enter_context(
                patch("api.bot.PipelineTask", return_value=mocks["mock_task"])
            )
            stack.enter_context(patch("api.bot.PipelineRunner", return_value=mocks["mock_runner"]))
            stack.enter_context(patch("api.bot.CallMetricsObserver", return_value=mocks["mock_observer"]))

            return await run_bot(
                mocks["mock_transport"],
                agent_config or self.AGENT_CONFIG,
                MagicMock(),
                **kwargs,
            )

    # ── Return value ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_returns_metrics_observer(self, bot_mocks):
        result = await self._run(bot_mocks)
        assert result == bot_mocks["mock_observer"]

    # ── Service factory call args ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_llm_called_with_merged_config(self, bot_mocks):
        """LLM receives knowledge_base fields merged in and language injected."""
        await self._run(bot_mocks)
        bot_mocks["mock_create_llm"].assert_called_once_with(
            {
                "name": "openai",
                "args": {"model": "gpt-4"},
                "knowledge_base_enabled": False,
                "knowledge_document_ids": [],
                "knowledge_top_k": 10,
            },
            vistaar_session_id=None,
            language="English",
            org_id="org_123",
            hold_messages=["Please hold."],
            hold_message_timeout_seconds=0.3,
        )

    @pytest.mark.asyncio
    async def test_stt_receives_language_and_sample_rate(self, bot_mocks):
        await self._run(bot_mocks, sample_rate=16000)
        bot_mocks["mock_create_stt"].assert_called_once_with(
            {"name": "deepgram", "language": "English"},
            16000,
            vad_analyzer=None,
            org_id="org_123",
        )

    @pytest.mark.asyncio
    async def test_tts_receives_language_and_sample_rate(self, bot_mocks):
        await self._run(bot_mocks, sample_rate=16000)
        bot_mocks["mock_create_tts"].assert_called_once_with(
            {"name": "elevenlabs", "language": "English"},
            16000,
            org_id="org_123",
        )

    # ── Pipeline composition ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_pipeline_contains_required_processors(self, bot_mocks):
        await self._run(bot_mocks)
        processors = bot_mocks["mock_pipeline_cls"].call_args[0][0]
        m = bot_mocks

        assert m["mock_transport"].input.return_value in processors
        assert m["mock_stt"] in processors
        assert m["mock_greeting_blocker"] in processors
        assert any(
            p.__class__.__name__ == "BargeInInterruptionProcessor" for p in processors
        ), "BargeInInterruptionProcessor missing from pipeline"
        assert m["mock_context_aggregator"].user.return_value in processors
        assert m["mock_llm"] in processors
        assert m["mock_goodbye"] in processors
        assert m["mock_tts"] in processors
        assert m["mock_greeting_completer"] in processors
        assert m["mock_online"] in processors
        assert m["mock_silence"] in processors
        assert m["mock_context_aggregator"].assistant.return_value in processors
        assert m["mock_transport"].output.return_value in processors

    @pytest.mark.asyncio
    async def test_pipeline_audio_processing_order(self, bot_mocks):
        """Audio must flow: transport-in → STT → LLM → TTS → transport-out."""
        await self._run(bot_mocks)
        processors = bot_mocks["mock_pipeline_cls"].call_args[0][0]
        m = bot_mocks
        idx = processors.index

        assert idx(m["mock_transport"].input.return_value) < idx(m["mock_stt"])
        assert idx(m["mock_stt"]) < idx(m["mock_llm"])
        assert idx(m["mock_llm"]) < idx(m["mock_tts"])
        assert idx(m["mock_tts"]) < idx(m["mock_transport"].output.return_value)

    # ── PipelineTask wiring ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_metrics_observer_registered_with_task(self, bot_mocks):
        await self._run(bot_mocks)
        _, task_kwargs = bot_mocks["mock_task_cls"].call_args
        assert task_kwargs["observers"] == [bot_mocks["mock_observer"]]

    # ── Event handlers ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_on_client_connected_queues_greeting_and_starts_recording(self, bot_mocks):
        audiobuffer = MagicMock()
        audiobuffer.start_recording = AsyncMock()
        on_hook = AsyncMock()

        await self._run(bot_mocks, audiobuffer=audiobuffer, on_client_connected_hook=on_hook)

        handler = bot_mocks["registered_handlers"]["on_client_connected"]
        await handler(bot_mocks["mock_transport"], MagicMock())

        audiobuffer.start_recording.assert_called_once()
        on_hook.assert_awaited_once()
        bot_mocks["mock_greeting_blocker"].start_greeting.assert_called_once()
        queued = bot_mocks["mock_task"].queue_frames.call_args[0][0]
        assert len(queued) == 1
        assert isinstance(queued[0], TTSSpeakFrame)
        assert queued[0].text == "Hello there!"

    @pytest.mark.asyncio
    async def test_on_client_disconnected_cancels_task(self, bot_mocks):
        await self._run(bot_mocks)
        handler = bot_mocks["registered_handlers"]["on_client_disconnected"]
        await handler(bot_mocks["mock_transport"], MagicMock())
        bot_mocks["mock_task"].cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_goodbye_callback_stops_pipeline_task(self, bot_mocks):
        """The callback wired into GoodbyeHangupProcessor must call task.stop_when_done."""
        await self._run(bot_mocks)
        schedule_call_end = bot_mocks["mock_goodbye_cls"].call_args[0][0]
        await schedule_call_end()
        bot_mocks["mock_task"].stop_when_done.assert_called_once()


class TestBotEntryPoint:
    @pytest.mark.asyncio
    @patch("api.bot.MinIOStorage")
    @patch("api.bot.parse_telephony_websocket", new_callable=AsyncMock)
    @patch("api.bot.VobizFrameSerializer")
    @patch("api.bot.SileroVADAnalyzer")
    @patch("api.bot.FastAPIWebsocketParams")
    @patch("api.bot.FastAPIWebsocketTransport")
    @patch("api.bot.patch_immediate_first_chunk")
    @patch("api.bot.run_bot")
    @patch("api.bot.submit_call_recording", new_callable=AsyncMock)
    async def test_bot_plivo_provider(
        self,
        mock_submit_recording,
        mock_run_bot,
        mock_patch_chunk,
        mock_transport_class,
        mock_params_class,
        mock_vad_class,
        mock_serializer_class,
        mock_parse_telephony,
        mock_storage_class,
    ):
        mock_storage = MagicMock()
        mock_storage.save_recording_bytes = AsyncMock()
        mock_storage.save_recording_from_chunks = AsyncMock()
        mock_storage.save_transcript_from_lines = AsyncMock()
        mock_storage_class.from_env.return_value = mock_storage
        mock_parse_telephony.return_value = (None, {"stream_id": "plivo_stream", "call_id": "plivo_call"})

        mock_parse_telephony.return_value = (
            None,
            {"stream_id": "plivo_stream", "call_id": "plivo_call"},
        )

        mock_serializer = MagicMock()
        mock_serializer_class.return_value = mock_serializer

        mock_vad = MagicMock()
        mock_vad_class.return_value = mock_vad

        mock_transport = MagicMock()
        mock_transport_class.return_value = mock_transport

        mock_observer = MagicMock()
        mock_observer.to_dict.return_value = {"latency": 0.3}
        
        async def mock_run_bot_side_effect(*args, **kwargs):
            hook = kwargs.get("on_client_connected_hook")
            if hook:
                await hook()
            return mock_observer
        mock_run_bot.side_effect = mock_run_bot_side_effect

        websocket_client = AsyncMock()
        agent_config = {
            "stt_model": {"name": "deepgram"},
            "org_id": "org_123",
        }

        result = await voice_bot(
            websocket_client=websocket_client,
            stream_sid=None,
            call_sid=None,
            agent_type="inbound",
            agent_config=agent_config,
            provider="plivo",
        )

        assert result == "plivo_call"
        websocket_client.accept.assert_called_once()
        mock_parse_telephony.assert_called_once_with(websocket_client)

        mock_serializer_class.assert_called_once()
        params_passed = mock_transport_class.call_args.kwargs["params"]
        mock_patch_chunk.assert_called_once_with(mock_transport)
        mock_run_bot.assert_awaited_once()
        mock_vad_class.assert_called_once()
        vad_params = mock_vad_class.call_args[1]["params"]
        assert vad_params.confidence == 0.3
        assert mock_vad._smoothing_factor == 0.1

        # Check call recording submission
        # plivo uses audiobuffer (not vobiz native); no audio chunks accumulated in mock,
        # so recording_url is None. latency from mock_observer.to_dict() = 0.3.
        mock_submit_recording.assert_called_once_with(
            call_sid="plivo_call",
            agent_type="inbound",
            agent_config=agent_config,
            storage=mock_storage,
            call_start_time=ANY,
            latency_metrics={"latency": 0.3},
            recording_url=None,
            omit_recording_url=False,
        )

    @pytest.mark.asyncio
    @patch("api.bot.MinIOStorage")
    @patch("api.bot.start_vobiz_call_recording", new_callable=AsyncMock)
    @patch("api.bot.wait_and_download_vobiz_recording", new_callable=AsyncMock)
    @patch("api.bot.VobizFrameSerializer")
    @patch("api.bot.SileroVADAnalyzer")
    @patch("api.bot.FastAPIWebsocketParams")
    @patch("api.bot.FastAPIWebsocketTransport")
    @patch("api.bot.patch_immediate_first_chunk")
    @patch("api.bot.run_bot")
    @patch("api.bot.submit_call_recording", new_callable=AsyncMock)
    async def test_bot_vobiz_provider_success(
        self,
        mock_submit_recording,
        mock_run_bot,
        mock_patch_chunk,
        mock_transport_class,
        mock_params_class,
        mock_vad_class,
        mock_serializer_class,
        mock_download_recording,
        mock_start_recording,
        mock_storage_class,
    ):
        mock_storage = MagicMock()
        mock_storage.save_recording_bytes = AsyncMock()
        mock_storage.save_recording_from_chunks = AsyncMock()
        mock_storage.save_transcript_from_lines = AsyncMock()
        mock_storage_class.from_env.return_value = mock_storage

        mock_serializer = MagicMock()
        mock_serializer_class.return_value = mock_serializer

        mock_vad = MagicMock()
        mock_vad_class.return_value = mock_vad

        mock_transport = MagicMock()
        mock_transport_class.return_value = mock_transport

        mock_observer = MagicMock()
        mock_observer.to_dict.return_value = {"latency": 0.2}
        
        async def mock_run_bot_side_effect(*args, **kwargs):
            hook = kwargs.get("on_client_connected_hook")
            if hook:
                await hook()
            return mock_observer
        mock_run_bot.side_effect = mock_run_bot_side_effect

        mock_start_recording.return_value = "vobiz_rec_999"
        mock_download_recording.return_value = b"raw_mp3_bytes"

        websocket_client = AsyncMock()
        agent_config = {
            "stt_model": {"name": "bhashini"},
            "org_id": "org_777",
            "call_timeout_seconds": 150,
        }

        # Run bot
        result = await voice_bot(
            websocket_client=websocket_client,
            stream_sid="stream_vobiz",
            call_sid="call_vobiz",
            agent_type="outbound",
            agent_config=agent_config,
            provider="vobiz",
        )

        assert result == "call_vobiz"

        mock_storage_class.from_env.assert_called_once()
        mock_vad_class.assert_called_once()
        mock_serializer_class.assert_called_once()
        mock_transport_class.assert_called_once()
        mock_patch_chunk.assert_called_once_with(mock_transport)
        mock_run_bot.assert_awaited_once()

        mock_start_recording.assert_called_once_with("call_vobiz", "org_777", 150)

        # Wait and download Vobiz recording must be triggered in finally
        mock_download_recording.assert_called_once_with("vobiz_rec_999", "org_777")
        mock_storage.save_recording_bytes.assert_called_once_with(
            "call_vobiz", b"raw_mp3_bytes", "mp3"
        )

        # submit_call_recording checks
        mock_submit_recording.assert_called_once_with(
            call_sid="call_vobiz",
            agent_type="outbound",
            agent_config=agent_config,
            storage=mock_storage,
            call_start_time=ANY,
            latency_metrics={"latency": 0.2},
            recording_url="minio://recordings/call_vobiz.mp3",
            omit_recording_url=False,
        )

    @pytest.mark.asyncio
    @patch("api.bot.MinIOStorage")
    @patch("api.bot.start_vobiz_call_recording", new_callable=AsyncMock)
    @patch("api.bot.wait_and_download_vobiz_recording", new_callable=AsyncMock)
    @patch("api.bot.VobizFrameSerializer")
    @patch("api.bot.SileroVADAnalyzer")
    @patch("api.bot.FastAPIWebsocketParams")
    @patch("api.bot.FastAPIWebsocketTransport")
    @patch("api.bot.patch_immediate_first_chunk")
    @patch("api.bot.run_bot")
    @patch("api.bot.submit_call_recording", new_callable=AsyncMock)
    async def test_bot_vobiz_recording_download_failed(
        self,
        mock_submit_recording,
        mock_run_bot,
        mock_patch_chunk,
        mock_transport_class,
        mock_params_class,
        mock_vad_class,
        mock_serializer_class,
        mock_download_recording,
        mock_start_recording,
        mock_storage_class,
    ):
        mock_storage = MagicMock()
        mock_storage.save_recording_bytes = AsyncMock()
        mock_storage.save_recording_from_chunks = AsyncMock()
        mock_storage.save_transcript_from_lines = AsyncMock()
        mock_storage_class.from_env.return_value = mock_storage
        mock_start_recording.return_value = "vobiz_rec_999"
        mock_download_recording.return_value = None  # Download failed

        # Fire the hook so recording_id is set; otherwise the finally block
        # skips the download entirely and the "failed download" path is never exercised.
        async def mock_run_bot_side_effect(*args, **kwargs):
            hook = kwargs.get("on_client_connected_hook")
            if hook:
                await hook()
            return MagicMock()

        mock_run_bot.side_effect = mock_run_bot_side_effect

        websocket_client = AsyncMock()
        agent_config = {
            "stt_model": {"name": "deepgram"},
            "org_id": "org_777",
        }

        result = await voice_bot(
            websocket_client=websocket_client,
            stream_sid="stream_vobiz",
            call_sid="call_vobiz",
            agent_type="outbound",
            agent_config=agent_config,
            provider="vobiz",
        )

        assert result == "call_vobiz"

        # Recording was started, download returned None → save_recording_bytes skipped.
        mock_start_recording.assert_called_once_with("call_vobiz", "org_777", ANY)
        mock_download_recording.assert_called_once_with("vobiz_rec_999", "org_777")
        mock_storage.save_recording_bytes.assert_not_called()
        mock_submit_recording.assert_called_once_with(
            call_sid="call_vobiz",
            agent_type="outbound",
            agent_config=agent_config,
            storage=mock_storage,
            call_start_time=ANY,
            latency_metrics=ANY,
            recording_url=None,
            omit_recording_url=True,
        )

    @pytest.mark.asyncio
    @patch("api.bot.MinIOStorage")
    @patch("api.bot.parse_telephony_websocket", new_callable=AsyncMock)
    @patch("api.bot.VobizFrameSerializer")
    @patch("api.bot.SileroVADAnalyzer")
    @patch("api.bot.FastAPIWebsocketParams")
    @patch("api.bot.FastAPIWebsocketTransport")
    @patch("api.bot.patch_immediate_first_chunk")
    @patch("api.bot.AudioBufferProcessor")
    @patch("api.bot.run_bot")
    @patch("api.bot.submit_call_recording", new_callable=AsyncMock)
    async def test_bot_non_vobiz_audiobuffer_success(
        self,
        mock_submit_recording,
        mock_run_bot,
        mock_audiobuffer_class,
        mock_patch_chunk,
        mock_transport_class,
        mock_params_class,
        mock_vad_class,
        mock_serializer_class,
        mock_parse_telephony,
        mock_storage_class,
    ):
        mock_storage = MagicMock()
        mock_storage.save_recording_bytes = AsyncMock()
        mock_storage.save_recording_from_chunks = AsyncMock()
        mock_storage.save_transcript_from_lines = AsyncMock()
        mock_storage_class.from_env.return_value = mock_storage
        mock_parse_telephony.return_value = (None, {"stream_id": "plivo_stream", "call_id": "plivo_call"})

        mock_audiobuffer = AsyncMock()
        mock_audiobuffer_class.return_value = mock_audiobuffer
        mock_audiobuffer.event_handler = MagicMock()

        # Setup audiobuffer event handler decorator capture
        audiobuffer_handlers = {}

        def mock_event_handler(event_name):
            def decorator(func):
                audiobuffer_handlers[event_name] = func
                return func

            return decorator

        mock_audiobuffer.event_handler.side_effect = mock_event_handler

        websocket_client = AsyncMock()
        agent_config = {
            "stt_model": {"name": "deepgram"},
            "org_id": "org_777",
        }

        # Fire audiobuffer handler during run_bot execution so the finally block
        # sees populated audio_chunks when it runs.
        async def mock_run_bot_with_audio(*args, **kwargs):
            handler = audiobuffer_handlers.get("on_audio_data")
            if handler:
                await handler(mock_audiobuffer, b"audio_chunk_1", 16000, 1)
                await handler(mock_audiobuffer, b"audio_chunk_2", 16000, 1)
            return MagicMock()

        mock_run_bot.side_effect = mock_run_bot_with_audio

        # Run bot
        await voice_bot(
            websocket_client=websocket_client,
            stream_sid="stream_plivo",
            call_sid="call_plivo",
            agent_type="inbound",
            agent_config=agent_config,
            provider="plivo",
        )

        assert "on_audio_data" in audiobuffer_handlers

        mock_storage.save_recording_from_chunks.assert_called_once_with(
            "call_plivo", [b"audio_chunk_1", b"audio_chunk_2"], 16000, 1
        )
        mock_submit_recording.assert_called_once_with(
            call_sid="call_plivo",
            agent_type="inbound",
            agent_config=agent_config,
            storage=mock_storage,
            call_start_time=ANY,
            latency_metrics=ANY,
            recording_url="minio://recordings/call_plivo.wav",
            omit_recording_url=False,
        )

    @pytest.mark.asyncio
    @patch("api.bot.MinIOStorage")
    @patch("api.bot.parse_telephony_websocket", new_callable=AsyncMock)
    @patch("api.bot.TranscriptProcessor")
    @patch("api.bot.VobizFrameSerializer")
    @patch("api.bot.SileroVADAnalyzer")
    @patch("api.bot.FastAPIWebsocketParams")
    @patch("api.bot.FastAPIWebsocketTransport")
    @patch("api.bot.patch_immediate_first_chunk")
    @patch("api.bot.run_bot")
    @patch("api.bot.submit_call_recording", new_callable=AsyncMock)
    async def test_bot_transcript_accumulation_and_callback(
        self,
        mock_submit_recording,
        mock_run_bot,
        mock_patch_chunk,
        mock_transport_class,
        mock_params_class,
        mock_vad_class,
        mock_serializer_class,
        mock_transcript_class,
        mock_parse_telephony,
        mock_storage_class,
    ):
        mock_storage = MagicMock()
        mock_storage.save_recording_bytes = AsyncMock()
        mock_storage.save_recording_from_chunks = AsyncMock()
        mock_storage.save_transcript_from_lines = AsyncMock()
        mock_storage_class.from_env.return_value = mock_storage
        mock_parse_telephony.return_value = (None, {"stream_id": "plivo_stream", "call_id": "plivo_call"})

        mock_transcript = MagicMock()
        mock_transcript_class.return_value = mock_transcript

        # Setup transcript event handler decorator capture
        transcript_handlers = {}

        def mock_event_handler(event_name):
            def decorator(func):
                transcript_handlers[event_name] = func
                return func

            return decorator

        mock_transcript.event_handler.side_effect = mock_event_handler

        websocket_client = AsyncMock()
        agent_config = {
            "stt_model": {"name": "deepgram"},
            "org_id": "org_777",
        }

        transcript_callback = AsyncMock()

        class MockMessage:
            def __init__(self, role, content, timestamp=None):
                self.role = role
                self.content = content
                self.timestamp = timestamp

        class MockTranscriptFrame:
            def __init__(self, messages):
                self.messages = messages

        messages = [
            MockMessage("user", "Hello", "10:00:01"),
            MockMessage("assistant", "Hi there", "10:00:02"),
        ]

        # Fire transcript handler during run_bot execution so the finally block
        # sees populated transcript_lines when it runs.
        async def mock_run_bot_with_transcript(*args, **kwargs):
            handler = transcript_handlers.get("on_transcript_update")
            if handler:
                await handler(mock_transcript, MockTranscriptFrame(messages))
            return MagicMock()

        mock_run_bot.side_effect = mock_run_bot_with_transcript

        # Run bot
        await voice_bot(
            websocket_client=websocket_client,
            stream_sid="stream_plivo",
            call_sid="call_plivo",
            agent_type="inbound",
            agent_config=agent_config,
            provider="plivo",
            transcript_callback=transcript_callback,
        )

        assert "on_transcript_update" in transcript_handlers

        # Check that callback was called for each message
        transcript_callback.assert_any_call("user", "Hello", "10:00:01")
        transcript_callback.assert_any_call("assistant", "Hi there", "10:00:02")

        # finally block saves accumulated transcript lines
        mock_storage.save_transcript_from_lines.assert_called_once_with(
            "call_plivo",
            ["[10:00:01] user: Hello", "[10:00:02] assistant: Hi there"],
        )

    @pytest.mark.asyncio
    @patch("api.bot.MinIOStorage")
    @patch("api.bot.parse_telephony_websocket", new_callable=AsyncMock)
    @patch("api.bot.VobizFrameSerializer")
    @patch("api.bot.SileroVADAnalyzer")
    @patch("api.bot.FastAPIWebsocketParams")
    @patch("api.bot.FastAPIWebsocketTransport")
    @patch("api.bot.patch_immediate_first_chunk")
    @patch("api.bot.run_bot")
    @patch("api.bot.submit_call_recording", new_callable=AsyncMock)
    async def test_bot_exception_in_run_bot_still_submits_recording(
        self,
        mock_submit_recording,
        mock_run_bot,
        mock_patch_chunk,
        mock_transport_class,
        mock_params_class,
        mock_vad_class,
        mock_serializer_class,
        mock_parse_telephony,
        mock_storage_class,
    ):
        mock_storage = MagicMock()
        mock_storage.save_recording_from_chunks = AsyncMock()
        mock_storage.save_transcript_from_lines = AsyncMock()
        mock_storage_class.from_env.return_value = mock_storage
        mock_parse_telephony.return_value = (None, {"stream_id": "s1", "call_id": "c1"})
        mock_run_bot.side_effect = RuntimeError("pipeline crashed")

        with pytest.raises(RuntimeError, match="pipeline crashed"):
            await voice_bot(
                websocket_client=AsyncMock(),
                stream_sid=None,
                call_sid=None,
                agent_type="inbound",
                agent_config={"stt_model": {"name": "deepgram"}, "org_id": "org1"},
                provider="plivo",
            )

        # The finally block in voice_bot must run even when run_bot raises.
        mock_submit_recording.assert_called_once()
