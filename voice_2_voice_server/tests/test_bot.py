import pytest
import json
import time
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
    @patch("api.bot.FastPunctuationAggregator")
    @patch("api.bot.OpenAILLMContext")
    @patch("api.bot.create_greeting_filters")
    @patch("api.bot.GoodbyeHangupProcessor")
    @patch("api.bot.UserOnlineDetectionFilter")
    @patch("api.bot.UserSilenceHangupProcessor")
    @patch("api.bot.Pipeline")
    @patch("api.bot.PipelineTask")
    @patch("api.bot.PipelineRunner")
    @patch("api.bot.CallMetricsObserver")
    @patch("api.bot.get_sample_rate")
    @patch("api.bot.get_ignore_user_speech_before_greeting")
    @patch("api.bot.get_interruption_min_words")
    @patch("api.bot.get_hold_messages")
    @patch("api.bot.get_hold_message_timeout_seconds")
    @patch("api.bot.get_user_online_detection_message")
    @patch("api.bot.get_user_online_detection_enabled")
    @patch("api.bot.get_user_online_detection_seconds")
    @patch("api.bot.get_user_silence_hangup_seconds")
    async def test_run_bot_conversational_happy_path(
        self,
        mock_get_silence_seconds,
        mock_get_online_seconds,
        mock_get_online_enabled,
        mock_get_online_msg,
        mock_get_hold_timeout,
        mock_get_hold_msgs,
        mock_get_min_words,
        mock_get_ignore_speech,
        mock_get_sample_rate,
        mock_metrics_observer_class,
        mock_runner_class,
        mock_task_class,
        mock_pipeline_class,
        mock_silence_processor_class,
        mock_online_filter_class,
        mock_goodbye_processor_class,
        mock_create_greeting_filters,
        mock_llm_context_class,
        mock_aggregator_class,
        mock_create_tts,
        mock_create_stt,
        mock_create_llm,
        mock_is_non_conversational,
    ):
        # 1. Setup config helpers
        mock_is_non_conversational.return_value = False
        mock_get_sample_rate.return_value = 16000
        mock_get_ignore_speech.return_value = True
        mock_get_min_words.return_value = 3
        mock_get_hold_msgs.return_value = ["Wait"]
        mock_get_hold_timeout.return_value = 0.5
        mock_get_online_msg.return_value = "Are you there?"
        mock_get_online_enabled.return_value = True
        mock_get_online_seconds.return_value = 5
        mock_get_silence_seconds.return_value = 10

        # 2. Setup mock services
        mock_llm = MagicMock()
        mock_llm.name = "mock_llm"
        mock_create_llm.return_value = mock_llm

        mock_stt = MagicMock()
        mock_stt.name = "mock_stt"
        mock_create_stt.return_value = mock_stt

        mock_tts = MagicMock()
        mock_tts.name = "mock_tts"
        mock_create_tts.return_value = mock_tts

        mock_context = MagicMock()
        mock_llm_context_class.return_value = mock_context

        mock_context_aggregator = MagicMock()
        mock_context_aggregator.user.return_value = "context_aggregator_user"
        mock_context_aggregator.assistant.return_value = "context_aggregator_assistant"
        mock_llm.create_context_aggregator.return_value = mock_context_aggregator

        mock_greeting_blocker = MagicMock()
        mock_greeting_completer = MagicMock()
        mock_create_greeting_filters.return_value = (
            None,
            mock_greeting_blocker,
            mock_greeting_completer,
        )

        mock_goodbye = MagicMock()
        mock_goodbye_processor_class.return_value = mock_goodbye

        mock_online = MagicMock()
        mock_online_filter_class.return_value = mock_online

        mock_silence = MagicMock()
        mock_silence_processor_class.return_value = mock_silence

        # 3. Setup transport
        mock_transport = MagicMock()
        mock_transport_input = MagicMock()
        mock_transport_output = MagicMock()
        mock_transport.input.return_value = mock_transport_input
        mock_transport.output.return_value = mock_transport_output

        # Setup event handler capture
        registered_handlers = {}

        def mock_event_handler(event_name):
            def decorator(func):
                registered_handlers[event_name] = func
                return func

            return decorator

        mock_transport.event_handler.side_effect = mock_event_handler

        # 4. Setup pipeline and task
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_task = MagicMock()
        mock_task.queue_frames = AsyncMock()
        mock_task.cancel = AsyncMock()
        mock_task.stop_when_done = AsyncMock()
        mock_task_class.return_value = mock_task

        # 5. Setup runner & observer
        mock_runner = AsyncMock()
        mock_runner.run = AsyncMock()
        mock_runner_class.return_value = mock_runner

        mock_observer = MagicMock()
        mock_metrics_observer_class.return_value = mock_observer

        # Setup audiobuffer & transcript
        mock_transcript = MagicMock()
        mock_transcript.user.return_value = "transcript_user"
        mock_transcript.assistant.return_value = "transcript_assistant"
        mock_audiobuffer = MagicMock()
        mock_audiobuffer.start_recording = AsyncMock()

        agent_config = {
            "llm_model": {"name": "openai", "args": {"model": "gpt-4"}},
            "stt_model": {"name": "deepgram"},
            "tts_model": {"name": "elevenlabs"},
            "greeting_message": "Hello there!",
            "org_id": "org_123",
            "language": "English",
        }

        on_connected_hook = AsyncMock()

        # Run
        result = await run_bot(
            mock_transport,
            agent_config,
            mock_transcript,
            audiobuffer=mock_audiobuffer,
            handle_sigint=True,
            on_client_connected_hook=on_connected_hook,
        )

        assert result == mock_observer
        mock_create_llm.assert_called_once_with(
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
            hold_messages=["Wait"],
            hold_message_timeout_seconds=0.5,
        )
        mock_create_stt.assert_called_once_with(
            {"name": "deepgram", "language": "English"},
            16000,
            vad_analyzer=None,
            org_id="org_123",
        )
        mock_create_tts.assert_called_once_with(
            {"name": "elevenlabs", "language": "English"}, 16000, org_id="org_123"
        )

        # Check Pipeline configuration — assert by membership, not position,
        # so that inserting/removing processors doesn't break these tests.
        processors = mock_pipeline_class.call_args[0][0]
        assert mock_transport_input in processors
        assert mock_stt in processors
        assert mock_greeting_blocker in processors
        assert any(
            p.__class__.__name__ == "BargeInInterruptionProcessor" for p in processors
        ), "Expected a BargeInInterruptionProcessor in the pipeline"
        assert "transcript_user" in processors
        assert "context_aggregator_user" in processors
        assert mock_llm in processors
        assert mock_goodbye in processors
        assert mock_tts in processors
        assert mock_greeting_completer in processors
        assert mock_online in processors
        assert mock_silence in processors
        assert "transcript_assistant" in processors
        assert mock_audiobuffer in processors
        assert mock_transport_output in processors
        assert "context_aggregator_assistant" in processors

        # Verify ordering of a few critical dependencies
        assert processors.index(mock_transport_input) < processors.index(mock_stt)
        assert processors.index(mock_stt) < processors.index(mock_llm)
        assert processors.index(mock_llm) < processors.index(mock_tts)
        assert processors.index(mock_tts) < processors.index(mock_transport_output)

        # Check task options
        mock_task_class.assert_called_once()
        task_args, task_kwargs = mock_task_class.call_args
        assert task_args[0] == mock_pipeline
        assert task_kwargs["observers"] == [mock_observer]

        # Verify handlers
        assert "on_client_connected" in registered_handlers
        assert "on_client_disconnected" in registered_handlers

        # Invoke connected handler
        await registered_handlers["on_client_connected"](mock_transport, MagicMock())
        mock_audiobuffer.start_recording.assert_called_once()
        on_connected_hook.assert_awaited_once()
        mock_greeting_blocker.start_greeting.assert_called_once()
        mock_task.queue_frames.assert_called_once()
        queued_args = mock_task.queue_frames.call_args[0][0]
        assert len(queued_args) == 1
        assert isinstance(queued_args[0], TTSSpeakFrame)
        assert queued_args[0].text == "Hello there!"

        # Invoke disconnected handler
        await registered_handlers["on_client_disconnected"](
            mock_transport, MagicMock()
        )
        mock_task.cancel.assert_called_once()

        # Invoke schedule_call_end
        # Find the goodbye processor's initialization callback
        goodbye_callback = mock_goodbye_processor_class.call_args[0][0]
        await goodbye_callback()
        mock_task.stop_when_done.assert_called_once()

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
        mock_wait_download_vobiz,
        mock_start_vobiz,
        mock_patch_chunk,
        mock_transport_class,
        mock_params_class,
        mock_vad_class,
        mock_serializer_class,
        mock_storage_class,
    ):
        mock_storage = MagicMock()
        mock_storage.save_recording_bytes = AsyncMock()
        mock_storage.save_recording_from_chunks = AsyncMock()
        mock_storage.save_transcript_from_lines = AsyncMock()
        mock_storage_class.from_env.return_value = mock_storage
        mock_start_vobiz.return_value = "vobiz_rec_999"
        mock_wait_download_vobiz.return_value = None  # Download failed

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

        # run_bot mock returned immediately without calling the hook, so
        # vobiz_recording_id stays None → no download attempt → recording_url=None.
        # In finally block, it skips download, but doesn't crash
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
