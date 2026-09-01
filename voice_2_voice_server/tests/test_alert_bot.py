import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pipecat.frames.frames import TTSSpeakFrame

from api.services import ServiceCreationError
from utils.pipelines.alert_bot import run_alert_bot

class TestRunAlertBot:
    @pytest.mark.asyncio
    @patch("utils.pipelines.alert_bot.get_sample_rate")
    @patch("utils.pipelines.alert_bot.create_tts_service")
    @patch("utils.pipelines.alert_bot.FastPunctuationAggregator")
    @patch("utils.pipelines.alert_bot.AlertHangupProcessor")
    @patch("utils.pipelines.alert_bot.CallMetricsObserver")
    @patch("utils.pipelines.alert_bot.Pipeline")
    @patch("utils.pipelines.alert_bot.PipelineTask")
    @patch("utils.pipelines.alert_bot.PipelineRunner")
    async def test_run_alert_bot_happy_path(
        self,
        mock_runner_class,
        mock_task_class,
        mock_pipeline_class,
        mock_observer_class,
        mock_hangup_class,
        mock_aggregator_class,
        mock_create_tts,
        mock_get_sample_rate,
    ):
        # 1. Setup sample rate and tts service mock
        mock_get_sample_rate.return_value = 8000
        mock_tts = MagicMock()
        mock_tts.name = "my_tts_service"
        mock_create_tts.return_value = mock_tts

        # 2. Setup transport mock
        mock_transport = MagicMock()
        mock_transport_input = MagicMock()
        mock_transport_output = MagicMock()
        mock_transport.input.return_value = mock_transport_input
        mock_transport.output.return_value = mock_transport_output

        # Setup event handler decorator capture
        registered_handlers = {}
        def mock_event_handler(event_name):
            def decorator(func):
                registered_handlers[event_name] = func
                return func
            return decorator
        mock_transport.event_handler.side_effect = mock_event_handler

        # 3. Setup transcript & audiobuffer mocks
        mock_transcript = MagicMock()
        mock_transcript_assistant = MagicMock()
        mock_transcript.assistant.return_value = mock_transcript_assistant
        mock_audiobuffer = AsyncMock()

        # 4. Setup pipeline and task mocks
        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_task = MagicMock()
        mock_task.queue_frames = AsyncMock()
        mock_task.stop_when_done = AsyncMock()
        mock_task.cancel = AsyncMock()
        mock_task_class.return_value = mock_task

        # 5. Setup runner mock
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock()
        mock_runner_class.return_value = mock_runner

        # 6. Setup observer and hangup mocks
        mock_observer = MagicMock()
        mock_observer_class.return_value = mock_observer
        mock_hangup = MagicMock()
        mock_hangup_class.return_value = mock_hangup

        # 7. Inputs
        agent_config = {
            "tts_model": {"voice_id": "v_123"},
            "language": "Hindi",
            "greeting_message": "Welcome to Voicera!",
            "org_id": "org_789"
        }
        on_connected_hook_called = False
        async def mock_on_connected_hook():
            nonlocal on_connected_hook_called
            on_connected_hook_called = True

        # Run the function
        result_observer = await run_alert_bot(
            transport=mock_transport,
            agent_config=agent_config,
            transcript=mock_transcript,
            audiobuffer=mock_audiobuffer,
            handle_sigint=True,
            sample_rate=16000,
            on_client_connected_hook=mock_on_connected_hook
        )

        # Asserts
        assert result_observer == mock_observer
        mock_get_sample_rate.assert_not_called()  # since sample_rate was passed
        mock_create_tts.assert_called_once_with(
            {"voice_id": "v_123", "language": "Hindi"},
            16000,
            org_id="org_789"
        )
        assert mock_tts._aggregate_sentences is True
        assert mock_tts._text_aggregator is mock_aggregator_class.return_value

        # Check processors passed to Pipeline
        mock_pipeline_class.assert_called_once_with([
            mock_transport_input,
            mock_tts,
            mock_hangup,
            mock_transcript_assistant,
            mock_audiobuffer,
            mock_transport_output
        ])

        # Check task settings
        mock_task_class.assert_called_once()
        args, kwargs = mock_task_class.call_args
        assert args[0] == mock_pipeline
        assert kwargs["observers"] == [mock_observer]
        assert kwargs["params"].allow_interruptions is False
        assert kwargs["params"].enable_metrics is True

        # Check runner execution
        mock_runner_class.assert_called_once_with(handle_sigint=True)
        mock_runner.run.assert_called_once_with(mock_task)

        # Verify handlers were registered
        assert "on_client_connected" in registered_handlers
        assert "on_client_disconnected" in registered_handlers

        # Invoke on_client_connected handler
        await registered_handlers["on_client_connected"](mock_transport, MagicMock())
        mock_audiobuffer.start_recording.assert_called_once()
        assert on_connected_hook_called is True
        mock_task.queue_frames.assert_called_once()
        queued_args = mock_task.queue_frames.call_args[0][0]
        assert len(queued_args) == 1
        assert isinstance(queued_args[0], TTSSpeakFrame)
        assert queued_args[0].text == "Welcome to Voicera!"

        # Invoke on_client_disconnected handler
        await registered_handlers["on_client_disconnected"](mock_transport, MagicMock())
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    @patch("utils.pipelines.alert_bot.get_sample_rate")
    @patch("utils.pipelines.alert_bot.create_tts_service")
    @patch("utils.pipelines.alert_bot.FastPunctuationAggregator")
    @patch("utils.pipelines.alert_bot.AlertHangupProcessor")
    @patch("utils.pipelines.alert_bot.CallMetricsObserver")
    @patch("utils.pipelines.alert_bot.Pipeline")
    @patch("utils.pipelines.alert_bot.PipelineTask")
    @patch("utils.pipelines.alert_bot.PipelineRunner")
    async def test_run_alert_bot_no_audiobuffer_and_no_greeting(
        self,
        mock_runner_class,
        mock_task_class,
        mock_pipeline_class,
        mock_observer_class,
        mock_hangup_class,
        mock_aggregator_class,
        mock_create_tts,
        mock_get_sample_rate,
    ):
        mock_get_sample_rate.return_value = 8000
        mock_tts = MagicMock()
        mock_tts.name = "my_tts"
        mock_create_tts.return_value = mock_tts

        mock_transport = MagicMock()
        registered_handlers = {}
        def mock_event_handler(event_name):
            def decorator(func):
                registered_handlers[event_name] = func
                return func
            return decorator
        mock_transport.event_handler.side_effect = mock_event_handler

        mock_task = MagicMock()
        mock_task.stop_when_done = AsyncMock()
        mock_task_class.return_value = mock_task

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock()
        mock_runner_class.return_value = mock_runner

        # Run with no audiobuffer and no greeting_message
        agent_config = {"tts_model": {}, "greeting_message": ""}
        await run_alert_bot(
            transport=mock_transport,
            agent_config=agent_config,
            transcript=MagicMock(),
            audiobuffer=None
        )

        mock_get_sample_rate.assert_called_once()

        # Check processors passed to Pipeline (does not contain audiobuffer)
        processors_called = mock_pipeline_class.call_args[0][0]
        assert None not in processors_called

        # Invoke on_client_connected handler
        await registered_handlers["on_client_connected"](mock_transport, MagicMock())
        # Since greeting message is missing, schedule_call_end should be triggered
        mock_task.stop_when_done.assert_called_once()

    @pytest.mark.asyncio
    @patch("utils.pipelines.alert_bot.create_tts_service")
    async def test_run_alert_bot_service_creation_error(self, mock_create_tts):
        mock_create_tts.side_effect = ServiceCreationError("Failed to create TTS")
        with pytest.raises(ServiceCreationError):
            await run_alert_bot(
                transport=MagicMock(),
                agent_config={},
                transcript=MagicMock()
            )

    @pytest.mark.asyncio
    @patch("utils.pipelines.alert_bot.create_tts_service")
    async def test_run_alert_bot_generic_exception(self, mock_create_tts):
        mock_create_tts.side_effect = ValueError("Generic error")
        with pytest.raises(ValueError):
            await run_alert_bot(
                transport=MagicMock(),
                agent_config={},
                transcript=MagicMock()
            )
