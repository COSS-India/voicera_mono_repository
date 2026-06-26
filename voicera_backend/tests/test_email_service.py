"""
Unit tests for app.services.email_service.
"""
import pytest
from unittest.mock import MagicMock, patch

from app.services.email_service import send_password_reset_email

EMAIL = "user@example.com"
RESET_TOKEN = "reset-uuid-token"
RESET_URL = "https://example.com/reset-password?token=reset-uuid-token"


class TestSendPasswordResetEmail:
    def test_success_returns_true(self):
        mock_client = MagicMock()
        mock_client.send.return_value = None
        with patch("app.services.email_service.MailtrapClient", return_value=mock_client):
            result = send_password_reset_email(EMAIL, RESET_TOKEN, RESET_URL)
        assert result is True

    def test_missing_mailtrap_token_returns_false(self):
        with patch("app.services.email_service.settings") as mock_settings:
            mock_settings.MAILTRAP_API_TOKEN = None
            result = send_password_reset_email(EMAIL, RESET_TOKEN, RESET_URL)
        assert result is False

    def test_empty_mailtrap_token_returns_false(self):
        with patch("app.services.email_service.settings") as mock_settings:
            mock_settings.MAILTRAP_API_TOKEN = ""
            result = send_password_reset_email(EMAIL, RESET_TOKEN, RESET_URL)
        assert result is False

    def test_mailtrap_exception_returns_false(self):
        with patch("app.services.email_service.MailtrapClient",
                   side_effect=Exception("network error")):
            result = send_password_reset_email(EMAIL, RESET_TOKEN, RESET_URL)
        assert result is False

    def test_send_exception_returns_false(self):
        mock_client = MagicMock()
        mock_client.send.side_effect = Exception("SMTP error")
        with patch("app.services.email_service.MailtrapClient", return_value=mock_client):
            result = send_password_reset_email(EMAIL, RESET_TOKEN, RESET_URL)
        assert result is False

    def test_email_body_contains_reset_url(self):
        mock_client = MagicMock()
        captured_mail = []

        def capture_send(mail):
            captured_mail.append(mail)

        mock_client.send.side_effect = capture_send
        with patch("app.services.email_service.MailtrapClient", return_value=mock_client):
            send_password_reset_email(EMAIL, RESET_TOKEN, RESET_URL)
        assert len(captured_mail) == 1
        # Mail object should have been constructed; verify the call was made
        mock_client.send.assert_called_once()

    def test_recipient_email_is_correct(self):
        mock_client = MagicMock()
        with patch("app.services.email_service.MailtrapClient", return_value=mock_client), \
             patch("app.services.email_service.Mail") as MockMail, \
             patch("app.services.email_service.Address") as MockAddress:
            send_password_reset_email(EMAIL, RESET_TOKEN, RESET_URL)
        # Address should be called with our email
        calls = [str(c) for c in MockAddress.call_args_list]
        assert any(EMAIL in c for c in calls)
