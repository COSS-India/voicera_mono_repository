"""
Email service for sending emails using Mailtrap.
"""
from mailtrap import Mail, Address, MailtrapClient
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def send_password_reset_email(email: str, reset_token: str, reset_url: str) -> bool:
    """
    Send password reset email using Mailtrap.
    
    Args:
        email: Recipient email address
        reset_token: Password reset token
        reset_url: Full reset password URL
        
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        if not settings.MAILTRAP_API_TOKEN:
            logger.error("MAILTRAP_API_TOKEN not configured")
            return False
        
        mail = Mail(
            sender=Address(email=settings.MAILTRAP_FROM_EMAIL, name=settings.MAILTRAP_FROM_NAME),
            to=[Address(email=email)],
            subject="Reset Your Password - Voicera",
            text=f"""
Hello,

You requested to reset your password for your Voicera account.

Click the link below to reset your password:
{reset_url}

This link will expire in 1 hour.

If you didn't request this password reset, please ignore this email.

Best regards,
Voicera Team
            """,
            html=f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .button {{ display: inline-block; padding: 12px 24px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
        .button:hover {{ background-color: #0056b3; }}
        .footer {{ margin-top: 30px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Reset Your Password</h2>
        <p>Hello,</p>
        <p>You requested to reset your password for your Voicera account.</p>
        <p>Click the button below to reset your password:</p>
        <a href="{reset_url}" class="button">Reset Password</a>
        <p>Or copy and paste this link into your browser:</p>
        <p style="word-break: break-all; color: #007bff;">{reset_url}</p>
        <p>This link will expire in 1 hour.</p>
        <p>If you didn't request this password reset, please ignore this email.</p>
        <div class="footer">
            <p>Best regards,<br>Voicera Team</p>
        </div>
    </div>
</body>
</html>
            """
        )
        
        # Send email using Mailtrap
        client = MailtrapClient(token=settings.MAILTRAP_API_TOKEN)
        client.send(mail)
        
        logger.info(f"Password reset email sent to: {email}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending password reset email: {str(e)}")
        return False

