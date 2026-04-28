from __future__ import annotations

import os
import resend

resend.api_key = os.getenv("RESEND_API_KEY", "")

_FROM = "FocusBuddy <noreply@thefocusbuddy.org>"


def _send(to: str, subject: str, html: str) -> None:
    if not resend.api_key:
        return
    resend.Emails.send({"from": _FROM, "to": [to], "subject": subject, "html": html})


def send_verification_email(to: str, username: str, verify_url: str) -> None:
    html = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto">
      <h2>Welcome to FocusBuddy, {username}!</h2>
      <p>Click the button below to verify your email address.</p>
      <a href="{verify_url}"
         style="display:inline-block;padding:12px 24px;background:#38bdf8;color:#0f172a;
                border-radius:8px;text-decoration:none;font-weight:600;">
        Verify Email
      </a>
      <p style="color:#64748b;font-size:.85rem;margin-top:1.5rem;">
        This link expires in 24 hours. If you didn't create an account, ignore this email.
      </p>
    </div>
    """
    _send(to, "Verify your FocusBuddy account", html)


def send_password_reset_email(to: str, username: str, reset_url: str) -> None:
    html = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto">
      <h2>Reset your password</h2>
      <p>Hi {username}, click the button below to choose a new password.</p>
      <a href="{reset_url}"
         style="display:inline-block;padding:12px 24px;background:#38bdf8;color:#0f172a;
                border-radius:8px;text-decoration:none;font-weight:600;">
        Reset Password
      </a>
      <p style="color:#64748b;font-size:.85rem;margin-top:1.5rem;">
        This link expires in 1 hour. If you didn't request this, ignore this email.
      </p>
    </div>
    """
    _send(to, "Reset your FocusBuddy password", html)
