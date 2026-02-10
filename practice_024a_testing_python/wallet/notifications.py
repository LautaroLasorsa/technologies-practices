"""Notification service — external dependency for testing with mocks."""

from typing import Protocol


class NotificationService(Protocol):
    def send(self, recipient: str, subject: str, body: str) -> bool:
        """Send a notification. Returns True if sent successfully."""
        ...


class EmailNotificationService:
    """Production implementation — sends real emails (not used in tests)."""

    def send(self, recipient: str, subject: str, body: str) -> bool:
        # In production, this would send an actual email
        raise NotImplementedError("Use a mock in tests")


class ConsoleNotificationService:
    """Development implementation — prints to console."""

    def send(self, recipient: str, subject: str, body: str) -> bool:
        print(f"[NOTIFICATION] To: {recipient} | Subject: {subject} | Body: {body}")
        return True
