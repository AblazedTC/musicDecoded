"""Simple logging utilities."""


def log_info(message: str):
    """Log an info message."""
    print(f"[INFO] {message}")


def log_error(message: str):
    """Log an error message."""
    print(f"[ERROR] {message}")


def log_debug(message: str):
    """Log a debug message."""
    print(f"[DEBUG] {message}")
