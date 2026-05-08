"""Tiny console progress helpers for local processing."""
import sys
import time
from typing import Optional


class Progress:
    def __init__(self, label: str, total: float, width: int = 28) -> None:
        self.label = label
        self.total = max(float(total), 0.0)
        self.width = width
        self.started = time.time()
        self.last_draw = 0.0
        self.last_current = 0.0

    def update(self, current: float, detail: str = "", force: bool = False) -> None:
        now = time.time()
        current = max(0.0, min(float(current), self.total if self.total else float(current)))
        if not force and now - self.last_draw < 3.0 and current < self.total:
            return

        self.last_draw = now
        self.last_current = current
        pct = 100.0 if self.total <= 0 else min(100.0, current / self.total * 100.0)
        filled = int(self.width * pct / 100.0)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = int(now - self.started)
        suffix = f" {detail}" if detail else ""
        sys.stdout.write(f"[{self.label}] [{bar}] {pct:5.1f}% {elapsed:4d}s{suffix}\n")
        sys.stdout.flush()

    def done(self, detail: str = "") -> None:
        self.update(self.total, detail=detail, force=True)
        sys.stdout.flush()


def stage(message: str, detail: Optional[str] = None) -> None:
    if detail:
        print(f"\n=== {message}: {detail} ===", flush=True)
    else:
        print(f"\n=== {message} ===", flush=True)


def user_log(message: str, detail: Optional[str] = None) -> None:
    if detail:
        print(f"[studio] {message}: {detail}", flush=True)
    else:
        print(f"[studio] {message}", flush=True)
