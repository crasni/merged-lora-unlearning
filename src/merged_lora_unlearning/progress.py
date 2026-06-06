from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


def info(message: str) -> None:
    print(f"[mlu] {message}", flush=True)


@contextmanager
def stage(name: str, details: str | None = None) -> Iterator[None]:
    suffix = f" | {details}" if details else ""
    info(f"START {name}{suffix}")
    started = time.monotonic()
    try:
        yield
    except Exception:
        info(f"FAIL  {name} | elapsed={_duration(time.monotonic() - started)}")
        raise
    info(f"DONE  {name} | elapsed={_duration(time.monotonic() - started)}")


def metric_summary(name: str, metrics: dict[str, float], keys: tuple[str, ...]) -> None:
    values = " ".join(f"{key}={metrics[key]:.4f}" for key in keys)
    info(f"{name} | {values}")


def _duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"

