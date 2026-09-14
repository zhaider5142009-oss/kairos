"""
Watchdog supervisor — wraps the main jarvis process and restarts on crash.

Usage:
    python -m jarvis_predict.watchdog --child "python jarvis_main.py"

Restart policy:
- Exit code 0 → exit cleanly (no restart).
- Exit code != 0 → wait 5s, restart, up to N times per rolling hour.
- After N restarts within an hour → give up (supervisor exits with code 1).

This is intentionally simple. The deeper "never dies" guarantee comes from
the Windows Scheduled Task (always-on with restart-on-failure), not from
this in-process watchdog.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

LOGGER = logging.getLogger("jarvis.watchdog")


class Watchdog:
    def __init__(self, *, child_cmd: list[str], max_restarts_per_hour: int = 10,
                 restart_delay_seconds: float = 5.0) -> None:
        self.child_cmd = child_cmd
        self.max_restarts_per_hour = max(1, int(max_restarts_per_hour))
        self.restart_delay_seconds = max(0.5, float(restart_delay_seconds))
        self._restart_times: deque[float] = deque()

    def run(self) -> int:
        LOGGER.info("Watchdog starting child: %s", self.child_cmd)
        while True:
            t0 = time.time()
            try:
                rc = subprocess.call(self.child_cmd)
            except FileNotFoundError as e:
                LOGGER.error("Child command not found: %s", e)
                return 127
            except KeyboardInterrupt:
                LOGGER.info("Watchdog interrupted; forwarding to child next loop")
                continue
            elapsed = time.time() - t0

            if rc == 0:
                LOGGER.info("Child exited cleanly (rc=0) after %.1fs — watchdog exiting", elapsed)
                return 0

            self._restart_times.append(time.time())
            self._trim_old_restarts()
            if len(self._restart_times) > self.max_restarts_per_hour:
                LOGGER.error(
                    "Max restarts (%d/hour) exceeded; giving up.",
                    self.max_restarts_per_hour,
                )
                return 1

            LOGGER.warning(
                "Child exited with rc=%d after %.1fs; restarting in %.1fs (%d/%d in last hour)",
                rc, elapsed, self.restart_delay_seconds,
                len(self._restart_times), self.max_restarts_per_hour,
            )
            time.sleep(self.restart_delay_seconds)

    def _trim_old_restarts(self) -> None:
        cutoff = time.time() - 3600.0
        while self._restart_times and self._restart_times[0] < cutoff:
            self._restart_times.popleft()


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis watchdog supervisor.")
    parser.add_argument("--child", required=True, nargs=argparse.REMAINDER,
                        help="Command + args to run as the supervised child.")
    parser.add_argument("--max-restarts", type=int, default=10)
    parser.add_argument("--delay", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
    )
    return Watchdog(
        child_cmd=args.child,
        max_restarts_per_hour=args.max_restarts,
        restart_delay_seconds=args.delay,
    ).run()


if __name__ == "__main__":
    sys.exit(main() or 0)


__all__ = ["Watchdog"]
