"""Process-wide budget tracker for the web UI.

The CLI's per-suite cap doesn't help a long-running server: a client
that can hit /api/ask repeatedly can replay until the OpenAI bill is
gone. This module is a single global counter the web app checks before
issuing a cycle and updates after.

In-process memory only — no Redis, no sqlite. A restart resets the
counter; that's fine for local single-user use where the operator
owns both the API key and the server.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class BudgetState:
    cap_usd: float
    spent_usd: float = 0.0

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    @property
    def is_exhausted(self) -> bool:
        return self.spent_usd >= self.cap_usd


class Budget:
    """Thread-safe budget counter. The lock matters because FastAPI
    serves requests on a thread pool — without it, two concurrent
    /api/ask hits could both pass the `is_exhausted` check, blow past
    the cap, and double-bill."""

    def __init__(self, cap_usd: float):
        self._state = BudgetState(cap_usd=cap_usd)
        self._lock = threading.Lock()

    def state(self) -> BudgetState:
        """Snapshot. Not held under the lock so callers shouldn't
        mutate the returned dataclass; treat it as a value object."""
        with self._lock:
            return BudgetState(
                cap_usd=self._state.cap_usd,
                spent_usd=self._state.spent_usd,
            )

    # Provisional hold per request. Sized as a generous over-estimate
    # of one cycle's worst-case spend on gpt-5.2 standard tier with
    # freeform tool returns; settle() subtracts the hold and adds the
    # real spend, so the steady-state error is zero.
    DEFAULT_HOLD_USD = 0.05

    def reserve(self, hold_usd: float = DEFAULT_HOLD_USD) -> bool:
        """Provisionally deduct an estimate of one cycle's max spend
        and return True if the cap still has room; return False
        without mutating if the cap would be exceeded.

        Calling reserve() then settle() correctly handles concurrent
        requests: two simultaneous /api/ask hits each take a hold, so
        the second one will see the budget already deducted and may
        be denied. Without the hold, both would pass the cap check
        and race past it — the TOCTOU window that the per-suite CLI
        check in cli.py cmd_run_suite has but the daemon cannot
        tolerate."""
        with self._lock:
            if self._state.spent_usd + hold_usd > self._state.cap_usd:
                return False
            self._state.spent_usd += hold_usd
            return True

    def settle(self, actual_spend_usd: float,
               hold_usd: float = DEFAULT_HOLD_USD) -> None:
        """Refund the provisional hold and add the actual cycle spend.
        Always call this after reserve(), even on cycle failure — pass
        whatever spend was incurred (which may be 0.0)."""
        with self._lock:
            self._state.spent_usd -= hold_usd
            self._state.spent_usd += actual_spend_usd
            # Defensive: never go negative if the caller passed wrong values
            self._state.spent_usd = max(0.0, self._state.spent_usd)
