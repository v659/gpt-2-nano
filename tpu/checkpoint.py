"""Time-based checkpointing with auto-resume.

Wraps Orbax's `CheckpointManager` to:
  * save every N hours of wall-clock training (not every N steps),
  * keep the most recent K checkpoints and prune older ones,
  * silently restore the latest checkpoint when training restarts.

The save is asynchronous, so it doesn't block the training step.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp


class TimeCheckpointer:
    def __init__(
        self,
        ckpt_dir: str | Path,
        interval_hours: float,
        keep_n: int = 3,
    ):
        # Orbax requires an absolute path.
        self.dir = Path(ckpt_dir).expanduser().resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_hours * 3600.0
        self.keep_n = keep_n

        self._mngr = ocp.CheckpointManager(
            directory=str(self.dir),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=keep_n,
                create=True,
                # We decide when to save — Orbax doesn't second-guess us.
                save_interval_steps=1,
                enable_async_checkpointing=True,
            ),
        )
        self._last_save_time = time.time()

    # ── public API ──────────────────────────────────────────────────────────

    def latest_step(self) -> int | None:
        return self._mngr.latest_step()

    def maybe_save(self, step: int, payload: dict[str, Any], force: bool = False) -> bool:
        """Save if at least `interval_hours` have elapsed since the last save."""
        elapsed = time.time() - self._last_save_time
        if not force and elapsed < self.interval_seconds:
            return False
        self.save(step, payload)
        return True

    def save(self, step: int, payload: dict[str, Any]) -> None:
        self._mngr.save(step, args=ocp.args.StandardSave(payload))
        self._last_save_time = time.time()
        print(f"[ckpt] saved step={step} → {self.dir} (async)")

    def restore(self, step: int, abstract_payload: dict[str, Any]) -> dict[str, Any]:
        """Restore a specific step. `abstract_payload` is the structure to fill."""
        return self._mngr.restore(
            step,
            args=ocp.args.StandardRestore(abstract_payload),
        )

    def restore_latest(self, abstract_payload: dict[str, Any]) -> tuple[dict[str, Any], int] | None:
        step = self.latest_step()
        if step is None:
            return None
        print(f"[ckpt] resuming from step={step} in {self.dir}")
        return self.restore(step, abstract_payload), step

    def wait(self) -> None:
        """Block until the most recent async save has flushed to disk."""
        self._mngr.wait_until_finished()


def abstract_state(state) -> Any:
    """Build the shape/dtype-only tree that Orbax needs to know what to restore into."""
    return jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
