"""Loads one method's models at a time and unloads them when idle.

- Only one method is on the GPU at once. Asking for a different method first unloads the current one.
- While a method stays in use its models stay loaded (jobs start immediately).
- After `idle_seconds` without a job everything is unloaded: VRAM and RAM go back to the system. The next job
  pays the load time again (about 10 s for Hemlock). idle_seconds == 0 unloads right after every job,
  idle_seconds < 0 never unloads.
- Mist runs as a subprocess that loads and frees its own models per job, so for Mist the manager only makes sure
  Hemlock's models are not holding VRAM while it runs.
"""
import gc
import threading
import time
from contextlib import contextmanager


class Cancelled(Exception):
    """The user pressed Stop."""


class ModelManager:
    def __init__(self, idle_seconds: float):
        self.idle_seconds = idle_seconds
        self._job_lock = threading.Lock()    # held for the whole job: one method on the GPU at a time
        self._state_lock = threading.Lock()  # held only for a moment, so the status display never waits for a job
        self._kind = None
        self._engine = None
        self._last_used = time.time()
        self._busy = False
        if idle_seconds >= 0:
            threading.Thread(target=self._watch, daemon=True, name="idle-unload").start()

    # ---------------------------------------------------------------- state for the UI
    def snapshot(self) -> dict:
        with self._state_lock:
            idle_left = None
            if self._kind and not self._busy and self.idle_seconds > 0:
                idle_left = max(0.0, self.idle_seconds - (time.time() - self._last_used))
            return {"kind": self._kind, "busy": self._busy, "idle_left": idle_left, "idle_seconds": self.idle_seconds}

    # ---------------------------------------------------------------- use
    @contextmanager
    def use(self, kind: str, factory=None, on_stage=None):
        """Yield the loaded engine for `kind` (None for subprocess based methods)."""
        with self._job_lock:
            with self._state_lock:
                self._busy = True
            try:
                if self._kind != kind:
                    self._unload()
                if kind != "mist" and self._engine is None:
                    if on_stage:
                        on_stage("Loading models")
                    eng = factory()
                    eng.load()
                    self._engine = eng
                with self._state_lock:
                    self._kind = kind
                yield self._engine
            finally:
                with self._state_lock:
                    self._last_used = time.time()
                    self._busy = False
                if self.idle_seconds == 0:
                    self._unload()

    def unload(self) -> bool:
        """Unload now if no job is running."""
        if self._job_lock.acquire(blocking=False):
            try:
                self._unload()
                return True
            finally:
                self._job_lock.release()
        return False

    # ---------------------------------------------------------------- internals
    def _unload(self):
        eng, self._engine = self._engine, None
        with self._state_lock:
            self._kind = None
        if eng is not None:
            try:
                eng.unload()
            except Exception as e:  # never let cleanup break the next job
                print(f"unload failed: {e!r}", flush=True)
        del eng
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _watch(self):
        while True:
            time.sleep(10)
            snap = self.snapshot()
            if snap["kind"] and not snap["busy"] and snap["idle_left"] == 0:
                if self._job_lock.acquire(blocking=False):
                    try:
                        s = self.snapshot()
                        if s["kind"] and not s["busy"] and s["idle_left"] == 0:
                            print(f"idle for {self.idle_seconds / 60:.1f} min, unloading {s['kind']}", flush=True)
                            self._unload()
                    finally:
                        self._job_lock.release()
