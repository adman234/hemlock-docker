"""Parent side of hemlock_worker.py: looks like an Engine (load, shade, unload) but lives in a subprocess."""
import json
import os
import socket
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path

from manager import Cancelled

ShadeResult = namedtuple("ShadeResult", "output concept target seconds")


class WorkerEngine:
    def __init__(self, config: dict):
        self.config = config
        self.proc = None
        self.f = None
        self.device = None

    # ------------------------------------------------------------------ lifecycle
    def load(self):
        script = str(Path(__file__).with_name("hemlock_worker.py"))
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        srv.settimeout(1.0)
        self.proc = subprocess.Popen([sys.executable, "-u", script, str(srv.getsockname()[1])], stdin=subprocess.DEVNULL,
                                     env=dict(os.environ, PYTHONUNBUFFERED="1"))
        conn, deadline = None, time.time() + 120
        try:
            while conn is None:
                if self.proc.poll() is not None:
                    raise RuntimeError("Hemlock engine process exited while starting. See the container log.")
                if time.time() > deadline:
                    raise RuntimeError("Hemlock engine process did not start in time.")
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    pass
        except Exception:
            self.unload()
            raise
        finally:
            srv.close()
        conn.settimeout(None)
        self.f = conn.makefile("rw", encoding="utf8", newline="\n")
        self._send(dict(self.config, cmd="init"))
        while True:
            ev = self._read()
            if ev["type"] == "ready":
                self.device = ev.get("device")
                return
            if ev["type"] == "error":
                self.unload()
                raise RuntimeError("Hemlock engine failed to start: " + ev["msg"])

    def unload(self):
        p, f, self.proc, self.f = self.proc, self.f, None, None
        if f is not None:
            try:
                f.write(json.dumps({"cmd": "quit"}) + "\n")
                f.flush()
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass
        if p is not None:
            try:
                p.wait(timeout=15)
            except Exception:
                p.kill()
                p.wait()

    # ------------------------------------------------------------------ work
    def shade(self, path, out_dir, params, tag=None, target=None, seed=None, on_stage=None, on_progress=None,
              reset_exif_orientation=True, write_json=False, description=None):
        """`description` is a template string such as '{tag}' (functions cannot cross the process boundary)."""
        self._send({"cmd": "job", "path": str(path), "out_dir": str(out_dir), "params": params.as_dict(), "tag": tag,
                    "target": target, "seed": seed, "reset_exif_orientation": reset_exif_orientation,
                    "description_template": description})
        cancelled = False
        while True:
            ev = self._read()
            t = ev["type"]
            try:
                if t == "stage" and on_stage and not cancelled:
                    on_stage(ev["stage"])
                elif t == "progress" and on_progress and not cancelled:
                    on_progress(ev["pct"])
            except Cancelled:
                cancelled = True
                self._send({"cmd": "cancel"})
            if t == "result":
                if cancelled:
                    raise Cancelled()
                return ShadeResult(ev["output"], ev["concept"], ev["target"], ev["seconds"])
            if t == "error":
                if ev.get("cancelled") or cancelled:
                    raise Cancelled()
                raise RuntimeError(ev["msg"])

    # ------------------------------------------------------------------ plumbing
    def _send(self, obj):
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()

    def _read(self):
        line = self.f.readline()
        if not line:
            code = self.proc.poll() if self.proc else None
            raise RuntimeError(f"Hemlock engine process ended unexpectedly (exit code {code}). See the container log.")
        return json.loads(line)