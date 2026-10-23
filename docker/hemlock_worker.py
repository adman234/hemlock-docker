"""Hemlock engine worker: runs in its own process so that stopping it gives ALL memory back.

A long-lived process that loads the models, then shades images on request. The web app starts it for the first
Hemlock job and stops it after the idle timeout; ending the process is the only reliable way to return RAM, VRAM
and the CUDA context to the system (unloading inside one process leaves gigabytes of allocator memory behind).

Usage: python hemlock_worker.py PORT. It connects to 127.0.0.1:PORT (the web app is listening) and speaks JSON lines:
  from the app: {"cmd":"init",...} once, then {"cmd":"job",...}, {"cmd":"cancel"}, {"cmd":"quit"}
  to the app:   {"type":"ready"|"stage"|"progress"|"result"|"error", ...}
A socket is used instead of stdin/stdout because on Windows a blocked pipe read stalls other calls on that handle.
"""
import json
import queue
import socket
import sys
import threading


class Cancelled(Exception):
    pass


def main():
    sock = socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=30)
    sock.settimeout(None)
    f = sock.makefile("rw", encoding="utf8", newline="\n")
    send_lock = threading.Lock()
    cancel = threading.Event()
    inbox = queue.Queue()

    def send(obj):
        with send_lock:
            f.write(json.dumps(obj) + "\n")
            f.flush()

    def reader():
        for line in f:
            if not line.strip():
                continue
            msg = json.loads(line)
            if msg.get("cmd") == "cancel":
                cancel.set()
            else:
                inbox.put(msg)
        inbox.put({"cmd": "quit"})  # the app went away

    threading.Thread(target=reader, daemon=True).start()
    from nsre.engine import Engine
    from nsre.params import Params

    eng = None
    while True:
        msg = inbox.get()
        c = msg.get("cmd")
        if c == "quit":
            return
        try:
            if c == "init":
                eng = Engine(
                    msg["sd_dir"], blip_dir=msg.get("blip_dir"), clip_dir=msg.get("clip_dir"),
                    use_concept_models=msg["use_concept_models"], lpips_pickle=msg.get("lpips_pickle"),
                    pool=msg.get("pool"), mapping=msg.get("mapping"),
                )
                eng.load()
                send({"type": "ready", "device": eng.device})
            elif c == "job":
                cancel.clear()

                def check():
                    if cancel.is_set():
                        raise Cancelled()

                def on_stage(s):
                    check()
                    send({"type": "stage", "stage": s})

                def on_progress(p):
                    check()
                    send({"type": "progress", "pct": p})

                tpl = msg.get("description_template")
                desc = None
                if tpl:
                    try:
                        tpl.format(tag="x")
                        desc = lambda t: tpl.format(tag=t)  # noqa: E731
                    except Exception:
                        desc = lambda t: t  # noqa: E731  a broken template falls back to the plain tag
                r = eng.shade(
                    msg["path"], msg["out_dir"], Params(**msg["params"]), tag=msg.get("tag"), target=msg.get("target"),
                    seed=msg.get("seed"), on_stage=on_stage, on_progress=on_progress,
                    reset_exif_orientation=msg.get("reset_exif_orientation", True), write_json=False, description=desc,
                )
                send({"type": "result", "output": r.output, "concept": r.concept, "target": r.target, "seconds": r.seconds})
        except Cancelled:
            send({"type": "error", "msg": "cancelled", "cancelled": True})
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            send({"type": "error", "msg": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()