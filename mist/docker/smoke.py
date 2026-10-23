"""Build-time check: start the app without the model download and make sure the page renders."""
import os
import subprocess
import sys
import time
import urllib.request

env = dict(os.environ, MIST_SKIP_MODEL="1", DATA_DIR="/tmp/smoke", MIST_PORT="7861")
proc = subprocess.Popen([sys.executable, "/app/app.py"], env=env)
try:
    for _ in range(60):
        time.sleep(1)
        if proc.poll() is not None:
            sys.exit("app exited during startup")
        try:
            r = urllib.request.urlopen("http://127.0.0.1:7861/", timeout=5)
        except Exception:
            continue
        assert r.status == 200 and b"Mist" in r.read(), "unexpected page"
        print("page ok")
        break
    else:
        sys.exit("page never came up")
finally:
    proc.terminate()
