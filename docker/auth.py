"""Optional password protection with a built-in limit on wrong guesses. Password only, there is no username.

Active when HEMLOCK_PASSWORD is set. Everything except the login page needs a valid cookie. After
HEMLOCK_LOGIN_MAX_FAILS wrong passwords from one address within the lock window, that address is locked out for
HEMLOCK_LOGIN_LOCK_MINUTES. A global cap on failures stops an attacker who rotates addresses.

Behind a reverse proxy the real address is taken from X-Real-IP (or the last X-Forwarded-For entry), which nginx
overwrites, and only when the direct peer is a private address. Directly exposed, the peer address is used.
"""
import asyncio
import hashlib
import hmac
import html
import ipaddress
import time
from collections import defaultdict, deque
from urllib.parse import parse_qs

COOKIE = "hm_auth"
SESSION_SECONDS = 30 * 24 * 3600

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hemlock</title><style>
:root{color-scheme:light dark;--bg:#fff;--fg:#1a1d21;--card:#f3f5f7;--line:#cfd6dc;--bad:#c0245c}
@media (prefers-color-scheme:dark){:root{--bg:#0b0f19;--fg:#e6e9ee;--card:#161c2b;--line:#2b3448;--bad:#ff7aa8}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:var(--bg);color:var(--fg);
font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:16px}
form{width:100%;max-width:340px;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px}
h1{margin:0 0 4px;font-size:1.6rem}p{margin:0 0 16px;opacity:.75;font-size:.95rem}
input[type=password]{width:100%;font-size:16px;padding:12px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}
button{width:100%;margin-top:12px;font-size:1.05rem;font-weight:600;padding:13px;border:0;border-radius:10px;background:#10b981;color:#fff;cursor:pointer}
.err{margin-top:12px;color:var(--bad);font-size:.92rem}.ghost{position:absolute;left:-9999px;opacity:0}
</style></head><body><form method="post" action="/login">
<h1>Hemlock</h1><p>Enter the password to continue.</p>
<input class="ghost" type="text" name="username" value="hemlock" autocomplete="username" tabindex="-1" readonly aria-hidden="true">
<input type="password" name="password" placeholder="Password" autocomplete="current-password" autofocus required>
<button type="submit">Continue</button>{msg}
</form></body></html>"""


def _is_private(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback
    except ValueError:
        return False


class PasswordGate:
    """Pure ASGI middleware, so streaming responses (Gradio's event stream) pass through untouched."""

    def __init__(self, app, password: str, max_fails: int = 5, lock_seconds: int = 900, global_fails: int = 60):
        self.app, self.password = app, password.encode("utf-8")
        self.key = hashlib.sha256(b"hemlock-session|" + self.password).digest()
        self.max_fails, self.lock_seconds, self.global_fails = max_fails, lock_seconds, global_fails
        self.fails = defaultdict(deque)  # address -> times of recent wrong passwords
        self.locked = {}  # address -> time the lock ends
        self.all_fails = deque()

    # ------------------------------------------------------------ helpers
    def client_ip(self, scope) -> str:
        peer = (scope.get("client") or ("?", 0))[0]
        if not _is_private(peer):
            return peer
        h = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        if h.get("x-real-ip"):
            return h["x-real-ip"].strip()
        if h.get("x-forwarded-for"):
            return h["x-forwarded-for"].split(",")[-1].strip()
        return peer

    def make_cookie(self) -> str:
        exp = str(int(time.time()) + SESSION_SECONDS)
        return exp + "." + hmac.new(self.key, exp.encode(), hashlib.sha256).hexdigest()

    def valid_cookie(self, value: str) -> bool:
        try:
            exp, sig = value.split(".", 1)
            return int(exp) > time.time() and hmac.compare_digest(sig, hmac.new(self.key, exp.encode(), hashlib.sha256).hexdigest())
        except Exception:
            return False

    def cookie_from(self, scope) -> str:
        for k, v in scope["headers"]:
            if k == b"cookie":
                for part in v.decode("latin-1").split(";"):
                    name, _, val = part.strip().partition("=")
                    if name == COOKIE:
                        return val
        return ""

    def retry_after(self, ip: str) -> int:
        now = time.time()
        until = self.locked.get(ip, 0)
        if until <= now:
            self.locked.pop(ip, None)
            until = 0
        while self.all_fails and self.all_fails[0] < now - self.lock_seconds:
            self.all_fails.popleft()
        if len(self.all_fails) >= self.global_fails:
            until = max(until, self.all_fails[0] + self.lock_seconds)
        return max(0, int(until - now) + 1) if until else 0

    def note_fail(self, ip: str):
        now = time.time()
        q = self.fails[ip]
        q.append(now)
        while q and q[0] < now - self.lock_seconds:
            q.popleft()
        self.all_fails.append(now)
        if len(q) >= self.max_fails:
            self.locked[ip] = now + self.lock_seconds
            q.clear()
        if len(self.fails) > 5000:  # keep memory bounded
            for k in [k for k, v in self.fails.items() if not v or v[-1] < now - self.lock_seconds]:
                self.fails.pop(k, None)

    # ------------------------------------------------------------ ASGI
    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        path = scope["path"]
        if scope["type"] == "http" and path == "/login":
            return await self.login(scope, receive, send)
        if self.valid_cookie(self.cookie_from(scope)):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        h = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        wants_page = scope["method"] == "GET" and "text/html" in h.get("accept", "") and not path.startswith("/hm_")
        if wants_page:
            return await self.respond(send, 303, b"", [(b"location", b"/login")])
        await self.respond(send, 401, b'{"detail":"login required"}', [(b"content-type", b"application/json")])

    async def respond(self, send, status, body, headers):
        await send({"type": "http.response.start", "status": status,
                    "headers": headers + [(b"content-length", str(len(body)).encode()), (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})

    async def login(self, scope, receive, send):
        page = lambda msg="": PAGE.replace("{msg}", f"<div class='err'>{html.escape(msg)}</div>" if msg else "").encode("utf-8")
        htmlh = [(b"content-type", b"text/html; charset=utf-8")]
        ip = self.client_ip(scope)
        if scope["method"] != "POST":
            wait = self.retry_after(ip)
            msg = f"Too many wrong passwords. Try again in {max(1, wait // 60)} min." if wait else ""
            return await self.respond(send, 429 if wait else 200, page(msg), htmlh + ([(b"retry-after", str(wait).encode())] if wait else []))
        body, more = b"", True
        while more and len(body) < 4096:
            m = await receive()
            body += m.get("body", b"")
            more = m.get("more_body", False)
        wait = self.retry_after(ip)
        if wait:
            return await self.respond(send, 429, page(f"Too many wrong passwords. Try again in {max(1, wait // 60)} min."),
                                      htmlh + [(b"retry-after", str(wait).encode())])
        given = (parse_qs(body.decode("utf-8", "replace")).get("password") or [""])[0].encode("utf-8")
        if hmac.compare_digest(given, self.password):
            h = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            secure = "; Secure" if h.get("x-forwarded-proto", "").lower() == "https" else ""
            cookie = f"{COOKIE}={self.make_cookie()}; Path=/; Max-Age={SESSION_SECONDS}; HttpOnly; SameSite=Lax{secure}"
            return await self.respond(send, 303, b"", [(b"location", b"/"), (b"set-cookie", cookie.encode())])
        self.note_fail(ip)
        await asyncio.sleep(0.7)  # slows down guessing even before any lockout
        wait = self.retry_after(ip)
        msg = f"Too many wrong passwords. Try again in {max(1, wait // 60)} min." if wait else "Wrong password."
        await self.respond(send, 429 if wait else 401, page(msg), htmlh + ([(b"retry-after", str(wait).encode())] if wait else []))
