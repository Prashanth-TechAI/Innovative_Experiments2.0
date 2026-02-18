import os
import re
import json
import logging
from logging.handlers import RotatingFileHandler

_SENSITIVE = [
    "password", "pwd", "secret", "apiKey", "accessToken", "authorization",
    "clientSecret", "privateKey", "certificate", "passphrase"
]
_RE = re.compile(
    r'("(?:' + "|".join(_SENSITIVE) + r')"\s*:\s*)"([^"]+)"',
    flags=re.IGNORECASE
)

def redact(msg: str) -> str:
    return _RE.sub(r'\1"<REDACTED>"', msg)

class DiskLogger(logging.Handler):

    def __init__(self, path: str, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
        super().__init__()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        self.handler.setFormatter(logging.Formatter(fmt='%(message)s'))

    def emit(self, record: logging.LogRecord):
        try:
            log_entry = {
                "timestamp": getattr(record, "asctime", self.formatTime(record)),
                "level": record.levelname,
                "logger": record.name,
                "message": redact(record.getMessage()),
                "module": record.module,
                "funcName": record.funcName,
                "lineNo": record.lineno,
            }
            jr = logging.LogRecord(
                name=record.name,
                level=record.levelno,
                pathname=record.pathname,
                lineno=record.lineno,
                msg=json.dumps(log_entry),
                args=(),
                exc_info=record.exc_info
            )
            self.handler.emit(jr)
        except Exception:
            pass

class McpLogger(logging.Handler):
    def __init__(self, server):
        super().__init__()
        self.server = server

    def emit(self, record: logging.LogRecord):
        try:
            msg = redact(record.getMessage())
            notif = {
                "jsonrpc": "1.0",
                "method": "logging",
                "params": {
                    "level": record.levelname,
                    "logger": record.name,
                    "message": msg,
                    "metadata": {
                        "module": record.module,
                        "funcName": record.funcName,
                        "lineNo": record.lineno
                    }
                }
            }
            for transport in list(self.server.log_subscribers):
                try:
                    transport.write_message(notif)
                except Exception:
                    self.server.log_subscribers.discard(transport)
        except Exception:
            pass

def setup_logging(server, log_path: str, level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    console_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(console_fmt)

    class RedactFilter(logging.Filter):
        def filter(self, rec: logging.LogRecord) -> bool:
            rec.msg = redact(rec.getMessage())
            rec.args = ()
            return True

    ch.addFilter(RedactFilter())
    root.addHandler(ch)
    dl = DiskLogger(path=log_path)
    dl.setLevel(level)
    root.addHandler(dl)
    ml = McpLogger(server)
    ml.setLevel(level)
    root.addHandler(ml)

    if not hasattr(server, "log_subscribers"):
        server.log_subscribers = set()

    root.debug("Logging initialized: console, disk, and MCP RPC")
