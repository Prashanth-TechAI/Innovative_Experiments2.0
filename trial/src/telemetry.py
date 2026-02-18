import time
import threading
import requests
import platform
import re
import logging
import atexit
from cachetools import LRUCache
from typing import Any, Dict, Optional
from bson import json_util

logger = logging.getLogger(__name__)

_SENSITIVE_KEYS = [
    "password", "pwd", "secret", "apiKey", "accessToken",
    "authorization", "clientSecret", "privateKey"
]
_RE = re.compile(
    r'("(?:(?:' + "|".join(_SENSITIVE_KEYS) + r'))"\s*:\s*)"([^"]+)"',
    flags=re.IGNORECASE
)

def _redact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    def redact_val(v):
        if isinstance(v, str):
            redacted = _RE.sub(r'\1"<REDACTED>"', v)
            if redacted != v:
                logger.debug("Redacted sensitive field value")
            return redacted
        elif isinstance(v, dict):
            return {k: redact_val(vv) for k, vv in v.items()}
        elif isinstance(v, list):
            return [redact_val(x) for x in v]
        else:
            return v
    return {k: redact_val(v) for k, v in d.items()}

class Telemetry:
    def __init__(self, config):
        self.config = config
        self.enabled = getattr(config, "telemetry", "").lower() == "enabled"
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._cache = LRUCache(maxsize=getattr(config, "telemetry_cache_size", 1000))
        self._lock = threading.Lock()
        self._next_idx = 0
        self._flush_interval = getattr(config, "telemetry_flush_interval", 60)
        self._max_retries    = getattr(config, "telemetry_max_retries", 3)
        self._timeout        = getattr(config, "telemetry_timeout", 5)

        if (
            self.enabled
            and getattr(config, "api_base_url", None)
            and getattr(config, "api_client_id", None)
            and getattr(config, "api_client_secret", None)
        ):
            self._stop_event = threading.Event()
            self._thread = threading.Thread(
                target=self._periodic_flush,
                daemon=True,
                name="TelemetryFlusher"
            )
            self._thread.start()
            atexit.register(self.shutdown)
            logger.info(
                f"Telemetry enabled: flushing every {self._flush_interval}s, "
                f"cache size {self._cache.maxsize}"
            )
        else:
            if self.enabled:
                logger.warning(
                    "Telemetry enabled but missing API config; background flush disabled"
                )
            else:
                logger.info("Telemetry disabled")

    def record(
        self,
        command: str,
        duration_ms: int,
        success: bool,
        arguments: Optional[Dict[str, Any]] = None
    ):

        if not self.enabled:
            return

        event: Dict[str, Any] = {
            "command":    command,
            "durationMs": duration_ms,
            "success":    success,
            "timestamp":  int(time.time() * 1000),
        }
        if arguments:
            event["arguments"] = _redact_dict(arguments)

        if command == "server_start":
            event["metadata"] = {
                "os":        platform.system(),
                "osVersion": platform.version(),
                "python":    platform.python_version(),
                "appName":   "MCP-Python"
            }

        with self._lock:
            if len(self._cache) >= self._cache.maxsize:
                oldest = min(self._cache.keys())
                self._cache.pop(oldest, None)
                logger.debug(f"Dropped oldest telemetry event at index {oldest}")
            idx = self._next_idx
            self._cache[idx] = event
            self._next_idx += 1
            logger.debug(f"Recorded telemetry event '{command}' at index {idx}")

    def flush(self):

        if not self.enabled or not getattr(self.config, "api_base_url", None):
            return

        with self._lock:
            events = list(self._cache.values())
            self._cache.clear()

        if not events:
            logger.debug("No telemetry events to flush")
            return

        try:
            payload = json_util.dumps(events)
        except Exception as e:
            logger.error(f"Failed to serialize telemetry events: {e}")
            return

        url = f"{self.config.api_base_url.rstrip('/')}/v2/telemetry"
        headers = {"Content-Type": "application/json"}
        auth = (self.config.api_client_id, self.config.api_client_secret)

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    data=payload,
                    headers=headers,
                    timeout=self._timeout,
                    auth=auth
                )
                status = resp.status_code

                if 200 <= status < 300:
                    logger.info(f"Flushed {len(events)} telemetry events successfully")
                    return

                if 400 <= status < 500:
                    logger.error(
                        f"Telemetry flush aborted (HTTP {status}): {resp.text}"
                    )
                    return

                logger.warning(
                    f"Telemetry flush attempt {attempt} failed (HTTP {status}); retrying..."
                )

            except requests.RequestException as e:
                logger.warning(
                    f"Telemetry flush attempt {attempt} network error: {e}"
                )

            time.sleep(attempt)
        logger.error("Max telemetry retries reached; dropping telemetry events")

    def _periodic_flush(self):
        logger.debug("Telemetry flusher thread started")
        while self._stop_event and not self._stop_event.wait(self._flush_interval):
            self.flush()
        logger.debug("Telemetry flusher thread exiting")

    def shutdown(self):
        if not self._thread or not self._stop_event:
            return
        logger.info("Shutting down telemetry")
        self._stop_event.set()
        self._thread.join(timeout=2)

        try:
            self.flush()
        except Exception as e:
            logger.warning(f"Error flushing telemetry on shutdown: {e}")
        logger.info("Telemetry shutdown complete")
