import sys
import logging
from bson import json_util

logger = logging.getLogger("mcp.transport")
logger.setLevel(logging.DEBUG)

class EjsonTransport:
    def __init__(self, reader=None, writer=None):
        self.reader = reader or sys.stdin
        self.writer = writer or sys.stdout

    def read_message(self):
        while True:
            try:
                line = self.reader.readline()
            except Exception as e:
                logger.exception("Error reading line from transport")
                return None

            if line == "":
                return None

            if not line.strip():
                continue

            try:
                return json_util.loads(line)
            except Exception:
                logger.warning(
                    "Failed to parse EJSON message: %r", line.strip(),
                    exc_info=True
                )
                continue

    def write_message(self, msg: dict):
        try:
            text = json_util.dumps(msg)
        except Exception:
            logger.exception("Failed to serialize message to EJSON: %r", msg)
            return

        try:
            self.writer.write(text + "\n")
            self.writer.flush()
        except Exception:
            logger.exception("Error writing message to transport: %r", msg)
