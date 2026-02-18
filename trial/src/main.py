import sys
import signal
import logging

from configs.config         import load_config
from src.transport          import EjsonTransport
from src.server             import RpcServer
from src.session            import Session
from src.telemetry          import Telemetry
from configs.logging_config import setup_logging
from tools                  import ALL_TOOLS

logger = logging.getLogger("mcp.main")
logger.setLevel(logging.DEBUG)

def main():
    try:
        config = load_config()
    except Exception:
        logger.exception("Failed to load config")
        sys.exit(1)

    try:
        session = Session(config)
        session.connect()
    except Exception:
        logger.exception("Failed to initialize or connect Session")
        sys.exit(1)

    try:
        telemetry = Telemetry(config)
    except Exception:
        logger.exception("Failed to initialize Telemetry")
        sys.exit(1)

    try:
        server = RpcServer(session, config, telemetry)
    except Exception:
        logger.exception("Failed to create RpcServer")
        sys.exit(1)

    try:
        setup_logging(server, config.log_path, config.log_level)
    except Exception:
        logger.exception("Error in setup_logging; continuing with default logger")

    try:
        transport = EjsonTransport()
    except Exception:
        logger.exception("Failed to create EjsonTransport")
        sys.exit(1)

    for cap in ("logging", "streaming", "interrupt"):
        try:
            transport.write_message({
                "jsonrpc": "1.0",
                "method":  "capabilities",
                "params":  {"name": cap, "enabled": True}
            })
        except Exception:
            logger.exception("Failed to announce capability '%s'", cap)

    try:
        server.resource("config://config", lambda: config.__dict__)
    except Exception:
        logger.exception("Failed to register config resource")

    for ToolClass in ALL_TOOLS:
        try:
            server.register_tool(ToolClass)
        except Exception:
            logger.exception("Failed to register tool %s", ToolClass)

    try:
        telemetry.record("server_start", 0, True)
    except Exception:
        logger.exception("Failed to record server_start telemetry")

    def _shutdown(signum, frame):
        try:
            telemetry.record("server_stop", 0, True)
        except Exception:
            logger.exception("Failed to record server_stop telemetry")
        try:
            telemetry.flush()
        except Exception:
            logger.exception("Failed to flush telemetry")
        try:
            server.close()
        except Exception:
            logger.exception("Error closing RpcServer")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        server.serve(transport)
    except Exception:
        logger.exception("Fatal error in serve loop")
        try:
            telemetry.record("server_stop", 0, False)
        except Exception:
            pass
        try:
            telemetry.flush()
        except Exception:
            pass
        try:
            server.close()
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
