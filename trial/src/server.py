import time
import logging
from typing import Callable, Any, Optional, Dict

class RpcServer:
    def __init__(self, session, config, telemetry):
        self.session     = session
        self.config      = config
        self.telemetry   = telemetry
        self.tools       = {}
        self.capabilities      = set()
        self.resource_handlers = {}
        self.log_subscribers   = set()
        self.stream_subscribers    = set()
        self.interrupt_subscribers = set()
        self.logger      = logging.getLogger("mcp.server")
        self.logger.setLevel(logging.DEBUG)

        self._on_initialized:  list[Callable[[], None]]            = []
        self._on_close:        list[Callable[[Optional[Exception]], None]] = []
        self._on_error:        list[Callable[[Exception], None]]   = []

    def on_initialized(self, fn: Callable[[], None]):
        self._on_initialized.append(fn)

    def on_close(self, fn: Callable[[Optional[Exception]], None]):
        self._on_close.append(fn)

    def on_error(self, fn: Callable[[Exception], None]):
        self._on_error.append(fn)

    def register_tool(self, tool_cls):
        try:
            tool = tool_cls(self.session, self.telemetry)
            self.tools[tool.name] = tool
            self.logger.info("Registered tool '%s'", tool.name)
        except Exception as e:
            self.logger.exception("Error registering tool %s: %s", tool_cls, e)

    def capability(self, name: str):
        try:
            self.capabilities.add(name)
        except Exception as e:
            self.logger.exception("Error adding capability '%s': %s", name, e)

    def resource(self, name: str, handler: Callable[[], Any]):
        try:
            self.resource_handlers[name] = handler
        except Exception as e:
            self.logger.exception("Error registering resource handler '%s': %s", name, e)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self.tools:
            self.logger.error("Tool '%s' not found", name)
            raise ValueError(f"Tool '{name}' not found")
        return self._call_tool(name, arguments)

    def serve(self, transport):
        self.transport = transport
        for fn in self._on_initialized:
            try:
                fn()
            except Exception as e:
                self.logger.exception("Error in on_initialized hook: %s", e)

        try:
            while True:
                try:
                    req = transport.read_message()
                except Exception as e:
                    self.logger.exception("Error reading message from transport: %s", e)
                    for err_fn in self._on_error:
                        try: err_fn(e)
                        except Exception: self.logger.exception("Error in on_error hook")
                    break

                if req is None:
                    break

                self._handle(req)

        except Exception as err:
            self.logger.exception("Fatal error in serve loop: %s", err)
            for fn in self._on_error:
                try: fn(err)
                except Exception:
                    self.logger.exception("Error in on_error hook")
            raise

        finally:
            for fn in self._on_close:
                try: fn(None)
                except Exception:
                    self.logger.exception("Error in on_close hook")

    def _handle(self, req: Dict):
        try:
            method = req.get("method")
            id_    = req.get("id")
        except Exception as e:
            self.logger.exception("Invalid request format: %s", e)
            return
        try:
            if method == "capability":
                result = self._handle_capability(req.get("params", {}))
            elif method == "resource":
                result = self._handle_resource(req.get("params", {}))
            else:
                params = req.get("params") or {}
                args   = params.get("arguments", params)
                result = self._call_tool(method, args)

            resp = {"jsonrpc": "1.0", "id": id_, "result": result}

        except Exception as e:
            self.logger.exception("Error handling request %r", req)
            resp = {
                "jsonrpc": "1.0",
                "id": id_,
                "error": {"code": -32000, "message": str(e)}
            }

        try:
            self.transport.write_message(resp)
        except Exception as e:
            self.logger.exception("Error writing response to transport: %s", e)

    def _handle_capability(self, params: Dict[str, Any]) -> bool:
        try:
            name    = params["name"]
            enabled = params.get("enabled", True)
        except KeyError as e:
            self.logger.error("Missing parameter in capability: %s", e)
            raise ValueError("Missing 'name' in capability params")

        try:
            if name not in ("logging", "streaming", "interrupt"):
                raise ValueError(f"Unknown capability '{name}'")

            if enabled:
                self.capabilities.add(name)
            else:
                self.capabilities.discard(name)

            if name == "logging":
                (self.log_subscribers if enabled else self.log_subscribers.discard)(self.transport)
            elif name == "streaming":
                (self.stream_subscribers if enabled else self.stream_subscribers.discard)(self.transport)
            elif name == "interrupt":
                (self.interrupt_subscribers if enabled else self.interrupt_subscribers.discard)(self.transport)

            return True

        except Exception as e:
            self.logger.exception("Error in _handle_capability: %s", e)
            raise

    def _handle_resource(self, params: Dict[str, Any]) -> Any:
        try:
            name = params["name"]
        except KeyError as e:
            self.logger.error("Missing 'name' in resource params: %s", e)
            raise ValueError("Missing 'name' in resource params")

        try:
            if name not in self.resource_handlers:
                raise ValueError(f"Unknown resource '{name}'")
            return self.resource_handlers[name]()
        except Exception as e:
            self.logger.exception("Error in _handle_resource for '%s': %s", name, e)
            raise

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        self.logger.info("Tool `%s` started with args %r", name, args)
        start   = time.time()
        success = False
        try:
            result  = self.tools[name].run(args)
            success = True
            return result
        except Exception as e:
            self.logger.exception("Tool `%s` failed: %s", name, e)
            raise
        finally:
            duration = int((time.time() - start) * 1000)
            try:
                self.logger.info("Tool `%s` finished in %dms", name, duration)
                self.telemetry.record(name, duration, success)
            except Exception as e:
                self.logger.exception("Error recording telemetry for tool `%s`: %s", name, e)

    def close(self):
        self.logger.info("Shutting down RPC server")
        for fn in self._on_close:
            try:
                fn(None)
            except Exception:
                self.logger.exception("Error in on_close hook")
        try:
            self.telemetry.flush()
        except Exception as e:
            self.logger.exception("Error flushing telemetry: %s", e)
        try:
            self.session.close()
        except Exception as e:
            self.logger.exception("Error closing session: %s", e)
