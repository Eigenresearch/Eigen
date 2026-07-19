"""§10.1 — Debugging integration: DAP (Debug Adapter Protocol)
server surface for VSCode step-through debugging.

Implements a minimal DAP server that can:
  - Set/clear breakpoints by source line
  - Step into, over, out of instructions
  - Inspect VM operand stack and frame locals
  - Continue execution until next breakpoint

This is a surface module — actual VSCode integration requires
a .vscode/launch.json that launches this server.
"""
from __future__ import annotations

import dataclasses
import typing


@dataclasses.dataclass
class Breakpoint:
    """A source-level breakpoint."""
    source: str
    line: int
    condition: str | None = None
    hit_count: int = 0
    enabled: bool = True


@dataclasses.dataclass
class StackFrame:
    """A debug stack frame."""
    func_name: str
    line: int
    ip: int
    locals: dict[str, typing.Any]
    operand_stack: list[typing.Any]


class DebugSession:
    """Manages a single debugging session with breakpoints and
    stepping state.
    """

    def __init__(self):
        self.breakpoints: dict[str, list[Breakpoint]] = {}
        self.stepping: bool = False
        self.step_mode: str = ""  # "into", "over", "out", "continue"
        self.paused: bool = False
        self.current_frame: StackFrame | None = None
        self.step_depth: int = 0
        self.target_depth: int = 0

    def set_breakpoint(self, source: str, line: int,
                        condition: str | None = None) -> Breakpoint:
        bp = Breakpoint(source=source, line=line, condition=condition)
        if source not in self.breakpoints:
            self.breakpoints[source] = []
        self.breakpoints[source].append(bp)
        return bp

    def clear_breakpoint(self, source: str, line: int):
        if source in self.breakpoints:
            self.breakpoints[source] = [
                bp for bp in self.breakpoints[source]
                if bp.line != line
            ]

    def clear_all_breakpoints(self, source: str | None = None):
        if source:
            self.breakpoints.pop(source, None)
        else:
            self.breakpoints.clear()

    def hit_breakpoint(self, source: str, line: int) -> bool:
        bps = self.breakpoints.get(source, [])
        for bp in bps:
            if bp.enabled and bp.line == line:
                bp.hit_count += 1
                if bp.condition is None:
                    return True
                # Condition evaluation would go here
                return True
        return False

    def should_pause(self, source: str, line: int,
                      call_depth: int) -> bool:
        if self.hit_breakpoint(source, line):
            self.paused = True
            return True
        if self.stepping:
            if self.step_mode == "into":
                self.paused = True
                return True
            elif self.step_mode == "over":
                if call_depth <= self.target_depth:
                    self.paused = True
                    return True
            elif self.step_mode == "out":
                if call_depth < self.target_depth:
                    self.paused = True
                    return True
        return False

    def step_into(self):
        self.stepping = True
        self.step_mode = "into"
        self.paused = False

    def step_over(self, call_depth: int):
        self.stepping = True
        self.step_mode = "over"
        self.target_depth = call_depth
        self.paused = False

    def step_out(self, call_depth: int):
        self.stepping = True
        self.step_mode = "out"
        self.target_depth = call_depth
        self.paused = False

    def continue_execution(self):
        self.stepping = False
        self.step_mode = "continue"
        self.paused = False

    def update_frame(self, func_name: str, line: int, ip: int,
                      locals_dict: dict, operand_stack: list):
        self.current_frame = StackFrame(
            func_name=func_name,
            line=line,
            ip=ip,
            locals=dict(locals_dict),
            operand_stack=list(operand_stack),
        )

    def handle_dap_request(self, request: dict) -> dict:
        """Handle a single DAP request (surface).

        Returns a DAP response dictionary.
        """
        cmd = request.get("command", "")
        args = request.get("arguments", {})
        seq = request.get("seq", 0)

        if cmd == "setBreakpoints":
            source = args.get("source", {}).get("path", "")
            lines = args.get("lines", [])
            self.clear_all_breakpoints(source)
            bps = []
            for ln in lines:
                bp = self.set_breakpoint(source, ln)
                bps.append({"id": id(bp), "line": ln, "verified": True})
            return {"seq": seq, "type": "response", "command": cmd,
                      "success": True, "body": {"breakpoints": bps}}

        elif cmd == "continue":
            self.continue_execution()
            return {"seq": seq, "type": "response", "command": cmd,
                      "success": True}

        elif cmd == "next":
            self.step_over(self.target_depth)
            return {"seq": seq, "type": "response", "command": cmd,
                      "success": True}

        elif cmd == "stepIn":
            self.step_into()
            return {"seq": seq, "type": "response", "command": cmd,
                      "success": True}

        elif cmd == "stepOut":
            self.step_out(self.target_depth)
            return {"seq": seq, "type": "response", "command": cmd,
                      "success": True}

        elif cmd == "stackTrace":
            frames = []
            if self.current_frame:
                frames.append({
                    "id": 0,
                    "name": self.current_frame.func_name,
                    "line": self.current_frame.line,
                    "column": 1,
                    "source": {"path": ""},
                })
            return {"seq": seq, "type": "response", "command": cmd,
                      "success": True,
                      "body": {"stackFrames": frames,
                                  "totalFrames": len(frames)}}

        elif cmd == "scopes":
            return {"seq": seq, "type": "response", "command": cmd,
                      "success": True,
                      "body": {"scopes": [
                          {"name": "Locals", "variablesReference": 1,
                            "expensive": False},
                          {"name": "Stack", "variablesReference": 2,
                            "expensive": False},
                      ]}}

        elif cmd == "variables":
            ref = args.get("variablesReference", 0)
            variables = []
            if ref == 1 and self.current_frame:
                for k, v in self.current_frame.locals.items():
                    variables.append({"name": k, "value": str(v),
                                        "type": type(v).__name__})
            elif ref == 2 and self.current_frame:
                for i, v in enumerate(self.current_frame.operand_stack):
                    variables.append({"name": f"[{i}]", "value": str(v),
                                        "type": type(v).__name__})
            return {"seq": seq, "type": "response", "command": cmd,
                      "success": True, "body": {"variables": variables}}

        return {"seq": seq, "type": "response", "command": cmd,
                  "success": False,
                  "message": f"Unknown command: {cmd}"}
