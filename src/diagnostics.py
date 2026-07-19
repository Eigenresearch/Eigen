from enum import Enum
from dataclasses import dataclass
from typing import Optional

class DiagnosticSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    HINT = "hint"

class DiagnosticCode:
    E001 = "E001"
    E002 = "E002"
    W001 = "W001"

@dataclass
class SourceLocation:
    filepath: str
    line: int | None = None
    column: int | None = None

    def __str__(self) -> str:
        loc_str = self.filepath
        if self.line is not None:
            loc_str += f":{self.line}"
            if self.column is not None:
                loc_str += f":{self.column}"
        return loc_str

@dataclass
class Diagnostic:
    severity: DiagnosticSeverity
    message: str
    location: SourceLocation | None = None
    code: str | None = None
    fix_suggestion: Optional[str] = None

    def __str__(self) -> str:
        loc_part = f"[{self.location}] " if self.location else ""
        code_part = f" ({self.code})" if self.code else ""
        fix_part = f" [fix: {self.fix_suggestion}]" if self.fix_suggestion else ""
        return f"{loc_part}{self.severity.value.upper()}: {self.message}{code_part}{fix_part}"

class DiagnosticEngine:
    def __init__(self):
        self.diagnostics = []

    def emit(
        self, severity: DiagnosticSeverity, message: str,
        location: SourceLocation | None = None, code: str | None = None,
        fix_suggestion: Optional[str] = None
    ) -> Diagnostic:
        diag = Diagnostic(severity, message, location, code, fix_suggestion)
        self.diagnostics.append(diag)
        return diag

    def get_warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.WARNING]

    def get_errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.ERROR]

    def get_infos(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.INFO]

    def get_hints(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == DiagnosticSeverity.HINT]

    def has_errors(self) -> bool:
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)

    def clear(self):
        self.diagnostics.clear()
