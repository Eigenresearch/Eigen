from .release import VERSION, CODENAME, RELEASE_LABEL
from .frontend.lexer import Lexer
from .frontend.parser import Parser
from .backend.vm import EigenVM

__version__ = VERSION
__all__ = ["Lexer", "Parser", "EigenVM", "VERSION", "CODENAME", "RELEASE_LABEL"]
