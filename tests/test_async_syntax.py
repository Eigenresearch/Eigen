"""Native async/await and operator-declaration frontend tests."""
import unittest

from src.frontend.ast import (
    AsyncFuncDeclNode,
    AwaitExprNode,
    ImplBlockNode,
    LetNode,
    OperatorDeclNode,
    ReturnNode,
)
from src.frontend.lexer import Lexer, TokenType
from src.frontend.parser import Parser
from src.semantic.type_checker import TypeChecker, TypeErrorException


def parse(source: str):
    return Parser(Lexer(source).tokenize()).parse()


class TestAsyncLexer(unittest.TestCase):
    def test_async_keywords_are_reserved(self):
        tokens = Lexer("async func f() { await task }").tokenize()
        token_types = [token.type for token in tokens]
        self.assertEqual(token_types[0], TokenType.ASYNC)
        self.assertEqual(token_types[1], TokenType.FUNC)
        self.assertIn(TokenType.AWAIT, token_types)
        self.assertIn(TokenType.TASK, token_types)

    def test_operator_keyword_is_reserved(self):
        token = Lexer("operator +").tokenize()[0]
        self.assertEqual(token.type, TokenType.OPERATOR)


class TestAsyncParser(unittest.TestCase):
    SOURCE = """eigen 1.0
async func fetch(value: int) -> int {
    return value
}
async func pipeline() -> int {
    let pending: task<int> = fetch(7)
    return await pending
}
"""

    def test_async_function_nodes(self):
        program = parse(self.SOURCE)
        self.assertEqual(len(program.body), 2)
        self.assertTrue(all(isinstance(node, AsyncFuncDeclNode)
                            for node in program.body))
        self.assertEqual(program.body[0].name, "fetch")
        self.assertEqual(program.body[0].return_type, "int")

    def test_await_expression_is_preserved(self):
        program = parse(self.SOURCE)
        pipeline = program.body[1]
        self.assertIsInstance(pipeline.body[0], LetNode)
        self.assertEqual(pipeline.body[0].type_name, "task<int>")
        self.assertIsInstance(pipeline.body[1], ReturnNode)
        self.assertIsInstance(pipeline.body[1].expr, AwaitExprNode)

    def test_async_program_type_checks(self):
        TypeChecker().check(parse(self.SOURCE))

    def test_await_outside_async_is_rejected(self):
        source = """eigen 1.0
async func fetch() -> int { return 1 }
func main() -> int { return await fetch() }
"""
        with self.assertRaisesRegex(
                TypeErrorException, "only valid inside an async function"):
            TypeChecker().check(parse(source))

    def test_await_requires_task_value(self):
        source = """eigen 1.0
async func bad() -> int { return await 1 }
"""
        with self.assertRaisesRegex(TypeErrorException, "expects task<T>"):
            TypeChecker().check(parse(source))


class TestOperatorDeclarations(unittest.TestCase):
    SOURCE = """eigen 1.0
struct Vector {
    value: int
}
impl Vector {
    operator +(other: Vector) -> Vector {
        return self
    }
    operator ==(other: Vector) -> bool {
        return true
    }
}
"""

    def test_operator_declarations_lower_to_canonical_method_names(self):
        program = parse(self.SOURCE)
        impl = next(node for node in program.body
                    if isinstance(node, ImplBlockNode))
        self.assertEqual(len(impl.methods), 2)
        self.assertTrue(all(isinstance(node, OperatorDeclNode)
                            for node in impl.methods))
        self.assertEqual(impl.methods[0].operator, "+")
        self.assertEqual(impl.methods[0].method_name, "__add__")
        self.assertEqual(impl.methods[1].method_name, "__eq__")

    def test_operator_implementation_type_checks(self):
        TypeChecker().check(parse(self.SOURCE))

    def test_non_overloadable_assignment_operator_is_rejected(self):
        source = """eigen 1.0
struct Vector { value: int }
impl Vector { operator =(other: Vector) -> Vector { return self } }
"""
        with self.assertRaisesRegex(SyntaxError, "overloadable operator"):
            parse(source)


if __name__ == "__main__":
    unittest.main()
