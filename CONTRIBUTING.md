# Contributing to Eigen

Thank you for your interest in contributing to the Eigen programming language! This document outlines our development workflows, code style, and how to submit contributions.

## Code of Conduct
By participating in this project, you agree to abide by the terms of our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs
- Search existing issues to ensure the bug hasn't already been reported.
- Open a new issue with a clear title, description, steps to reproduce, and environment details.

### Suggesting Enhancements
- Open a discussion or issue explaining the feature and why it would be beneficial to Eigen.

### Submitting Pull Requests (PRs)
1. Fork the repository and create your branch from `main`.
2. Ensure your changes do not break existing tests. Run `python -m unittest discover -s tests`.
3. Add tests for any new features or bug fixes.
4. Document your changes if applicable.
5. Submit a PR describing your changes.

## Code Style
- **Python**: Follow PEP 8 guidelines. Write clean, readable code with comments explaining complex algorithms (especially quantum math).
- **Eigen Source**: Keep `.eig` files formatted cleanly:
  - Indent loops and conditional blocks with 4 spaces.
  - Declare variables at the top of scopes.
  - End quantum subroutines (`qfunc`) with a `return` statement.

## Project Structure
- `src/`: Lexer, Parser, AST, Type Checker, EQIR Graph, Optimizer, Simulator, Runtime, Equivalence Checker, and CLI.
- `stdlib/`: Standard Library files.
- `examples/`: Runnable example programs.
- `tests/`: Unit test suite.
- `docs/`: Technical specifications.
