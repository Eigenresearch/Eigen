use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TokenType {
    // Directives
    Eigen,
    // Types
    Qubit, Cbit, Int, Float, StringType, Bool, Array, Map,
    // Keywords
    Module, Import, Qfunc, Func, Struct, Enum, Try, Catch, Throw, Noise, Depolarizing, Bitflip, Let, If, For, In, While, Break, Continue, Measure, Return, Parallel, Task,
    // Built-ins / Utilities
    Trace, Print, Assert,
    // Constants
    Pi, Tau, E,
    // Gates
    GateH, GateX, GateY, GateZ, GateS, GateT, GateCnot, GateCz, GateSwap, GateRx, GateRy, GateRz,
    // Literals
    StringLit, True, False, Null,
    // General
    Identifier, IntLit, FloatLit,
    // Operators & Delimiters
    Lparen, Rparen, Lbrace, Rbrace, Lbrack, Rbrack, Comma, Colon, Dot, Arrow, Equals, Eq, Ne, Lt, Gt, Le, Ge,
    Plus, Minus, Mul, Div,
    AddAssign, SubAssign, MulAssign, DivAssign,
    And, Or, Not,
    Eof,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Token<'a> {
    pub token_type: TokenType,
    pub value: &'a str,
    pub line: usize,
    pub column: usize,
}

pub struct Lexer<'a> {
    source: &'a str,
    pos: usize,
    line: usize,
    column: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    fn error(&self, msg: &str) -> String {
        format!("Lexer Error at line {}, col {}: {}", self.line, self.column, msg)
    }

    fn peek(&self, offset: usize) -> Option<char> {
        self.source[self.pos..].chars().nth(offset)
    }

    fn advance(&mut self) {
        if self.pos < self.source.len() {
            let ch = self.source[self.pos..].chars().next().unwrap();
            self.pos += ch.len_utf8();
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token<'a>>, String> {
        let mut tokens = Vec::new();

        while self.pos < self.source.len() {
            let current_char = self.source[self.pos..].chars().next().unwrap();

            // Skip whitespace
            if current_char.is_whitespace() {
                if current_char == '\n' {
                    self.line += 1;
                    self.column = 1;
                    self.pos += 1;
                } else {
                    let start_pos = self.pos;
                    while self.pos < self.source.len() {
                        let ch = self.source[self.pos..].chars().next().unwrap();
                        if ch.is_whitespace() && ch != '\n' {
                            self.pos += ch.len_utf8();
                        } else {
                            break;
                        }
                    }
                    self.column += self.pos - start_pos;
                }
                continue;
            }

            // Skip comments (# and //)
            if current_char == '#' || (current_char == '/' && self.peek(1) == Some('/')) {
                while self.pos < self.source.len() {
                    let ch = self.source[self.pos..].chars().next().unwrap();
                    if ch == '\n' {
                        break;
                    }
                    self.pos += ch.len_utf8();
                }
                // We do NOT consume the '\n' because whitespace check will handle it to increment line count.
                continue;
            }

            // Double-quoted string literals
            if current_char == '"' {
                let start_col = self.column;
                let start_line = self.line;
                self.advance(); // consume open quote
                let start_pos = self.pos;

                while self.pos < self.source.len() {
                    let ch = self.source[self.pos..].chars().next().unwrap();
                    if ch == '"' {
                        break;
                    }
                    if ch == '\n' {
                        return Err(format!("Lexer Error at line {}, col {}: Unterminated string literal", start_line, start_col));
                    }
                    if ch == '\\' {
                        self.advance(); // consume '\\'
                        if self.pos < self.source.len() {
                            self.advance(); // consume escaped char
                        }
                    } else {
                        self.advance();
                    }
                }

                if self.pos >= self.source.len() || self.source[self.pos..].chars().next() != Some('"') {
                    return Err(format!("Lexer Error at line {}, col {}: Unterminated string literal", start_line, start_col));
                }

                let literal_val = &self.source[start_pos..self.pos];
                self.advance(); // consume close quote
                tokens.push(Token {
                    token_type: TokenType::StringLit,
                    value: literal_val,
                    line: start_line,
                    column: start_col,
                });
                continue;
            }

            // Multi-character operators and symbols
            if current_char == '-' && self.peek(1) == Some('>') {
                let start_col = self.column;
                self.advance();
                self.advance();
                tokens.push(Token { token_type: TokenType::Arrow, value: "->", line: self.line, column: start_col });
                continue;
            }

            if current_char == '=' && self.peek(1) == Some('=') {
                let start_col = self.column;
                self.advance();
                self.advance();
                tokens.push(Token { token_type: TokenType::Eq, value: "==", line: self.line, column: start_col });
                continue;
            }

            if current_char == '!' && self.peek(1) == Some('=') {
                let start_col = self.column;
                self.advance();
                self.advance();
                tokens.push(Token { token_type: TokenType::Ne, value: "!=", line: self.line, column: start_col });
                continue;
            }

            if current_char == '<' && self.peek(1) == Some('=') {
                let start_col = self.column;
                self.advance();
                self.advance();
                tokens.push(Token { token_type: TokenType::Le, value: "<=", line: self.line, column: start_col });
                continue;
            }

            if current_char == '>' && self.peek(1) == Some('=') {
                let start_col = self.column;
                self.advance();
                self.advance();
                tokens.push(Token { token_type: TokenType::Ge, value: ">=", line: self.line, column: start_col });
                continue;
            }

            if current_char == '+' && self.peek(1) == Some('=') {
                let start_col = self.column;
                self.advance();
                self.advance();
                tokens.push(Token { token_type: TokenType::AddAssign, value: "+=", line: self.line, column: start_col });
                continue;
            }

            if current_char == '-' && self.peek(1) == Some('=') {
                let start_col = self.column;
                self.advance();
                self.advance();
                tokens.push(Token { token_type: TokenType::SubAssign, value: "-=", line: self.line, column: start_col });
                continue;
            }

            if current_char == '*' && self.peek(1) == Some('=') {
                let start_col = self.column;
                self.advance();
                self.advance();
                tokens.push(Token { token_type: TokenType::MulAssign, value: "*=", line: self.line, column: start_col });
                continue;
            }

            if current_char == '/' && self.peek(1) == Some('=') {
                let start_col = self.column;
                self.advance();
                self.advance();
                tokens.push(Token { token_type: TokenType::DivAssign, value: "/=", line: self.line, column: start_col });
                continue;
            }

            // Single character tokens
            let single_char_token = match current_char {
                '(' => Some(TokenType::Lparen),
                ')' => Some(TokenType::Rparen),
                '{' => Some(TokenType::Lbrace),
                '}' => Some(TokenType::Rbrace),
                '[' => Some(TokenType::Lbrack),
                ']' => Some(TokenType::Rbrack),
                ',' => Some(TokenType::Comma),
                ':' => Some(TokenType::Colon),
                '.' => Some(TokenType::Dot),
                '=' => Some(TokenType::Equals),
                '+' => Some(TokenType::Plus),
                '-' => Some(TokenType::Minus),
                '*' => Some(TokenType::Mul),
                '/' => Some(TokenType::Div),
                '<' => Some(TokenType::Lt),
                '>' => Some(TokenType::Gt),
                _ => None,
            };

            if let Some(t_type) = single_char_token {
                let val = &self.source[self.pos..self.pos + 1];
                tokens.push(Token {
                    token_type: t_type,
                    value: val,
                    line: self.line,
                    column: self.column,
                });
                self.advance();
                continue;
            }

            // Numbers
            if current_char.is_ascii_digit() {
                let start_pos = self.pos;
                let start_col = self.column;
                while self.pos < self.source.len() && self.source[self.pos..].chars().next().unwrap().is_ascii_digit() {
                    self.advance();
                }

                // Check for decimal point
                if self.pos < self.source.len() && self.source[self.pos..].starts_with('.') {
                    if let Some(next_ch) = self.peek(1) {
                        if next_ch.is_ascii_digit() {
                            self.advance(); // consume '.'
                            while self.pos < self.source.len() && self.source[self.pos..].chars().next().unwrap().is_ascii_digit() {
                                self.advance();
                            }
                            let val = &self.source[start_pos..self.pos];
                            tokens.push(Token {
                                token_type: TokenType::FloatLit,
                                value: val,
                                line: self.line,
                                column: start_col,
                            });
                            continue;
                        }
                    }
                }

                let val = &self.source[start_pos..self.pos];
                tokens.push(Token {
                    token_type: TokenType::IntLit,
                    value: val,
                    line: self.line,
                    column: start_col,
                });
                continue;
            }

            // Identifiers / Keywords
            if current_char.is_alphabetic() || current_char == '_' {
                let start_pos = self.pos;
                let start_col = self.column;
                while self.pos < self.source.len() {
                    let ch = self.source[self.pos..].chars().next().unwrap();
                    if ch.is_alphanumeric() || ch == '_' {
                        self.advance();
                    } else {
                        break;
                    }
                }
                let val = &self.source[start_pos..self.pos];
                let t_type = match val {
                    "eigen" => TokenType::Eigen,
                    "qubit" => TokenType::Qubit,
                    "cbit" => TokenType::Cbit,
                    "int" => TokenType::Int,
                    "float" => TokenType::Float,
                    "string" => TokenType::StringType,
                    "bool" => TokenType::Bool,
                    "array" => TokenType::Array,
                    "map" => TokenType::Map,
                    "module" => TokenType::Module,
                    "import" => TokenType::Import,
                    "qfunc" => TokenType::Qfunc,
                    "func" => TokenType::Func,
                    "struct" => TokenType::Struct,
                    "enum" => TokenType::Enum,
                    "try" => TokenType::Try,
                    "catch" => TokenType::Catch,
                    "throw" => TokenType::Throw,
                    "noise" => TokenType::Noise,
                    "depolarizing" => TokenType::Depolarizing,
                    "bitflip" => TokenType::Bitflip,
                    "let" => TokenType::Let,
                    "if" => TokenType::If,
                    "for" => TokenType::For,
                    "in" => TokenType::In,
                    "while" => TokenType::While,
                    "break" => TokenType::Break,
                    "continue" => TokenType::Continue,
                    "measure" => TokenType::Measure,
                    "return" => TokenType::Return,
                    "parallel" => TokenType::Parallel,
                    "task" => TokenType::Task,
                    "trace" => TokenType::Trace,
                    "print" => TokenType::Print,
                    "assert" => TokenType::Assert,
                    "PI" => TokenType::Pi,
                    "TAU" => TokenType::Tau,
                    "E" => TokenType::E,
                    "H" => TokenType::GateH,
                    "X" => TokenType::GateX,
                    "Y" => TokenType::GateY,
                    "Z" => TokenType::GateZ,
                    "S" => TokenType::GateS,
                    "T" => TokenType::GateT,
                    "CNOT" => TokenType::GateCnot,
                    "CZ" => TokenType::GateCz,
                    "SWAP" => TokenType::GateSwap,
                    "RX" => TokenType::GateRx,
                    "RY" => TokenType::GateRy,
                    "RZ" => TokenType::GateRz,
                    "true" => TokenType::True,
                    "false" => TokenType::False,
                    "null" => TokenType::Null,
                    "and" => TokenType::And,
                    "or" => TokenType::Or,
                    "not" => TokenType::Not,
                    _ => TokenType::Identifier,
                };

                tokens.push(Token {
                    token_type: t_type,
                    value: val,
                    line: self.line,
                    column: start_col,
                });
                continue;
            }

            return Err(self.error(&format!("Unexpected character: {:?}", current_char)));
        }

        tokens.push(Token {
            token_type: TokenType::Eof,
            value: "",
            line: self.line,
            column: self.column,
        });

        Ok(tokens)
    }
}
