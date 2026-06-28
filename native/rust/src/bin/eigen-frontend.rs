use std::io::{self, Read};
use serde_json;
use eigen_native::frontend::lexer::Lexer;
use eigen_native::frontend::parser::Parser;

fn main() {
    let mut source = String::new();
    if io::stdin().read_to_string(&mut source).is_err() {
        eprintln!("Error: Failed to read from stdin");
        std::process::exit(1);
    }

    let mut lexer = Lexer::new(&source);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    let mut parser = Parser::new(tokens);
    let root_id = match parser.parse() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    // Serialize the AST and root_id to stdout
    let output = serde_json::json!({
        "root_id": root_id,
        "ast": parser.ast,
    });
    match serde_json::to_string_pretty(&output) {
        Ok(json_str) => {
            println!("{}", json_str);
        }
        Err(e) => {
            eprintln!("Error: Failed to serialize AST to JSON: {}", e);
            std::process::exit(1);
        }
    }
}
