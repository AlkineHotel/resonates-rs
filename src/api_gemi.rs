use tree_sitter::{Language, Node, Parser, Query, QueryCursor};
use std::collections::HashMap;
use anyhow::{Result, anyhow};

#[derive(Debug, Clone)]
pub struct BackendEndpoint {
    pub method: String,
    pub path: String,
    pub handler: String,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone)]
pub struct FrontendApiCall {
    pub method: Option<String>,
    pub path: String,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub context_snippet: String,
}

pub fn extract_axum_routes(file_content: &str, file_path: &str) -> Result<Vec<BackendEndpoint>> {
    let mut parser = Parser::new();
    parser.set_language(tree_sitter_rust::language())?;
    let tree = parser.parse(file_content, None).ok_or_else(|| anyhow!("Failed to parse Rust code"))?;
    let root_node = tree.root_node();

    let query_string = r#"
        (call_expression
            function: (field_expression
                field: (field_identifier) @method
                field: (field_identifier) @route_macro
            )
            arguments: (arguments
                (string_literal) @path
                (identifier) @handler
            )
        )
    "#;

    let query = Query::new(tree_sitter_rust::language(), query_string)?;
    let mut cursor = QueryCursor::new();
    let mut endpoints = Vec::new();

    for match_ in cursor.matches(&query, root_node, file_content.as_bytes()) {
        let mut method = "".to_string();
        let mut path = "".to_string();
        let mut handler = "".to_string();

        for capture in match_.captures {
            let node = capture.node;
            let text = node.utf8_text(file_content.as_bytes())?;
            match query.capture_names()[capture.index as usize].as_str() {
                "method" => method = text.to_string(),
                "path" => path = text.trim_matches('"').to_string(),
                "handler" => handler = text.to_string(),
                _ => {},
            }
        }

        if !method.is_empty() && !path.is_empty() && !handler.is_empty() {
            // Filter for common HTTP methods used with .route()
            let http_methods = ["get", "post", "put", "delete", "patch", "head", "options"];
            if http_methods.contains(&method.as_str()) {
                endpoints.push(BackendEndpoint {
                    method,
                    path,
                    handler,
                    file_path: file_path.to_string(),
                    start_line: root_node.start_position().row + 1,
                    end_line: root_node.end_position().row + 1,
                });
            }
        }
    }

    Ok(endpoints)
}

pub fn extract_frontend_api_calls(file_content: &str, file_path: &str) -> Result<Vec<FrontendApiCall>> {
    let mut parser = Parser::new();
    parser.set_language(tree_sitter_javascript::language())?;
    let tree = parser.parse(file_content, None).ok_or_else(|| anyhow!("Failed to parse JavaScript/TypeScript code"))?;
    let root_node = tree.root_node();

    let query_string = r#"
        (call_expression
            function: (member_expression
                property: (property_identifier) @method
                object: (identifier) @object
            )
            arguments: (arguments
                (string_literal) @path
            )
        )
        (call_expression
            function: (identifier) @function
            arguments: (arguments
                (string_literal) @path
            )
        )
    "#;

    let query = Query::new(tree_sitter_javascript::language(), query_string)?;
    let mut cursor = QueryCursor::new();
    let mut api_calls = Vec::new();

    for match_ in cursor.matches(&query, root_node, file_content.as_bytes()) {
        let mut method: Option<String> = None;
        let mut path = "".to_string();
        let mut object: Option<String> = None;
        let mut function: Option<String> = None;

        for capture in match_.captures {
            let node = capture.node;
            let text = node.utf8_text(file_content.as_bytes())?;
            match query.capture_names()[capture.index as usize].as_str() {
                "method" => method = Some(text.to_string()),
                "object" => object = Some(text.to_string()),
                "function" => function = Some(text.to_string()),
                "path" => path = text.trim_matches('"').to_string(),
                _ => {},
            }
        }

        let is_api_call = if let Some(obj) = &object {
            obj == "fetch" || obj == "axios" || obj.ends_with("Api") || obj.ends_with("Service")
        } else if let Some(func) = &function {
            func == "fetch" || func.ends_with("Api") || func.ends_with("Service")
        } else {
            false
        };

        if is_api_call && !path.is_empty() {
            let start_byte = match_.pattern_start_byte;
            let end_byte = match_.pattern_end_byte;
            let context_snippet = file_content[start_byte..end_byte].to_string();

            api_calls.push(FrontendApiCall {
                method,
                path,
                file_path: file_path.to_string(),
                start_line: root_node.start_position().row + 1,
                end_line: root_node.end_position().row + 1,
                context_snippet,
            });
        }
    }

    Ok(api_calls)
}
