use std::str;
use tree_sitter::{Language, Node, Parser, Range};
use std::fmt;

// --- From error.rs ---
pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

// --- From chunk.rs ---
// A chunk of code with a subtree and a range.
#[derive(Debug)]
pub struct Chunk {
    // Subtree representation of the code chunk.
    pub subtree: String,
    // Range of the code chunk.
    pub range: Range,
    // Size of the code chunk.
    pub size: usize,
}

impl fmt::Display for Chunk {
    // Display the chunk with its range and subtree.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{start}..{end}]: {size}\n{substree}",
            start = self.range.start_point.row,
            end = self.range.end_point.row,
            size = self.size,
            substree = self.subtree,
        )
    }
}

impl Chunk {
    pub fn utf8_lossy(&self, code: &[u8]) -> String {
        String::from_utf8_lossy(&code[self.range.start_byte..self.range.end_byte]).to_string()
    }
}

// --- From sizer.rs ---
// An interface for counting the size of a code chunk.
pub trait Sizer {
    fn size(&self, text: &str) -> Result<usize>;
}

// --- From sizer/chars.rs ---
// A marker struct for counting characters in code chunks.
pub struct CharCounter;

impl Sizer for CharCounter {
    // Count the number of characters in the given text.
    fn size(&self, text: &str) -> Result<usize> {
        Ok(text.chars().count())
    }
}

// --- From sizer/words.rs ---
// A marker struct for counting words in code chunks.
pub struct WordCounter;

impl Sizer for WordCounter {
    // Count the number of words in the given text.
    fn size(&self, text: &str) -> Result<usize> {
        Ok(text.split_whitespace().count())
    }
}


// --- From splitter.rs ---
// Default maximum size of a chunk.
const DEFAULT_MAX_SIZE: usize = 512;

// A struct for splitting code into chunks.
pub struct Splitter<T: Sizer> {
    // Language of the code.
    language: Language,
    // Sizer for counting the size of code chunks.
    sizer: T,
    // Maximum size of a code chunk.
    max_size: usize,
}

impl<T> Splitter<T>
where
    T: Sizer,
{
    // Create a new `Splitter` that counts the size of code chunks with the given sizer.
    pub fn new(language: Language, sizer: T) -> Result<Self> {
        // Ensure tree-sitter-<language> crate can be loaded
        Parser::new().set_language(&language)?;

        Ok(Self {
            language,
            sizer,
            max_size: DEFAULT_MAX_SIZE,
        })
    }

    // Set the maximum size of a chunk. The default is 512.
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }

    // Split the code into chunks with no larger than `max_size`.
    pub fn split(&self, code: &[u8]) -> Result<Vec<Chunk>> {
        if code.is_empty() {
            return Ok(vec![]);
        }

        let mut parser = Parser::new();
        parser
            .set_language(&self.language)
            .expect("Error loading tree-sitter language");
        let tree = parser.parse(code, None).ok_or("Error parsing code")?;
        let root_node = tree.root_node();

        let chunks = self.split_node(&root_node, 0, code)?;

        Ok(chunks)
    }

    fn split_node(&self, node: &Node, depth: usize, code: &[u8]) -> Result<Vec<Chunk>> {
        let text = node.utf8_text(code)?;
        let chunk_size = self.sizer.size(text)?;

        if chunk_size == 0 {
            return Ok(vec![]);
        }

        if chunk_size <= self.max_size {
            return Ok(vec![Chunk {
                subtree: format!("{}: {{}}", format_node(node, depth), chunk_size),
                range: node.range(),
                size: chunk_size,
            }]);
        }

        let chunks = node
            // Traverse the children in depth-first order
            .children(&mut node.walk())
            .map(|child| self.split_node(&child, depth + 1, code))
            .collect::<Result<Vec<_>>>()? 
            .into_iter()
            // Join the tail and head of neighboring chunks if possible
            .try_fold(Vec::new(), |mut acc, mut next| -> Result<Vec<Chunk>> {
                if let Some(tail) = acc.pop() {
                    if let Some(head) = next.first_mut() {
                        let joined_size = self.joined_size(&tail, head, code)?;
                        if joined_size <= self.max_size {
                            // Concatenate the tail and head names
                            head.subtree = format!("{}\n{{}}", tail.subtree, head.subtree);
                            head.range.start_byte = tail.range.start_byte;
                            head.range.start_point = tail.range.start_point;
                            head.size = joined_size;
                        } else {
                            acc.push(tail);
                        }
                    } else {
                        // Push the tail back if next is empty
                        acc.push(tail);
                    }
                }
                acc.append(&mut next);
                Ok(acc)
            })?;

        Ok(chunks)
    }

    fn joined_size(&self, chunk: &Chunk, next: &Chunk, code: &[u8]) -> Result<usize> {
        let joined_bytes = &code[chunk.range.start_byte..next.range.end_byte];
        let joined_text = str::from_utf8(joined_bytes)?;
        self.sizer.size(joined_text)
    }
}

fn format_node(node: &Node, depth: usize) -> String {
    format!(
        "{indent}{{branch}} {{kind:<32}} [{{start}}..{{end}}]",
        indent = "│  ".repeat(depth.saturating_sub(1)),
        branch = if depth > 0 { "├─" } else { "" },
        kind = node.kind(),
        start = node.start_position().row,
        end = node.end_position().row
    )
}