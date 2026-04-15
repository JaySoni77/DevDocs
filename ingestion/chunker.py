import os
import ast
import tiktoken
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import markdown

class Chunker:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        # We use gpt-3.5-turbo encoding as a proxy for token counting if exact BGE tokenizer isn't available
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.chunk_size = 800  # Target tokens
        self.overlap = 50      # Overlap tokens

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_markdown(self, text: str) -> List[Dict[str, Any]]:
        # Simple heading-based split
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, "html.parser")
        
        chunks = []
        current_header = ""
        current_content = []
        
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre', 'ul', 'ol']):
            if element.name.startswith('h'):
                if current_content:
                    chunks.append({
                        "header": current_header,
                        "content": "\n".join(current_content)
                    })
                    current_content = []
                current_header = element.get_text()
            else:
                current_content.append(element.get_text())
        
        if current_content:
            chunks.append({
                "header": current_header,
                "content": "\n".join(current_content)
            })
            
        return chunks

    def chunk_python(self, text: str) -> List[Dict[str, Any]]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # Fallback to simple line-based chunking if syntax is invalid
            return [{"header": "file_level", "content": text}]

        chunks = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                chunks.append({
                    "header": node.name,
                    "content": ast.get_source_segment(text, node)
                })
        
        if not chunks:
            chunks.append({"header": "file_level", "content": text})
            
        return chunks

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        # Split by double newline
        paragraphs = text.split("\n\n")
        return [{"header": "text_block", "content": p} for p in paragraphs if p.strip()]

    def process_file(self, filename: str, content: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == ".md" or ext == ".rst":
            raw_chunks = self.chunk_markdown(content)
        elif ext == ".py":
            raw_chunks = self.chunk_python(content)
        else:
            raw_chunks = self.chunk_text(content)
            
        final_chunks = []
        for rc in raw_chunks:
            # If a chunk is too large, we might need further sub-splitting (omitted for now for simplicity)
            final_chunks.append({
                "filename": filename,
                "header": rc["header"],
                "content": rc["content"],
                "tokens": self.count_tokens(rc["content"])
            })
            
        return final_chunks

if __name__ == "__main__":
    chunker = Chunker()
    test_py = """
class MyClass:
    def method(self):
        print("hello")

def my_func():
    return 42
"""
    print("Testing Python Chunking:")
    for c in chunker.process_file("test.py", test_py):
        print(f"[{c['header']}] Tokens: {c['tokens']}\n{c['content'][:50]}...")

    test_md = """
# Title
Intro text here.

## Section 1
Details about section 1.
- list item

## Section 2
More details.
"""
    print("\nTesting Markdown Chunking:")
    for c in chunker.process_file("test.md", test_md):
        print(f"[{c['header']}] Tokens: {c['tokens']}\n{c['content'][:50]}...")
