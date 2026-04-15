import os
from typing import List, Dict, Any

class IngestionGuards:
    def __init__(self):
        self.MAX_REPO_SIZE_MB = 50
        self.MAX_FILES_PER_REPO = 300
        self.ALLOWED_FILE_EXTENSIONS = {".md", ".py", ".rst", ".txt", ".ipynb"}
        self.BLOCKED_REPO_PATTERNS = ["..", "//", "etc/passwd", "etc/shadow"]

    def validate_file_count(self, files: List[Dict[str, Any]]) -> bool:
        if len(files) > self.MAX_FILES_PER_REPO:
            raise ValueError(f"Repository exceeds the limit of {self.MAX_FILES_PER_REPO} files.")
        return True

    def validate_file_extension(self, filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.ALLOWED_FILE_EXTENSIONS:
            return False
        return True

    def validate_path(self, path: str) -> bool:
        """Basic path traversal and malicious path prevention."""
        for pattern in self.BLOCKED_REPO_PATTERNS:
            if pattern in path:
                raise ValueError(f"Malicious path pattern detected: {pattern}")
        return True

    def validate_repo_url(self, url: str) -> bool:
        """Check if the URL looks like a valid GitHub repo."""
        if not url.startswith("https://github.com/"):
            raise ValueError("Invalid GitHub URL. Must start with https://github.com/")
        return True

if __name__ == "__main__":
    guards = IngestionGuards()
    # Test valid
    try:
        guards.validate_repo_url("https://github.com/fastapi/fastapi")
        guards.validate_path("src/main.py")
        print("Guards passed for valid inputs.")
    except Exception as e:
        print(f"Guards failed: {e}")

    # Test invalid
    try:
        guards.validate_path("../../etc/passwd")
    except Exception as e:
        print(f"Guards correctly blocked malicious path: {e}")
