import os
import requests
import base64
from typing import List, Dict, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

class GitHubLoader:
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN_1")
        self.headers = {"Authorization": f"token {self.github_token}"} if self.github_token else {}
        self.allowed_extensions = {".md", ".py", ".rst", ".txt", ".ipynb"}
        self.ignored_dirs = {"node_modules", ".git", "__pycache__", "dist", "build", "venv", ".env"}

    def parse_github_url(self, url: str) -> Dict[str, str]:
        """Extract owner and repo from GitHub URL."""
        path = urlparse(url).path.strip("/")
        parts = path.split("/")
        if len(parts) >= 2:
            return {"owner": parts[0], "repo": parts[1]}
        raise ValueError("Invalid GitHub URL. Expected format: https://github.com/owner/repo")

    def fetch_repo_files(self, url: str, branch: str = "main") -> List[Dict[str, str]]:
        """List all relevant files in a repo using GitHub Tree API."""
        details = self.parse_github_url(url)
        owner, repo = details["owner"], details["repo"]
        
        # Get the tree recursively
        tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        response = requests.get(tree_url, headers=self.headers)
        
        if response.status_code != 200:
            # Try 'master' if 'main' fails
            if branch == "main":
                return self.fetch_repo_files(url, branch="master")
            raise Exception(f"Failed to fetch repo tree: {response.json().get('message', 'Unknown error')}")

        tree = response.json().get("tree", [])
        files = []
        for item in tree:
            if item["type"] == "blob":
                path = item["path"]
                ext = os.path.splitext(path)[1].lower()
                
                # Filter by extension and ignore directories
                if ext in self.allowed_extensions:
                    if not any(ignored in path.split("/") for ignored in self.ignored_dirs):
                        files.append({
                            "path": path,
                            "url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}",
                            "repo": f"{owner}/{repo}",
                            "filename": os.path.basename(path)
                        })
        return files

    def fetch_file_content(self, raw_url: str) -> str:
        """Fetch the content of a file from its raw URL."""
        response = requests.get(raw_url, headers=self.headers)
        if response.status_code == 200:
            return response.text
        raise Exception(f"Failed to fetch file content from {raw_url}")

if __name__ == "__main__":
    # Simple test
    loader = GitHubLoader()
    try:
        test_url = "https://github.com/tiangolo/fastapi"
        print(f"Fetching files for {test_url}...")
        files = loader.fetch_repo_files(test_url)
        print(f"Found {len(files)} relevant files.")
        if files:
            first_file = files[0]
            print(f"Fetching content for: {first_file['path']}")
            content = loader.fetch_file_content(first_file["url"])
            print(f"Content length: {len(content)} chars")
            print(f"Snippet: {content[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
