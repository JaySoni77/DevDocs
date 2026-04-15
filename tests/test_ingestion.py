import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.github_loader import GitHubLoader
from ingestion.chunker import Chunker
from ingestion.guards import IngestionGuards

def run_test_ingestion(repo_url: str):
    print(f"--- Starting Ingestion Test for {repo_url} ---")
    
    loader = GitHubLoader()
    chunker = Chunker()
    guards = IngestionGuards()

    try:
        # Step 1: Validate URL
        print("[1] Validating URL...")
        guards.validate_repo_url(repo_url)

        # Step 2: Fetch File List
        print("[2] Fetching file list from GitHub...")
        files = loader.fetch_repo_files(repo_url)
        print(f"    Found {len(files)} relevant files.")

        # Step 3: Validate File Count
        print("[3] Checking file counts and guards...")
        guards.validate_file_count(files)

        # Step 4: Fetch and Chunk Sample Files (top 3)
        print("[4] Fetching and chunking top 3 files...")
        for file_info in files[:3]:
            # Path guard
            guards.validate_path(file_info["path"])
            
            print(f"    Processing: {file_info['path']}...")
            content = loader.fetch_file_content(file_info["url"])
            chunks = chunker.process_file(file_info["path"], content)
            
            print(f"    - Generated {len(chunks)} chunks.")
            if chunks:
                first_chunk = chunks[0]
                print(f"    - First Chunk [{first_chunk['header']}]: {first_chunk['tokens']} tokens")
                print(f"      Snippet: {first_chunk['content'][:100]}...")

        print("\n--- Ingestion Test Completed Successfully ---")

    except ValueError as ve:
        print(f"\nGuard Violation: {ve}")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/tiangolo/fastapi"
    run_test_ingestion(url)
