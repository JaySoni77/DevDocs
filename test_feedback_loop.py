import json
import os
from datetime import datetime

# Mock the Gradio LikeData for testing the log_feedback function directly
class MockLikeData:
    def __init__(self, index, value, liked):
        self.index = index
        self.value = value
        self.liked = liked

def log_feedback_test(data):
    """Mirror of the function in app.py for standalone verification."""
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "repo": "pypa/sampleproject",
        "index": data.index,
        "value": data.value,
        "rating": 1 if data.liked else 0
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/feedback.jsonl", "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    print(f"Feedback logged: {feedback_entry}")

if __name__ == "__main__":
    # Simulate a 'liked' event
    mock_data = MockLikeData(index=[0, 1], value="liked", liked=True)
    log_feedback_test(mock_data)
    
    # Verify file exists and has content
    if os.path.exists("logs/feedback.jsonl"):
        print("Success: logs/feedback.jsonl exists.")
        with open("logs/feedback.jsonl", "r") as f:
            lines = f.readlines()
            print(f"Total feedback entries: {len(lines)}")
            print(f"Last entry: {lines[-1].strip()}")
    else:
        print("Failure: logs/feedback.jsonl not found.")
