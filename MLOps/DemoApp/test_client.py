import requests
import json

BASE_URL = "http://localhost:8000"

def test_chat():
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "message": "Explain MLOps in one sentence",
            "user_id": "student_1"
        }
    )
    print("Chat Response:")
    print(json.dumps(response.json(), indent=2))

def view_metrics():
    response = requests.get(f"{BASE_URL}/metrics")
    print("\nMLOps Metrics Dashboard:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    # Test 1: Send a message
    test_chat()
    
    # Test 2: Check metrics
    view_metrics()