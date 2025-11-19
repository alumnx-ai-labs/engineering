import requests
import json

# Test 1: Short query
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "Hi",
        "user_id": "test1"
    }
)
print("Short Query Response:")
print(json.dumps(response.json(), indent=2))

# Test 2: Long query
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "Explain the architecture of a production ML system with all components",
        "user_id": "test2"
    }
)
print("\nLong Query Response:")
print(json.dumps(response.json(), indent=2))

# Test 3: Check metrics
metrics = requests.get("http://localhost:8000/metrics")
print("\nMetrics Dashboard:")
print(json.dumps(metrics.json(), indent=2))