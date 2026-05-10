"""
Quick test script to verify the pets classification API is working.
"""
import requests
import time

# Wait for server to start
print("Waiting for ML server to start...")
for i in range(30):
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print(f"✓ Server is running! ({i+1} attempts)")
            break
    except requests.exceptions.ConnectionError:
        time.sleep(1)
else:
    print("✗ Server did not start in time")
    exit(1)

# Test health endpoint
print("\n--- Testing /health endpoint ---")
response = requests.get("http://localhost:8000/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

# Test pets classification with a cat image
print("\n--- Testing /pets/predict endpoint ---")
test_image = "cats_set/cat.4001.jpg"
with open(test_image, "rb") as f:
    files = {"file": (test_image, f, "image/jpeg")}
    response = requests.post("http://localhost:8000/pets/predict", files=files)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

print("\n✓ All tests completed!")
