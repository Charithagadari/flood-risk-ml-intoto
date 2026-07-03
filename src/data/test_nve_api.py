import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVE_API_KEY")

if not API_KEY:
    raise ValueError("NVE_API_KEY not found. Add it to your .env file.")

BASE_URL = "https://hydapi.nve.no/api/v1"

headers = {
    "Accept": "application/json",
    "X-API-Key": API_KEY,
}

url = f"{BASE_URL}/Parameters"

response = requests.get(url, headers=headers, timeout=30)

print("Status code:", response.status_code)
print(response.text[:1000])