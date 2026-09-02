import json
import os
import urllib.request
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GLM_API_KEY")

if not API_KEY:
    raise RuntimeError("GLM_API_KEY not found")

URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

payload = {
    "model": "glm-4.5-flash",
    "messages": [
        {
            "role": "user",
            "content": "Explain Python in one short sentence."
        }
    ],
    "temperature": 0.2,
}

request = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    },
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    print("=" * 60)
    print("GLM API CONNECTION SUCCESS")
    print("=" * 60)
    print("Model:", payload["model"])
    print()
    print("Response:")
    print(data["choices"][0]["message"]["content"])

except Exception as exc:
    print("GLM API request failed:")
    print(type(exc).__name__, exc)