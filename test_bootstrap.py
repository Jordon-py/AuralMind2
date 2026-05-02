import server
import json
try:
    res = server.bootstrap()
    print("Bootstrap success!")
    print(f"Tools: {len(res.tools)}")
except Exception as e:
    print(f"Bootstrap failed: {e}")
