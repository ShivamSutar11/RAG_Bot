import sys

with open("rag_engine.py", "r") as f:
    text = f.read()

env_parser = """import os
import requests
import json

# Local .env parser
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k] = v
"""

# Remove the old parser block
text = text.replace("""import os
import requests
import json

# Local .env parser so we don't need python-dotenv
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k] = v""", "")

# Insert at the top
lines = text.split("\n")
lines.insert(2, env_parser)

with open("rag_engine.py", "w") as f:
    f.write("\n".join(lines))
