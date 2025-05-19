import os
from dotenv import load_dotenv

load_dotenv()
print("✅ Loaded API key:", os.getenv("GOOGLE_API_KEY"))
