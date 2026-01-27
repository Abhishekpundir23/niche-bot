import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load keys
load_dotenv()

# 2. Connect to Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    print("Fetching available models from Groq...")
    # 3. List models
    models = client.models.list()
    
    print(f"\n{'MODEL ID':<40} | {'OWNER':<15}")
    print("-" * 60)
    
    # 4. Print them cleanly
    for model in models.data:
        print(f"{model.id:<40} | {model.owned_by:<15}")
        
except Exception as e:
    print(f"Error: {e}")