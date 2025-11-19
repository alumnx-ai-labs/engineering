import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"Testing API key: {api_key[:10]}...")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    response = model.generate_content("Say 'Hello, MLOps!'")
    print(f"✅ API Key is VALID!")
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"❌ API Key is INVALID!")
    print(f"Error: {str(e)}")