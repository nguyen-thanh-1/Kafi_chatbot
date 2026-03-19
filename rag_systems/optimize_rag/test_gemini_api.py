import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ Error: GEMINI_API_KEY not found in .env")
    exit(1)

print(f"✅ API Key found: {API_KEY[:10]}...")

try:
    # Configure the Gemini API
    genai.configure(api_key=API_KEY)
    
    # Use a lightweight model for testing
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Attempt to generate a simple response
    print("🔄 Sending test request to Gemini...")
    response = model.generate_content("Say 'Gemini is working!'")
    
    print(f"🤖 Response: {response.text}")
    print("✅ Gemini API is working correctly.")
except Exception as e:
    print(f"❌ Error during API call: {e}")
