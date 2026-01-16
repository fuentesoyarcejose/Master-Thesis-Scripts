import os
from google import genai

# Use the key from the original script or env var
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyB8wv4vhyxvbFdJ4YSK7gBa28YI9UrApjk")

def list_models():
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        # The new SDK might have a different way to list models, 
        # but often it's client.models.list() or similar.
        # Let's try the standard way for the new SDK if possible, 
        # or fall back to what we can guess.
        
        print("Attempting to list models...")
        # Based on typical patterns for this SDK
        models = client.models.list() 
        
        print("Available models:")
        for m in models:
            print(f"- {m.name}")
            
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_models()
