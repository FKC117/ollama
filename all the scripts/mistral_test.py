# Test Mistral model with optimized configuration
import requests
import json
import time

def test_mistral_optimized():
    """Test Mistral model with optimized configuration"""
    
    mistral_payload = {
        "model": "mistral:latest",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False,
        "options": {
            "num_ctx": 512,  # Smaller context for faster processing
            "num_thread": 4,  # Limit threads to reduce resource usage
            "temperature": 0.1,  # Lower temperature for more predictable responses
            "top_p": 0.9,  # Control response randomness
            "repeat_penalty": 1.1  # Prevent repetitive responses
        }
    }
    
    print("🔍 Testing Mistral:latest with optimized settings...")
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=mistral_payload,
            timeout=60  # Increased timeout for Mistral
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("message", {}).get("content", "No response")
            print(f"✅ Mistral:latest - Response time: {end_time - start_time:.2f} seconds")
            print(f"📝 Response: {content}")
            return True
        else:
            print(f"❌ Mistral:latest - HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Mistral:latest - Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Mistral:latest - Connection error. Is Ollama running?")
        return False
    except Exception as e:
        print(f"❌ Mistral:latest - Error: {e}")
        return False

# Test Mistral with optimized settings
if __name__ == "__main__":
    success = test_mistral_optimized()
    if success:
        print("\n✅ Mistral model is working with optimized settings!")
    else:
        print("\n❌ Mistral model failed with optimized settings") 