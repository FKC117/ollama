# Mistral Connection for Jupyter Notebook (WSL Configuration)
import requests
import json
import time
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

def test_wsl_connection():
    """Test connection to Ollama running in WSL"""
    
    # WSL typically exposes services on localhost, but let's test different endpoints
    endpoints = [
        "http://localhost:11434",  # Standard localhost
        "http://127.0.0.1:11434",  # Explicit localhost
        "http://172.17.0.1:11434",  # Docker bridge network
    ]
    
    for endpoint in endpoints:
        try:
            print(f"🔍 Testing connection to {endpoint}...")
            response = requests.get(f"{endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Connected to Ollama at {endpoint}")
                return endpoint
        except Exception as e:
            print(f"❌ Failed to connect to {endpoint}: {e}")
    
    return None

def setup_mistral_pandasai(api_base="http://localhost:11434"):
    """Setup PandasAI with Mistral model"""
    
    try:
        # Configure LiteLLM to use Ollama with Mistral
        llm = LiteLLM(
            model="ollama/mistral:latest",
            api_base=api_base
        )
        
        # Configure PandasAI with optimized settings
        pai.config.set({
            "llm": llm,
            "verbose": True,
            "enforce_privacy": False,
            "max_retries": 2,
            "enable_cache": False
        })
        
        print("✅ PandasAI configured with Mistral!")
        return True
        
    except Exception as e:
        print(f"❌ Error configuring PandasAI: {e}")
        return False

def test_mistral_chat():
    """Test Mistral chat functionality"""
    
    payload = {
        "model": "mistral:latest",
        "messages": [{"role": "user", "content": "Hello, can you help me analyze medical data?"}],
        "stream": False,
        "options": {
            "num_ctx": 512,
            "num_thread": 4,
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }
    
    try:
        print("🔍 Testing Mistral chat...")
        start_time = time.time()
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=60
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("message", {}).get("content", "No response")
            print(f"✅ Mistral chat test successful - {end_time - start_time:.2f} seconds")
            print(f"📝 Response: {content[:200]}...")
            return True
        else:
            print(f"❌ Chat test failed - HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chat test error: {e}")
        return False

# Main setup function for Jupyter notebook
def setup_mistral_for_notebook():
    """Complete setup for Jupyter notebook with Mistral"""
    
    print("🚀 Setting up Mistral for Jupyter Notebook...")
    
    # Test WSL connection
    api_base = test_wsl_connection()
    if not api_base:
        print("❌ Could not connect to Ollama in WSL")
        print("💡 Make sure Ollama is running in WSL:")
        print("   wsl")
        print("   ollama serve")
        return None
    
    # Setup PandasAI
    if not setup_mistral_pandasai(api_base):
        return None
    
    # Test chat functionality
    if not test_mistral_chat():
        return None
    
    print("✅ Mistral is ready for use in Jupyter notebook!")
    return api_base

# Example usage for Jupyter notebook
if __name__ == "__main__":
    setup_mistral_for_notebook() 