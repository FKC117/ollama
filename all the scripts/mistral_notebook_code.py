# ===== CELL 1: Import and Setup =====
# Copy this into the first cell of your Jupyter notebook

import requests
import json
import time
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

print("📦 Libraries imported successfully!")

# ===== CELL 2: Test WSL Connection =====
# Copy this into the second cell

def test_wsl_connection():
    """Test connection to Ollama running in WSL"""
    
    endpoints = [
        "http://localhost:11434",  # Standard localhost
        "http://127.0.0.1:11434",  # Explicit localhost
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

# Test connection
api_base = test_wsl_connection()

# ===== CELL 3: Setup PandasAI with Mistral =====
# Copy this into the third cell

if api_base:
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
        
    except Exception as e:
        print(f"❌ Error configuring PandasAI: {e}")
else:
    print("❌ No connection to Ollama. Make sure it's running in WSL:")
    print("   wsl")
    print("   ollama serve")

# ===== CELL 4: Load and Prepare Your Data =====
# Copy this into the fourth cell

# Load your Excel file
df = pd.read_excel(r"D:\Python code\ollama\GBM_V1.11.xlsx")

print("Original column names:")
print(df.columns.tolist())

# Clean column names - remove special characters and spaces
df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)

print("\nCleaned column names:")
print(df.columns.tolist())

# Create PandasAI DataFrame
df1 = pai.DataFrame(df)
print("\n✅ PandasAI DataFrame created successfully!")

# ===== CELL 5: Test Mistral with Medical Data =====
# Copy this into the fifth cell

if api_base:
    try:
        print("🔍 Testing Mistral with medical data analysis...")
        
        # Test a simple query
        response = df1.chat("What is the median age of the patients?")
        print(f"📊 Response: {response}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
else:
    print("❌ No connection available")

# ===== CELL 6: Advanced Medical Data Analysis =====
# Copy this into the sixth cell (optional - for more complex queries)

if api_base:
    try:
        print("🔍 Running advanced medical data analysis...")
        
        # You can ask more complex questions
        queries = [
            "What is the distribution of patients by gender?",
            "What is the average tumor size?",
            "How many patients are still alive vs deceased?",
            "What is the most common surgery type?"
        ]
        
        for query in queries:
            print(f"\n📋 Query: {query}")
            try:
                response = df1.chat(query)
                print(f"📊 Answer: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Error during advanced analysis: {e}")
else:
    print("❌ No connection available")

# ===== CELL 7: Direct Mistral Chat Test =====
# Copy this into the seventh cell (optional - for direct chat testing)

def test_mistral_direct():
    """Test direct Mistral chat functionality"""
    
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
        print("🔍 Testing direct Mistral chat...")
        start_time = time.time()
        response = requests.post(
            f"{api_base}/api/chat",
            json=payload,
            timeout=60
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("message", {}).get("content", "No response")
            print(f"✅ Direct chat successful - {end_time - start_time:.2f} seconds")
            print(f"📝 Response: {content}")
            return True
        else:
            print(f"❌ Direct chat failed - HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Direct chat error: {e}")
        return False

if api_base:
    test_mistral_direct()
else:
    print("❌ No connection available") 