# ===== ULTRA FAST NOTEBOOK CODE =====
# Copy this into your data.ipynb for maximum speed

# ===== CELL 1: Import and Ultra-Fast Setup =====
import requests
import json
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

print("🚀 Setting up Ultra Fast Configuration...")

# Get available models and select fastest
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        data = response.json()
        models = [model['name'] for model in data.get('models', [])]
        print("📋 Available models:")
        for model in models:
            print(f"  - {model}")
        
        # Select fastest model (order by speed)
        speed_order = ['tinyllama:latest', 'llama3.2:1b', 'llama3.2:3b', 'llama3:latest', 'phi3:mini', 'mistral:latest']
        selected_model = None
        
        for model in speed_order:
            if model in models:
                selected_model = model
                break
        
        if not selected_model:
            selected_model = models[0] if models else 'mistral:latest'
        
        print(f"🚀 Selected fastest model: {selected_model}")
        
        # Configure with ultra-fast settings
        llm = LiteLLM(
            model=f"ollama/{selected_model}",
            api_base="http://localhost:11434"
        )
        
        # Ultra-fast PandasAI settings
        pai.config.set({
            "llm": llm,
            "verbose": False,
            "enforce_privacy": False,
            "max_retries": 1,
            "enable_cache": True,
            "save_charts": False,
            "save_logs": False,
        })
        
        print("✅ Ultra-fast configuration complete!")
        
    else:
        print("❌ Ollama not accessible")
        
except Exception as e:
    print(f"❌ Error: {e}")

# ===== CELL 2: Load Data Optimized =====
# Load your Excel file
df = pd.read_excel(r"D:\Python code\ollama\GBM_V1.11.xlsx")

# Clean column names
df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)

# Create PandasAI DataFrame
df1 = pai.DataFrame(df)
print("✅ Data loaded and ready for ultra-fast analysis!")

# ===== CELL 3: Ultra-Fast Query Test =====
# Test with ultra-fast query
try:
    print("🔍 Testing ultra-fast query...")
    response = df1.chat("Count rows")
    print(f"📊 Response: {response}")
    print("✅ Ultra-fast query successful!")
except Exception as e:
    print(f"⚠️ Query test failed: {e}")

# ===== CELL 4: Ready for Ultra-Fast Analysis =====
print("\n⚡ Ready for ultra-fast analysis!")
print("\n💡 Ultra-fast query examples:")
print("- df1.chat('Count patients')")
print("- df1.chat('Count by gender')")
print("- df1.chat('Sum age')")
print("- df1.chat('Average age')")
print("- df1.chat('Count alive')")

# ===== CELL 5: Manual Ultra-Fast Query =====
# Change the question for your specific analysis
try:
    question = "Count patients"
    print(f"🔍 Ultra-fast query: {question}")
    
    response = df1.chat(question)
    print(f"📊 Answer: {response}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# ===== CELL 6: Batch Ultra-Fast Queries =====
# Multiple ultra-fast queries for comprehensive analysis
print("🔍 Running batch of ultra-fast queries...")

ultra_fast_queries = [
    "Count patients",
    "Count by gender", 
    "Sum age",
    "Average age",
    "Count alive"
]

for i, query in enumerate(ultra_fast_queries, 1):
    print(f"\n📋 Query {i}: {query}")
    try:
        response = df1.chat(query)
        print(f"📊 Answer: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n✅ Ultra-fast analysis complete!") 