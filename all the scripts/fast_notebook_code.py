# ===== FAST MISTRAL NOTEBOOK CODE =====
# Copy this into your data.ipynb for faster responses

# ===== CELL 1: Import and Fast Setup =====
import requests
import json
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

print("⚡ Setting up Fast Mistral Configuration...")

# Check available models
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        data = response.json()
        models = [model['name'] for model in data.get('models', [])]
        print("📋 Available models:")
        for model in models:
            print(f"  - {model}")
        
        # Choose fastest available model
        fast_models = ['llama3.2:3b', 'phi3:mini', 'llama3.2:1b', 'mistral:latest']
        selected_model = None
        
        for model in fast_models:
            if model in models:
                selected_model = model
                break
        
        if not selected_model:
            selected_model = 'mistral:latest'
        
        print(f"🚀 Using fastest model: {selected_model}")
        
        # Configure with speed optimizations
        llm = LiteLLM(
            model=f"ollama/{selected_model}",
            api_base="http://localhost:11434"
        )
        
        # Speed-optimized PandasAI settings
        pai.config.set({
            "llm": llm,
            "verbose": False,
            "enforce_privacy": False,
            "max_retries": 1,
            "enable_cache": True,  # Cache for faster repeated queries
            "save_charts": False,  # Disable chart saving
            "save_logs": False,    # Disable log saving
        })
        
        print("✅ Fast configuration complete!")
        
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
print("✅ Data loaded and ready for fast analysis!")

# ===== CELL 3: Fast Query Test =====
# Test with a simple, fast query
try:
    print("🔍 Testing fast query...")
    response = df1.chat("Count the number of rows")
    print(f"📊 Response: {response}")
    print("✅ Fast query successful!")
except Exception as e:
    print(f"⚠️ Query test failed: {e}")

# ===== CELL 4: Ready for Fast Analysis =====
print("\n⚡ Ready for fast analysis!")
print("\n💡 Fast query examples:")
print("- df1.chat('Count total patients')")
print("- df1.chat('Count patients by gender')")
print("- df1.chat('What is the average age?')")
print("- df1.chat('How many patients are alive?')")
print("- df1.chat('Count by surgery type')")

# ===== CELL 5: Manual Fast Query =====
# Change the question for your specific analysis
try:
    question = "Count total patients"
    print(f"🔍 Fast query: {question}")
    
    response = df1.chat(question)
    print(f"📊 Answer: {response}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# ===== CELL 6: Batch Fast Queries =====
# Multiple fast queries for comprehensive analysis
print("🔍 Running batch of fast queries...")

fast_queries = [
    "Count total patients",
    "Count patients by gender", 
    "What is the average age?",
    "How many patients are alive?",
    "Count by surgery type"
]

for i, query in enumerate(fast_queries, 1):
    print(f"\n📋 Query {i}: {query}")
    try:
        response = df1.chat(query)
        print(f"📊 Answer: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n✅ Fast analysis complete!") 