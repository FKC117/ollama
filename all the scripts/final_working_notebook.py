# ===== FINAL WORKING NOTEBOOK CODE =====
# Copy this into your data.ipynb for reliable analysis

# ===== CELL 1: Import and Smart Setup =====
import requests
import json
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

print("🧠 Setting up Smart Model Configuration...")

# Get available models and select best for analysis
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        data = response.json()
        models = [model['name'] for model in data.get('models', [])]
        print("📋 Available models:")
        for model in models:
            print(f"  - {model}")
        
        # Select best model for data analysis (capability over speed)
        analysis_models = ['llama3:latest', 'mistral:latest', 'tinyllama:latest']
        selected_model = None
        
        for model in analysis_models:
            if model in models:
                selected_model = model
                break
        
        if not selected_model:
            selected_model = models[0] if models else 'mistral:latest'
        
        print(f"🧠 Selected model for analysis: {selected_model}")
        
        # Configure with smart settings
        llm = LiteLLM(
            model=f"ollama/{selected_model}",
            api_base="http://localhost:11434"
        )
        
        # Smart PandasAI settings
        pai.config.set({
            "llm": llm,
            "verbose": False,
            "enforce_privacy": False,
            "max_retries": 2,
            "enable_cache": True,
            "save_charts": False,
            "save_logs": False,
        })
        
        print("✅ Smart configuration complete!")
        
    else:
        print("❌ Ollama not accessible")
        
except Exception as e:
    print(f"❌ Error: {e}")

# ===== CELL 2: Load Data =====
# Load your Excel file
df = pd.read_excel(r"D:\Python code\ollama\GBM_V1.11.xlsx")

# Clean column names
df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)

# Create PandasAI DataFrame
df1 = pai.DataFrame(df)
print("✅ Data loaded and ready for analysis!")

# ===== CELL 3: Test Query =====
# Test with a natural language query
try:
    print("🧠 Testing smart query...")
    response = df1.chat("How many patients are in this dataset?")
    print(f"📊 Response: {response}")
    print("✅ Smart query successful!")
except Exception as e:
    print(f"⚠️ Query test failed: {e}")

# ===== CELL 4: Ready for Analysis =====
print("\n🧠 Ready for smart analysis!")
print("\n💡 Smart query examples:")
print("- df1.chat('How many patients are there?')")
print("- df1.chat('What is the average age of patients?')")
print("- df1.chat('How many patients are male vs female?')")
print("- df1.chat('How many patients are still alive?')")
print("- df1.chat('What is the most common surgery type?')")

# ===== CELL 5: Manual Query =====
# Change the question for your specific analysis
try:
    question = "How many patients are there?"
    print(f"🧠 Smart query: {question}")
    
    response = df1.chat(question)
    print(f"📊 Answer: {response}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# ===== CELL 6: Batch Analysis =====
# Multiple queries for comprehensive analysis
print("🧠 Running comprehensive analysis...")

analysis_queries = [
    "How many patients are there?",
    "What is the average age of patients?", 
    "How many patients are male vs female?",
    "How many patients are still alive?",
    "What is the most common surgery type?"
]

for i, query in enumerate(analysis_queries, 1):
    print(f"\n📋 Query {i}: {query}")
    try:
        response = df1.chat(query)
        print(f"📊 Answer: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n✅ Analysis complete!") 