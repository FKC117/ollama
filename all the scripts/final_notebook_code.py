# ===== FINAL WORKING CODE FOR JUPYTER NOTEBOOK =====
# Copy each cell below into your data.ipynb notebook

# ===== CELL 1: Import Libraries =====
import requests
import json
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

print("📦 Libraries imported successfully!")

# ===== CELL 2: Test Connection and Setup PandasAI =====
def test_ollama_connection():
    """Test connection to Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama is accessible!")
            return True
        else:
            print(f"❌ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

# Test connection
if test_ollama_connection():
    try:
        # Configure LiteLLM with Mistral
        llm = LiteLLM(
            model="ollama/mistral:latest",
            api_base="http://localhost:11434"
        )
        
        # Configure PandasAI with optimized settings
        pai.config.set({
            "llm": llm,
            "verbose": False,  # Reduce verbosity
            "enforce_privacy": False,
            "max_retries": 1,  # Reduce retries
            "enable_cache": False
        })
        
        print("✅ PandasAI configured with Mistral!")
        
    except Exception as e:
        print(f"❌ Error configuring PandasAI: {e}")
else:
    print("❌ No connection to Ollama. Make sure it's running in WSL:")
    print("   wsl")
    print("   ollama serve")

# ===== CELL 3: Load and Prepare Your Data =====
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

# ===== CELL 4: Test Simple Query =====
# Test a simple query to verify everything is working
try:
    print("🔍 Testing simple query...")
    response = df1.chat("How many rows are in this dataset?")
    print(f"📊 Response: {response}")
    print("✅ Query test successful!")
except Exception as e:
    print(f"⚠️ Query test failed: {e}")
    print("💡 You can still try manual queries below")

# ===== CELL 5: Ready for Analysis =====
print("\n🎉 Setup complete! You can now analyze your medical data.")
print("\nExample queries you can try:")
print("- df1.chat('What is the median age of patients?')")
print("- df1.chat('How many patients are male vs female?')")
print("- df1.chat('What is the average tumor size?')")
print("- df1.chat('How many patients are still alive?')")
print("- df1.chat('What is the most common surgery type?')")

# ===== CELL 6: Manual Query Example =====
# You can run this cell to test a specific query
# Just change the question in the quotes

try:
    # Example query - change this to any question you want
    question = "What is the median age of patients?"
    print(f"🔍 Asking: {question}")
    
    response = df1.chat(question)
    print(f"📊 Answer: {response}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Try a different question or check if Ollama is still running")

# ===== CELL 7: Advanced Analysis (Optional) =====
# This cell contains multiple complex queries for comprehensive analysis

print("🔍 Running comprehensive medical data analysis...")

queries = [
    "What is the distribution of patients by gender?",
    "What is the average age of patients?",
    "How many patients are still alive vs deceased?",
    "What is the most common surgery type?",
    "What is the average tumor size?",
    "How many patients received radiation therapy?",
    "What is the survival rate analysis?"
]

for i, query in enumerate(queries, 1):
    print(f"\n📋 Query {i}: {query}")
    try:
        response = df1.chat(query)
        print(f"📊 Answer: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Moving to next query...")

print("\n✅ Analysis complete!") 