# Simple Mistral Setup for Jupyter Notebook
import requests
import json
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

def quick_connection_test():
    """Quick test to verify Ollama is accessible"""
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

def setup_pandasai_mistral():
    """Setup PandasAI with Mistral - simplified version"""
    try:
        # Configure LiteLLM with Mistral
        llm = LiteLLM(
            model="ollama/mistral:latest",
            api_base="http://localhost:11434"
        )
        
        # Configure PandasAI with minimal settings
        pai.config.set({
            "llm": llm,
            "verbose": False,  # Reduce verbosity
            "enforce_privacy": False,
            "max_retries": 1,  # Reduce retries
            "enable_cache": False
        })
        
        print("✅ PandasAI configured with Mistral!")
        return True
        
    except Exception as e:
        print(f"❌ Error configuring PandasAI: {e}")
        return False

def load_and_prepare_data():
    """Load and prepare the medical data"""
    try:
        # Load Excel file
        df = pd.read_excel(r"D:\Python code\ollama\GBM_V1.11.xlsx")
        
        print("Original column names:")
        print(df.columns.tolist())
        
        # Clean column names
        df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        print("\nCleaned column names:")
        print(df.columns.tolist())
        
        # Create PandasAI DataFrame
        df1 = pai.DataFrame(df)
        print("\n✅ PandasAI DataFrame created successfully!")
        
        return df1
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def test_simple_query(df1):
    """Test a simple query with the data"""
    try:
        print("🔍 Testing simple query...")
        response = df1.chat("How many rows are in this dataset?")
        print(f"📊 Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

# Main setup function
def setup_for_notebook():
    """Complete setup for Jupyter notebook"""
    print("🚀 Setting up Mistral for Jupyter Notebook...")
    
    # Step 1: Test connection
    if not quick_connection_test():
        print("💡 Make sure Ollama is running in WSL:")
        print("   wsl")
        print("   ollama serve")
        return None
    
    # Step 2: Setup PandasAI
    if not setup_pandasai_mistral():
        return None
    
    # Step 3: Load data
    df1 = load_and_prepare_data()
    if df1 is None:
        return None
    
    # Step 4: Test query
    if not test_simple_query(df1):
        print("⚠️ Setup complete but query test failed. You can still try manual queries.")
    
    print("✅ Setup complete! You can now use df1.chat() for queries.")
    return df1

# For direct execution
if __name__ == "__main__":
    df1 = setup_for_notebook()
    if df1 is not None:
        print("\n🎉 Ready to analyze medical data with Mistral!")
        print("Try: df1.chat('What is the median age of patients?')") 