# Fast Mistral Setup - Optimized for Speed
import requests
import json
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

def check_available_models():
    """Check what models are available for faster processing"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            print("📋 Available models:")
            for model in models:
                print(f"  - {model}")
            return models
        return []
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return []

def setup_fast_mistral():
    """Setup with optimized settings for speed"""
    try:
        # Use faster model if available, otherwise use Mistral with optimized settings
        available_models = check_available_models()
        
        # Prefer smaller models for speed
        fast_models = ['llama3.2:3b', 'llama3.2:1b', 'phi3:mini', 'mistral:latest']
        selected_model = None
        
        for model in fast_models:
            if model in available_models:
                selected_model = model
                break
        
        if not selected_model:
            selected_model = 'mistral:latest'
        
        print(f"🚀 Using model: {selected_model}")
        
        # Configure LiteLLM with speed optimizations
        llm = LiteLLM(
            model=f"ollama/{selected_model}",
            api_base="http://localhost:11434"
        )
        
        # Configure PandasAI with speed optimizations
        pai.config.set({
            "llm": llm,
            "verbose": False,
            "enforce_privacy": False,
            "max_retries": 1,
            "enable_cache": True,  # Enable cache for faster repeated queries
            "save_charts": False,  # Disable chart saving
            "save_logs": False,    # Disable log saving
        })
        
        print("✅ PandasAI configured for speed!")
        return True
        
    except Exception as e:
        print(f"❌ Error configuring PandasAI: {e}")
        return False

def load_data_optimized():
    """Load data with optimizations"""
    try:
        # Load Excel file
        df = pd.read_excel(r"D:\Python code\ollama\GBM_V1.11.xlsx")
        
        # Clean column names
        df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        # Create PandasAI DataFrame with optimizations
        df1 = pai.DataFrame(df)
        print("✅ Data loaded and optimized!")
        
        return df1
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def test_fast_query(df1):
    """Test with a simple, fast query"""
    try:
        print("🔍 Testing fast query...")
        response = df1.chat("Count the number of rows")
        print(f"📊 Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

# Main setup
def setup_fast_notebook():
    """Complete fast setup"""
    print("⚡ Setting up Fast Mistral Configuration...")
    
    # Test connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama not accessible")
            return None
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None
    
    # Setup PandasAI
    if not setup_fast_mistral():
        return None
    
    # Load data
    df1 = load_data_optimized()
    if df1 is None:
        return None
    
    # Test query
    test_fast_query(df1)
    
    print("\n⚡ Fast setup complete!")
    print("💡 Tips for faster responses:")
    print("   - Ask simple, specific questions")
    print("   - Use 'count', 'sum', 'average' instead of complex analysis")
    print("   - Avoid asking for charts or visualizations")
    print("   - Try: df1.chat('Count patients by gender')")
    
    return df1

# For direct execution
if __name__ == "__main__":
    df1 = setup_fast_notebook()
    if df1 is not None:
        print("\n🎉 Ready for fast analysis!")
        print("Try: df1.chat('Count total patients')") 