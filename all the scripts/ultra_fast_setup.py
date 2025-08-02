# Ultra Fast Setup - Auto-selects fastest model
import requests
import json
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

def get_available_models():
    """Get list of available models"""
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

def select_fastest_model(available_models):
    """Select the fastest available model"""
    # Order by speed (fastest first)
    speed_order = [
        'tinyllama:latest',      # Fastest
        'llama3.2:1b',          # Very fast
        'llama3.2:3b',          # Fast
        'llama3:latest',         # Medium-fast
        'phi3:mini',             # Fast
        'mistral:latest'         # Slowest but most capable
    ]
    
    for model in speed_order:
        if model in available_models:
            return model
    
    # Fallback to first available
    return available_models[0] if available_models else 'mistral:latest'

def setup_ultra_fast():
    """Setup with the fastest available model"""
    print("⚡ Setting up Ultra Fast Configuration...")
    
    # Get available models
    available_models = get_available_models()
    if not available_models:
        print("❌ No models available")
        return None
    
    # Select fastest model
    fastest_model = select_fastest_model(available_models)
    print(f"🚀 Selected fastest model: {fastest_model}")
    
    try:
        # Configure LiteLLM with ultra-fast settings
        llm = LiteLLM(
            model=f"ollama/{fastest_model}",
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
            "custom_whitelisted_dependencies": [],  # No extra dependencies
        })
        
        print("✅ Ultra-fast configuration complete!")
        return fastest_model
        
    except Exception as e:
        print(f"❌ Error configuring: {e}")
        return None

def load_data_fast():
    """Load data with optimizations"""
    try:
        df = pd.read_excel(r"D:\Python code\ollama\GBM_V1.11.xlsx")
        df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        df1 = pai.DataFrame(df)
        print("✅ Data loaded and optimized!")
        return df1
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def test_ultra_fast_query(df1, model_name):
    """Test with ultra-fast query"""
    try:
        print(f"🔍 Testing {model_name} with ultra-fast query...")
        response = df1.chat("Count rows")
        print(f"📊 Response: {response}")
        print("✅ Ultra-fast query successful!")
        return True
    except Exception as e:
        print(f"⚠️ Query test failed: {e}")
        return False

# Main ultra-fast setup
def setup_ultra_fast_notebook():
    """Complete ultra-fast setup"""
    print("🚀 Setting up Ultra Fast Mistral Configuration...")
    
    # Test connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama not accessible")
            return None
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None
    
    # Setup with fastest model
    model_name = setup_ultra_fast()
    if not model_name:
        return None
    
    # Load data
    df1 = load_data_fast()
    if df1 is None:
        return None
    
    # Test query
    test_ultra_fast_query(df1, model_name)
    
    print(f"\n⚡ Ultra-fast setup complete with {model_name}!")
    print("💡 Ultra-fast query tips:")
    print("   - Use 'Count' instead of 'How many'")
    print("   - Use 'Sum' instead of 'Total'")
    print("   - Use 'Average' instead of 'Mean'")
    print("   - Try: df1.chat('Count patients')")
    print("   - Try: df1.chat('Sum age')")
    print("   - Try: df1.chat('Average age')")
    
    return df1

# For direct execution
if __name__ == "__main__":
    df1 = setup_ultra_fast_notebook()
    if df1 is not None:
        print("\n🎉 Ready for ultra-fast analysis!")
        print("Try: df1.chat('Count patients')") 