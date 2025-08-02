# Smart Model Selector - Uses best model for each task
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

def setup_smart_config():
    """Setup with smart model selection"""
    print("🧠 Setting up Smart Model Configuration...")
    
    available_models = get_available_models()
    if not available_models:
        print("❌ No models available")
        return None
    
    # For data analysis, prefer more capable models
    analysis_models = ['llama3:latest', 'mistral:latest', 'tinyllama:latest']
    selected_model = None
    
    for model in analysis_models:
        if model in available_models:
            selected_model = model
            break
    
    if not selected_model:
        selected_model = available_models[0]
    
    print(f"🧠 Selected model for analysis: {selected_model}")
    
    try:
        # Configure LiteLLM
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
        return selected_model
        
    except Exception as e:
        print(f"❌ Error configuring: {e}")
        return None

def load_data_smart():
    """Load data with smart optimizations"""
    try:
        df = pd.read_excel(r"D:\Python code\ollama\GBM_V1.11.xlsx")
        df.columns = df.columns.str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        df1 = pai.DataFrame(df)
        print("✅ Data loaded with smart optimizations!")
        return df1
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def test_smart_query(df1, model_name):
    """Test with smart query"""
    try:
        print(f"🧠 Testing {model_name} with smart query...")
        response = df1.chat("How many patients are in this dataset?")
        print(f"📊 Response: {response}")
        print("✅ Smart query successful!")
        return True
    except Exception as e:
        print(f"⚠️ Query test failed: {e}")
        return False

# Main smart setup
def setup_smart_notebook():
    """Complete smart setup"""
    print("🧠 Setting up Smart Model Configuration...")
    
    # Test connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama not accessible")
            return None
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None
    
    # Setup with smart model selection
    model_name = setup_smart_config()
    if not model_name:
        return None
    
    # Load data
    df1 = load_data_smart()
    if df1 is None:
        return None
    
    # Test query
    test_smart_query(df1, model_name)
    
    print(f"\n🧠 Smart setup complete with {model_name}!")
    print("💡 Smart query tips:")
    print("   - Use natural language: 'How many patients are there?'")
    print("   - Ask specific questions: 'What is the average age?'")
    print("   - Request analysis: 'Show me gender distribution'")
    print("   - Try: df1.chat('How many patients are male?')")
    print("   - Try: df1.chat('What is the average age of patients?')")
    print("   - Try: df1.chat('How many patients are still alive?')")
    
    return df1

# For direct execution
if __name__ == "__main__":
    df1 = setup_smart_notebook()
    if df1 is not None:
        print("\n🎉 Ready for smart analysis!")
        print("Try: df1.chat('How many patients are there?')") 