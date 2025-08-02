# ===== REAL AI SOLUTION FOR NOTEBOOK =====
# Copy this into your notebook for actual AI analysis

# ===== CELL 1: Import and Setup =====
import requests
import json
import pandas as pd
import time

print("🚀 Real AI Analysis Setup Complete!")

# ===== CELL 2: Load Any Dataset =====
def load_any_dataset(file_path):
    """Load any dataset (Excel, CSV, etc.)"""
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print("❌ Unsupported file format. Use Excel (.xlsx) or CSV (.csv)")
            return None
        
        print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        print(f"📋 Columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

# Load your dataset
file_path = r"D:\Python code\ollama\GBM_V1.11.xlsx"  # Change this to your file
df = load_any_dataset(file_path)

# ===== CELL 3: AI Analysis Function =====
def ask_ai_question(question, df, model="tinyllama:latest"):
    """Ask any question to AI about the dataset"""
    try:
        # Get comprehensive data summary
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "sample_data": df.head(3).to_dict('records')
        }
        
        # Create detailed prompt with actual data
        prompt = f"""
        I have a dataset with the following characteristics:
        
        DATASET INFO:
        - Total rows: {summary['total_rows']}
        - Total columns: {summary['total_columns']}
        - Column names: {summary['column_names']}
        - Data types: {summary['data_types']}
        - Missing values: {summary['missing_values']}
        - Numeric columns: {summary['numeric_columns']}
        - Categorical columns: {summary['categorical_columns']}
        
        SAMPLE DATA (first 3 rows):
        {summary['sample_data']}
        
        USER QUESTION: {question}
        
        Please analyze this dataset and provide insights based on the actual data. 
        Be specific and reference the actual column names and data.
        """
        
        # API call to model
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_ctx": 1024,
                "num_thread": 4,
                "temperature": 0.3
            }
        }
        
        print(f"🔍 Asking AI: {question}")
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
            print(f"📊 AI Response: {content}")
            print(f"⏱️ Time: {end_time - start_time:.1f} seconds")
            return content
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ===== CELL 4: Test AI Analysis =====
# Test with a real question
if df is not None:
    question = "What are the main patterns and insights in this dataset?"
    print(f"\n🔍 Testing AI analysis: {question}")
    ask_ai_question(question, df)

# ===== CELL 5: Ready for Your Questions =====
print("\n🎉 Ready for AI Analysis!")
print("\n💡 You can now ask any question using:")
print("   ask_ai_question('Your question here', df)")
print("\nExample questions:")
print("- 'Is there a relationship between age and survival?'")
print("- 'What are the key factors affecting patient outcomes?'")
print("- 'Analyze the gender distribution and its implications'")
print("- 'What patterns do you see in the tumor size data?'")
print("- 'How do different surgery types relate to survival rates?'")

# ===== CELL 6: Ask Your Own Question =====
# Change this question to whatever you want to analyze
your_question = "Is there any relationship between age and survival?"
print(f"\n🔍 Your question: {your_question}")
ask_ai_question(your_question, df)

# ===== CELL 7: Multiple Questions =====
# Ask multiple questions in sequence
questions = [
    "What are the main patterns in this medical dataset?",
    "Is there a relationship between age and survival?",
    "What factors might influence patient outcomes?",
    "Analyze the gender distribution and its implications"
]

print("\n🔍 Running multiple AI analyses...")
for i, question in enumerate(questions, 1):
    print(f"\n📋 Question {i}: {question}")
    ask_ai_question(question, df)
    print("-" * 50)

print("\n✅ AI analysis complete!") 