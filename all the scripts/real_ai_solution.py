# Real AI Solution - Works with any dataset and any question
import requests
import json
import pandas as pd
import time

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

def get_data_summary(df):
    """Get comprehensive data summary for any dataset"""
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_names": list(df.columns),
        "data_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Add sample data for context
    summary["sample_data"] = df.head(3).to_dict('records')
    
    return summary

def ask_ai_question(question, df, model="tinyllama:latest"):
    """Ask any question to AI about the dataset"""
    try:
        # Get comprehensive data summary
        summary = get_data_summary(df)
        
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
                "num_ctx": 1024,  # Larger context for complex analysis
                "num_thread": 4,
                "temperature": 0.3  # Slightly higher for better analysis
            }
        }
        
        print(f"🔍 Asking AI: {question}")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=60  # Longer timeout for complex analysis
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

def interactive_analysis():
    """Interactive analysis session"""
    print("🚀 Real AI Dataset Analysis")
    print("=" * 50)
    
    # Load dataset
    file_path = r"D:\Python code\ollama\GBM_V1.11.xlsx"  # Change this to your file
    df = load_any_dataset(file_path)
    
    if df is None:
        return
    
    print("\n💡 You can ask any question about your dataset!")
    print("Examples:")
    print("- 'What are the main patterns in this data?'")
    print("- 'Is there a relationship between age and survival?'")
    print("- 'What are the key insights from this dataset?'")
    print("- 'Analyze the gender distribution and its implications'")
    print("- 'What factors might influence patient outcomes?'")
    
    # Ask first question
    first_question = "What are the main patterns and insights in this dataset?"
    print(f"\n🔍 First analysis: {first_question}")
    ask_ai_question(first_question, df)
    
    # Interactive questions
    while True:
        print("\n" + "="*50)
        user_question = input("Ask a question about your data (or type 'quit' to exit): ")
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_question.strip():
            ask_ai_question(user_question, df)
        else:
            print("Please enter a question.")

# Main execution
if __name__ == "__main__":
    interactive_analysis() 