# Simple Fast Solution - Direct pandas analysis
import requests
import json
import pandas as pd
import time

def quick_analysis():
    """Quick analysis using direct pandas operations"""
    print("⚡ Starting quick analysis...")
    
    # Load data
    df = pd.read_excel(r"D:\Python code\ollama\GBM_V1.11.xlsx")
    print(f"✅ Loaded {len(df)} rows of data")
    
    # Quick pandas analysis
    print("\n📊 Quick Analysis Results:")
    print(f"Total patients: {len(df)}")
    print(f"Average age: {df['AGE'].mean():.1f} years")
    print(f"Gender distribution:")
    print(df['SEX'].value_counts())
    print(f"Alive vs Deceased:")
    print(df['DEATH/ ALIVE (7 AUG 2024)'].value_counts())
    
    return df

def simple_model_query(question, df):
    """Simple model query for specific questions"""
    try:
        # Prepare a simple prompt
        prompt = f"""
        Based on this medical dataset with {len(df)} patients:
        - Average age: {df['AGE'].mean():.1f}
        - Gender distribution: {dict(df['SEX'].value_counts())}
        - Alive/Deceased: {dict(df['DEATH/ ALIVE (7 AUG 2024)'].value_counts())}
        
        Question: {question}
        Answer briefly:
        """
        
        # Simple API call
        payload = {
            "model": "tinyllama:latest",  # Use fastest model
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_ctx": 256,  # Very small context
                "num_thread": 2,  # Few threads
                "temperature": 0.1  # Low temperature
            }
        }
        
        print(f"🔍 Asking: {question}")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=30  # Short timeout
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("message", {}).get("content", "No response")
            print(f"📊 Answer: {content}")
            print(f"⏱️ Time: {end_time - start_time:.1f} seconds")
            return True
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Main function
def run_fast_analysis():
    """Run fast analysis"""
    print("🚀 Starting Fast Analysis...")
    
    # Load and analyze data
    df = quick_analysis()
    
    # Simple model queries
    questions = [
        "How many patients are there?",
        "What is the average age?",
        "How many are male vs female?",
        "How many are still alive?"
    ]
    
    print("\n🧠 Simple Model Queries:")
    for question in questions:
        simple_model_query(question, df)
        print("-" * 50)
    
    print("\n✅ Fast analysis complete!")
    print("💡 You can now ask specific questions using:")
    print("   simple_model_query('Your question here', df)")

if __name__ == "__main__":
    run_fast_analysis() 