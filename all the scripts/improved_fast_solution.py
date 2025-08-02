# Improved Fast Solution - Better data context
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

def improved_model_query(question, df):
    """Improved model query with better data context"""
    try:
        # Get actual data statistics
        total_patients = len(df)
        avg_age = df['AGE'].mean()
        gender_counts = df['SEX'].value_counts().to_dict()
        alive_counts = df['DEATH/ ALIVE (7 AUG 2024)'].value_counts().to_dict()
        
        # Create a clear, factual prompt
        prompt = f"""
        I have a medical dataset with EXACTLY {total_patients} patients.
        
        ACTUAL DATA:
        - Total patients: {total_patients}
        - Average age: {avg_age:.1f} years
        - Gender: {gender_counts}
        - Alive/Deceased: {alive_counts}
        
        Question: {question}
        
        Answer based ONLY on the data above. Be brief and accurate.
        """
        
        # Simple API call
        payload = {
            "model": "tinyllama:latest",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_ctx": 512,
                "num_thread": 2,
                "temperature": 0.1
            }
        }
        
        print(f"🔍 Asking: {question}")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=30
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

def direct_pandas_answer(question, df):
    """Direct pandas answer without model"""
    print(f"🔍 Direct pandas answer for: {question}")
    
    if "how many patients" in question.lower():
        print(f"📊 Answer: There are {len(df)} patients in the dataset.")
    
    elif "average age" in question.lower():
        avg_age = df['AGE'].mean()
        print(f"📊 Answer: The average age is {avg_age:.1f} years.")
    
    elif "male" in question.lower() and "female" in question.lower():
        gender_counts = df['SEX'].value_counts()
        print(f"📊 Answer: Male: {gender_counts.get('Male', 0)}, Female: {gender_counts.get('Female', 0)}")
    
    elif "alive" in question.lower():
        alive_counts = df['DEATH/ ALIVE (7 AUG 2024)'].value_counts()
        print(f"📊 Answer: Alive status: {dict(alive_counts)}")
    
    else:
        print("📊 Answer: Please ask a specific question about patient count, age, gender, or survival status.")

# Main function
def run_improved_analysis():
    """Run improved analysis"""
    print("🚀 Starting Improved Analysis...")
    
    # Load and analyze data
    df = quick_analysis()
    
    print("\n🧠 Model Queries (with improved context):")
    questions = [
        "How many patients are there?",
        "What is the average age?",
        "How many are male vs female?",
        "How many are still alive?"
    ]
    
    for question in questions:
        improved_model_query(question, df)
        print("-" * 50)
    
    print("\n📊 Direct Pandas Answers (instant):")
    for question in questions:
        direct_pandas_answer(question, df)
        print("-" * 30)
    
    print("\n✅ Improved analysis complete!")
    print("💡 Use direct_pandas_answer() for instant results!")
    print("💡 Use improved_model_query() for AI interpretation!")

if __name__ == "__main__":
    run_improved_analysis() 