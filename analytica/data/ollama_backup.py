# Ollama/Llama Backup Code
# This file contains the Ollama integration code that was removed from views.py
# Keep this for future reference or if you want to switch back to Ollama

import requests
import json

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ DEBUG - Ollama connection test successful!")
            print(f"🤖 DEBUG - Available models: {[m['name'] for m in models]}")
            return True, models
        else:
            print(f"❌ DEBUG - Ollama connection test failed: HTTP {response.status_code}")
            return False, []
    except Exception as e:
        print(f"❌ DEBUG - Ollama connection test error: {e}")
        return False, []

def generate_ai_response(context):
    """Generate AI response using Ollama AI only"""
    try:
        # Test Ollama connection first
        print(f"🔍 DEBUG - Testing Ollama connection...")
        is_connected, available_models = test_ollama_connection()
        
        if not is_connected:
            return "Error: Cannot connect to Ollama service. Please check if Ollama is running on localhost:11434"
        
        # Check if our model is available
        model_names = [m['name'] for m in available_models]
        if 'llama3:latest' not in model_names:
            print(f"⚠️ DEBUG - Llama3 not found! Available models: {model_names}")
            return f"Error: Llama3 model not found. Available models: {', '.join(model_names)}"
        
        print(f"✅ DEBUG - Llama3 model found and ready!")
        # Create a focused data summary
        dataset_info = context['dataset_info']
        cleaning_report = context['cleaning_report']
        
        # Safely extract numeric columns
        numeric_cols = []
        try:
            if 'cleaning_summary' in cleaning_report and 'numeric_columns' in cleaning_report['cleaning_summary']:
                numeric_cols = cleaning_report['cleaning_summary']['numeric_columns']
                # Handle case where numeric_cols might be an integer (count) instead of list
                if isinstance(numeric_cols, int):
                    # Get actual numeric column names from summary stats
                    numeric_cols = [col for col in dataset_info.get('summary_stats', {}).keys() 
                                  if col in dataset_info.get('summary_stats', {})]
        except Exception as e:
            numeric_cols = []
        
        # Safely create summary stats
        summary_stats = dataset_info.get('summary_stats', {})
        key_stats = {}
        
        # Only process if we have numeric columns and summary stats
        if numeric_cols and summary_stats:
            for col in numeric_cols[:3]:  # Only first 3 numeric columns
                if col in summary_stats:
                    try:
                        stats = summary_stats[col]
                        if isinstance(stats, dict):
                            key_stats[col] = {
                                'mean': round(stats.get('mean', 0), 2),
                                'min': round(stats.get('min', 0), 2),
                                'max': round(stats.get('max', 0), 2)
                            }
                    except:
                        continue
        
        # Safely get missing and duplicate counts
        missing_count = 0
        duplicate_count = 0
        try:
            if 'cleaning_summary' in cleaning_report:
                missing_count = cleaning_report['cleaning_summary'].get('total_missing_values', 0)
                duplicate_count = cleaning_report['cleaning_summary'].get('total_duplicate_rows', 0)
        except:
            pass
        
        # Prepare a very focused prompt with actual data
        full_data = context.get('full_data', [])
        value_counts = context.get('value_counts', {})
        
        # Create a sample of actual data for the AI
        data_sample = full_data[:20] if full_data else []  # First 20 rows
        
        # Format data sample clearly
        data_sample_str = ""
        if data_sample:
            data_sample_str = "Sample data rows:\n"
            for i, row in enumerate(data_sample[:5]):
                data_sample_str += f"Row {i+1}: {row}\n"
        else:
            data_sample_str = "No data available"
        
        # Create value counts summary with clear formatting
        value_counts_str = ""
        if value_counts:
            value_counts_str = "DATASET SUMMARY:\n"
            for col, counts in value_counts.items():
                if counts:
                    # Generic approach - show top values for each column
                    top_values = dict(list(counts.items())[:5])  # Top 5 values
                    value_counts_str += f"{col}: {top_values}\n"
        
        # Check if this is a simple greeting
        user_question = context['user_question'].lower().strip()
        if user_question in ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']:
            return "Hello! I'm ready to analyze your dataset. Ask me anything about your data - I can provide summaries, distributions, correlations, or answer specific questions about your dataset."
        
        # Create a domain-agnostic prompt
        prompt = f"""You are analyzing a REAL dataset. Here is the ACTUAL data:

DATASET INFO:
- Rows: {dataset_info['rows']} records
- Columns: {dataset_info['columns']}
- Column names: {', '.join(dataset_info['column_names'])}
- Missing values: {missing_count}
- Duplicate rows: {duplicate_count}

ACTUAL DATA SAMPLE (first 5 records):
{data_sample_str}

ACTUAL DATA SUMMARY:
{value_counts_str}

USER QUESTION: {context['user_question']}

CRITICAL INSTRUCTIONS:
1. You MUST use the ACTUAL data provided above to answer the question.
2. Do NOT make up or guess values - use only the real data shown.
3. If asked for counts, use the value_counts data provided.
4. If asked for distributions, use the actual value distributions shown.
5. Format any tables in markdown format with | separators.
6. Be specific about the actual values in the dataset.
7. Do NOT provide generic responses - analyze the real data.
8. Do NOT confuse column values with record counts.
9. Interpret the data correctly: each row represents one record.
10. Focus on the actual data patterns and relationships shown.

Answer the user's question using ONLY the actual data provided above:"""

        # Debug: Print what data we're sending to AI
        print(f"DEBUG - Data being sent to AI:")
        print(f"Dataset rows: {dataset_info['rows']}")
        print(f"Dataset columns: {dataset_info['columns']}")
        print(f"Column names: {dataset_info['column_names']}")
        print(f"Value counts keys: {list(value_counts.keys())}")
        print(f"Sample data available: {len(full_data)} rows")
        print(f"User question: {context['user_question']}")
        
        # Debug: Print actual value counts to verify data
        print(f"DEBUG - Actual value counts:")
        for col, counts in value_counts.items():
            print(f"{col}: {dict(list(counts.items())[:3])}")  # Show first 3 values
        
        # Use Ollama API for AI response
        ollama_url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": "llama3:latest",  # Use Llama3 for better analysis
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 4096,   # Larger context for Llama3
                "num_thread": 8,    # More threads for Llama3
                "temperature": 0.2, # Lower temperature for more focused responses
                "top_k": 40,        # Larger token selection for Llama3
                "top_p": 0.9        # Nucleus sampling
            }
        }
        
        # Debug: Print connection attempt
        print(f"🔗 DEBUG - Attempting to connect to Ollama...")
        print(f"🌐 URL: {ollama_url}")
        print(f"🤖 Model: {payload['model']}")
        print(f"📝 Prompt length: {len(prompt)} characters")
        print(f"⏱️ Timeout: 300 seconds")
        
        try:
            response = requests.post(ollama_url, json=payload, timeout=300)  # 300 second timeout for Ollama responses
            print(f"✅ DEBUG - Ollama connection successful!")
            print(f"📊 Status Code: {response.status_code}")
            print(f"📄 Response Headers: {dict(response.headers)}")
        except requests.exceptions.ConnectTimeout:
            print(f"❌ DEBUG - Connection timeout! Ollama might not be running.")
            return "Error: Connection timeout. Please check if Ollama is running on localhost:11434"
        except requests.exceptions.ConnectionError:
            print(f"❌ DEBUG - Connection refused! Ollama service not available.")
            return "Error: Cannot connect to Ollama. Please start Ollama service first."
        except Exception as e:
            print(f"❌ DEBUG - Unexpected connection error: {e}")
            return f"Error: Connection failed - {str(e)}"
        
        if response.status_code == 200:
            print(f"✅ DEBUG - Ollama API response successful!")
            result = response.json()
            print(f"📄 DEBUG - Response keys: {list(result.keys())}")
            
            ai_response = result.get('response', 'I apologize, but I couldn\'t generate a response at this time.')
            print(f"🤖 DEBUG - AI Response length: {len(ai_response)} characters")
            print(f"📝 DEBUG - AI Response preview: {ai_response[:200]}...")
            
            return ai_response
        else:
            print(f"❌ DEBUG - Ollama API error: {response.status_code}")
            print(f"📄 DEBUG - Error response: {response.text}")
            return f"Sorry, I'm having trouble connecting to the AI service. HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "Sorry, I encountered an error while processing your request. Please try again."

# Usage instructions:
# To switch back to Ollama, replace the generate_gemini_response() call in views.py with:
# ai_response = generate_ai_response(context)
#
# Make sure Ollama is running on localhost:11434
# Make sure you have the required models installed: ollama pull llama3:latest 