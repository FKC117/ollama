from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
import os
import pandas as pd
from datetime import datetime
import json
import requests
import time

# Create your views here.
def home(request):
    """Unified home view that handles file upload, processing, and analysis"""
    context = {
        'file_uploaded': False,
        'dataset_processed': False,
        'file_info': None,
        'dataset_info': None,
        'cleaned_dataset': None
    }
    
    if request.method == 'POST':
        # Handle file upload
        if 'dataset' in request.FILES:
            uploaded_file = request.FILES['dataset']
            
            # Validate file type
            allowed_extensions = ['.xlsx', '.xls', '.csv']
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension not in allowed_extensions:
                messages.error(request, 'Please upload a valid file type (.xlsx, .xls, or .csv)')
                return render(request, 'home.html', context)
            
            # Validate file size (max 10MB)
            if uploaded_file.size > 10 * 1024 * 1024:
                messages.error(request, 'File size must be less than 10MB')
                return render(request, 'home.html', context)
            
            try:
                # Create uploads directory if it doesn't exist
                upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
                os.makedirs(upload_dir, exist_ok=True)
                
                # Save file with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{uploaded_file.name}"
                file_path = os.path.join(upload_dir, filename)
                
                with open(file_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                
                # Store file info in session for processing
                request.session['uploaded_file'] = {
                    'path': file_path,
                    'original_name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'upload_time': timestamp
                }
                
                # Load and preview the dataset immediately
                try:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_excel(file_path)
                    
                    # Get basic dataset info - convert pandas types to strings for JSON serialization
                    data_types_dict = {}
                    for col, dtype in df.dtypes.items():
                        data_types_dict[str(col)] = str(dtype)
                    
                    missing_values_dict = {}
                    for col, count in df.isnull().sum().items():
                        missing_values_dict[str(col)] = int(count)
                    
                    dataset_info = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': [str(col) for col in df.columns],
                        'data_types': data_types_dict,
                        'missing_values': missing_values_dict,
                        'sample_data': df.head(5).to_dict('records')
                    }
                    
                    # Store dataset info in session
                    request.session['dataset_info'] = dataset_info
                    
                    # Update context for immediate display
                    context.update({
                        'file_uploaded': True,
                        'file_info': request.session['uploaded_file'],
                        'dataset_info': dataset_info
                    })
                    
                    messages.success(request, f'File "{uploaded_file.name}" uploaded successfully!')
                    
                except Exception as e:
                    messages.error(request, f'Error reading dataset: {str(e)}')
                    return render(request, 'home.html', context)
                
            except Exception as e:
                messages.error(request, f'Error uploading file: {str(e)}')
                return render(request, 'home.html', context)
        else:
            messages.error(request, 'Please select a file to upload')
    
    # Check if we have processed data from session
    if 'cached_cleaned_data' in request.session:
        context['dataset_processed'] = True
        context['cleaned_dataset'] = request.session.get('cleaned_dataset', {})
    
    # Check if we have uploaded file from session
    if 'uploaded_file' in request.session:
        context['file_uploaded'] = True
        context['file_info'] = request.session['uploaded_file']
        if 'dataset_info' in request.session:
            context['dataset_info'] = request.session['dataset_info']
    
    return render(request, 'home.html', context)

def process_dataset(request):
    """Process the uploaded dataset"""
    if request.method == 'POST':
        if 'uploaded_file' not in request.session:
            return JsonResponse({'error': 'No file uploaded'}, status=400)
        
        try:
            file_info = request.session['uploaded_file']
            
            # Import and use the cleaning script
            from .cleaning import clean_and_prepare_dataset
            
            # Clean and prepare the dataset
            cleaned_df, cleaning_report = clean_and_prepare_dataset(file_info['path'])
            
            if cleaned_df is not None:
                # Cache the cleaned data in session for AI chat
                request.session['cached_cleaned_data'] = {
                    'shape': list(cleaned_df.shape),
                    'columns': [str(col) for col in cleaned_df.columns],
                    'data_types': {str(k): str(v) for k, v in cleaned_df.dtypes.to_dict().items()},
                    'sample_data': cleaned_df.head(10).to_dict('records'),
                    'summary_stats': cleaned_df.describe().to_dict(),
                    'missing_values': cleaned_df.isnull().sum().to_dict(),
                    'cleaning_report': cleaning_report
                }
                
                # Store cleaned dataset info
                request.session['cleaned_dataset'] = {
                    'shape': list(cleaned_df.shape),  # Convert tuple to list
                    'columns': [str(col) for col in cleaned_df.columns],  # Convert to strings
                    'cleaning_report': cleaning_report
                }
                
                return JsonResponse({
                    'success': True,
                    'message': 'Dataset processed successfully',
                    'cleaning_report': cleaning_report
                })
            else:
                return JsonResponse({
                    'error': 'Failed to process dataset'
                }, status=500)
                
        except Exception as e:
            return JsonResponse({
                'error': f'Error processing dataset: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def ai_chat(request):
    """Handle AI chat requests with cached data"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            
            # Check if we have cached cleaned data
            if 'cached_cleaned_data' not in request.session:
                return JsonResponse({
                    'error': 'No dataset available. Please upload and process a dataset first.'
                }, status=400)
            
            cached_data = request.session['cached_cleaned_data']
            
            # Prepare context for AI
            context = {
                'dataset_info': {
                    'rows': cached_data['shape'][0],
                    'columns': cached_data['shape'][1],
                    'column_names': cached_data['columns'],
                    'data_types': cached_data['data_types'],
                    'missing_values': cached_data['missing_values'],
                    'sample_data': cached_data['sample_data'][:5],  # First 5 rows
                    'summary_stats': cached_data['summary_stats']
                },
                'cleaning_report': cached_data['cleaning_report'],
                'user_question': user_message
            }
            
            # Generate AI response using our existing AI infrastructure
            ai_response = generate_ai_response(context)
            
            return JsonResponse({
                'success': True,
                'response': ai_response
            })
            
        except Exception as e:
            return JsonResponse({
                'error': f'Error processing chat request: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def generate_ai_response(context):
    """Generate AI response using Ollama AI only"""
    try:
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
        
        # Prepare a very focused prompt
        prompt = f"""You are analyzing a real dataset with the following characteristics:

Dataset: {dataset_info['rows']} rows, {dataset_info['columns']} columns
Columns: {', '.join(dataset_info['column_names'][:8])}{'...' if len(dataset_info['column_names']) > 8 else ''}
Missing: {missing_count}, Duplicates: {duplicate_count}

Key statistics: {', '.join([f"{col}(mean={stats['mean']})" for col, stats in key_stats.items()])}

User Question: {context['user_question']}

IMPORTANT: Analyze the actual data provided above. Do NOT provide code examples, SQL queries, or programming instructions. Instead, give insights about the data patterns, trends, and findings based on the statistics shown. Focus on what the data reveals about the subject matter.

Provide a brief, insightful analysis:"""
        
        # Use Ollama API for AI response
        ollama_url = "http://localhost:11434/api/generate"
        
        payload = {
            "model": "tinyllama:latest",  # Use TinyLlama for speed
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 1024,   # Smaller context for TinyLlama
                "num_thread": 4,    # Fewer threads for TinyLlama
                "temperature": 0.3, # Lower temperature for focused responses
                "top_k": 20,        # Smaller token selection
                "top_p": 0.8        # Nucleus sampling
            }
        }
        
        response = requests.post(ollama_url, json=payload, timeout=180)  # 180 second timeout for Ollama responses
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get('response', 'I apologize, but I couldn\'t generate a response at this time.')
            return ai_response
        else:
            print(f"Ollama API error: {response.status_code} - {response.text}")
            return "Sorry, I'm having trouble connecting to the AI service. Please check if Ollama is running."
            
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "Sorry, I encountered an error while processing your request. Please try again."