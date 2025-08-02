from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
import os
import pandas as pd
from datetime import datetime
import json

# Create your views here.
def home(request):
    """Home view with file upload functionality"""
    if request.method == 'POST':
        # Handle file upload
        if 'dataset' in request.FILES:
            uploaded_file = request.FILES['dataset']
            
            # Validate file type
            allowed_extensions = ['.xlsx', '.xls', '.csv']
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension not in allowed_extensions:
                messages.error(request, 'Please upload a valid file type (.xlsx, .xls, or .csv)')
                return render(request, 'home.html')
            
            # Validate file size (max 10MB)
            if uploaded_file.size > 10 * 1024 * 1024:
                messages.error(request, 'File size must be less than 10MB')
                return render(request, 'home.html')
            
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
                
                messages.success(request, f'File "{uploaded_file.name}" uploaded successfully!')
                return redirect('upload_success')
                
            except Exception as e:
                messages.error(request, f'Error uploading file: {str(e)}')
                return render(request, 'home.html')
        else:
            messages.error(request, 'Please select a file to upload')
    
    return render(request, 'home.html')

def upload_success(request):
    """Show upload success and processing options"""
    if 'uploaded_file' not in request.session:
        return redirect('home')
    
    file_info = request.session['uploaded_file']
    
    # Try to load and preview the dataset
    try:
        if file_info['path'].endswith('.csv'):
            df = pd.read_csv(file_info['path'])
        else:
            df = pd.read_excel(file_info['path'])
        
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
        
    except Exception as e:
        messages.error(request, f'Error reading dataset: {str(e)}')
        return redirect('home')
    
    context = {
        'file_info': file_info,
        'dataset_info': dataset_info
    }
    
    return render(request, 'upload_success.html', context)

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
                # Store cleaned dataset info - ensure JSON serializable
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

def results(request):
    """Show analysis results"""
    if 'cleaned_dataset' not in request.session:
        return redirect('home')
    
    cleaned_dataset = request.session['cleaned_dataset']
    
    context = {
        'cleaned_dataset': cleaned_dataset
    }
    
    return render(request, 'results.html', context)