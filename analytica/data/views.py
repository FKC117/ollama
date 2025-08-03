from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import auth
from .models import SubscriptionPlan, UserSubscription
import os
import pandas as pd
from datetime import datetime
import json
import requests
import time
import google.generativeai as genai
from .token_utils import estimate_tokens, track_token_usage, get_user_usage_summary, get_daily_usage_chart_data, check_user_token_limit, get_subscription_tiers

# Configure Gemini API
genai.configure(api_key="AIzaSyBa8X7pjpMQFwn5OkzrW4IvSKGMECJWd44")

# Authentication Views
def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'auth/login.html')

def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Account created successfully! Welcome, {user.username}!')
            return redirect('home')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        form = UserCreationForm()
    
    return render(request, 'auth/register.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')

# Subscription Views
@login_required(login_url='login')
def subscription_plans(request):
    """Display available subscription plans"""
    plans = SubscriptionPlan.objects.filter(is_active=True)
    user_subscription, created = UserSubscription.objects.get_or_create(
        user=request.user,
        defaults={'monthly_token_limit': 10000, 'monthly_cost_bdt': 0.0}
    )
    
    context = {
        'plans': plans,
        'user_subscription': user_subscription,
    }
    return render(request, 'subscription/plans.html', context)

@login_required(login_url='login')
def select_plan(request, plan_id):
    """Select a subscription plan for the user"""
    try:
        plan = SubscriptionPlan.objects.get(id=plan_id, is_active=True)
        user_subscription, created = UserSubscription.objects.get_or_create(
            user=request.user,
            defaults={'monthly_token_limit': 10000, 'monthly_cost_bdt': 0.0}
        )
        
        # Update user subscription
        user_subscription.plan = plan
        user_subscription.monthly_token_limit = plan.monthly_token_limit
        user_subscription.monthly_cost_bdt = plan.monthly_cost_bdt
        user_subscription.current_month_tokens_used = 0  # Reset usage
        user_subscription.reset_date = datetime.now().date()
        user_subscription.save()
        
        messages.success(request, f'Successfully subscribed to {plan.display_name}!')
        return redirect('subscription_plans')
        
    except SubscriptionPlan.DoesNotExist:
        messages.error(request, 'Selected plan not found.')
        return redirect('subscription_plans')

@login_required(login_url='login')
def my_subscription(request):
    """Display user's current subscription details"""
    user_subscription, created = UserSubscription.objects.get_or_create(
        user=request.user,
        defaults={'monthly_token_limit': 10000, 'monthly_cost_bdt': 0.0}
    )
    
    # Get usage summary
    usage_summary = get_user_usage_summary(request.user)
    
    context = {
        'user_subscription': user_subscription,
        'usage_summary': usage_summary,
    }
    return render(request, 'subscription/my_subscription.html', context)

# Create your views here.
@login_required(login_url='login')
def home(request):
    """Unified home view that handles file upload, processing, and analysis"""
    context = {
        'file_uploaded': False,
        'dataset_processed': False,
        'file_info': None,
        'dataset_info': None,
        'cleaned_dataset': None,
        'ai_ready': False
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
                    
                    # Automatically process the dataset
                    try:
                        from .cleaning import clean_and_prepare_dataset
                        
                        # Clean and prepare the dataset
                        cleaned_df, cleaning_report = clean_and_prepare_dataset(file_path)
                        
                        if cleaned_df is not None:
                            # Cache the cleaned data in session for AI chat
                            request.session['cached_cleaned_data'] = {
                                'shape': list(cleaned_df.shape),
                                'columns': [str(col) for col in cleaned_df.columns],
                                'data_types': {str(k): str(v) for k, v in cleaned_df.dtypes.to_dict().items()},
                                'sample_data': cleaned_df.head(10).to_dict('records'),
                                'summary_stats': cleaned_df.describe().to_dict(),
                                'missing_values': cleaned_df.isnull().sum().to_dict(),
                                'cleaning_report': cleaning_report,
                                # Add actual data for AI analysis - send first 500 rows or entire dataset if smaller
                                'full_data': cleaned_df.to_dict('records'),  # All rows for analysis
                                'value_counts': {col: cleaned_df[col].value_counts().to_dict() for col in cleaned_df.columns}
                            }
                            
                            # Store cleaned dataset info
                            request.session['cleaned_dataset'] = {
                                'shape': list(cleaned_df.shape),  # Convert tuple to list
                                'columns': [str(col) for col in cleaned_df.columns],  # Convert to strings
                                'cleaning_report': cleaning_report
                            }
                            
                            # Update context for immediate display
                            context.update({
                                'file_uploaded': True,
                                'file_info': request.session['uploaded_file'],
                                'dataset_info': dataset_info,
                                'dataset_processed': True,
                                'cleaned_dataset': request.session['cleaned_dataset']
                            })
                            
                            messages.success(request, f'File "{uploaded_file.name}" uploaded and processed successfully!')
                            
                            # Redirect to prevent POST resubmission
                            return redirect('home')
                        else:
                            messages.error(request, 'Failed to process dataset')
                            return render(request, 'home.html', context)
                            
                    except Exception as e:
                        messages.error(request, f'Error processing dataset: {str(e)}')
                        return render(request, 'home.html', context)
                    
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
    
    # Check if AI is ready
    if 'ai_data_ready' in request.session:
        context['ai_ready'] = True
    
    return render(request, 'home.html', context)

def pass_to_ai(request):
    """Pass the processed dataset to AI for analysis"""
    if request.method == 'POST':
        try:
            # Check if we have cached cleaned data
            if 'cached_cleaned_data' not in request.session:
                return JsonResponse({
                    'error': 'No processed dataset available. Please upload and process a dataset first.'
                }, status=400)
            
            # Mark AI as ready
            request.session['ai_data_ready'] = True
            
            # Get dataset info for display
            file_info = request.session.get('uploaded_file', {})
            dataset_info = request.session.get('dataset_info', {})
            
            return JsonResponse({
                'success': True,
                'message': 'Dataset passed to AI successfully!',
                'dataset_name': file_info.get('original_name', 'Unknown Dataset'),
                'rows': dataset_info.get('rows', 0),
                'columns': dataset_info.get('columns', 0)
            })
            
        except Exception as e:
            return JsonResponse({
                'error': f'Error passing data to AI: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

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
                    'cleaning_report': cleaning_report,
                    # Add actual data for AI analysis - send first 500 rows or entire dataset if smaller
                    'full_data': cleaned_df.to_dict('records'),  # All rows for analysis
                    'value_counts': {col: cleaned_df[col].value_counts().to_dict() for col in cleaned_df.columns}
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
    """Handle AI chat requests with cached data using Gemini API"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            
            # Check if we have cached cleaned data
            if 'cached_cleaned_data' not in request.session:
                return JsonResponse({
                    'error': 'No dataset available. Please upload and process a dataset first.'
                }, status=400)
            
            # Debug: Print session info
            print(f"DEBUG - Session data available:")
            print(f"Cached data keys: {list(request.session.get('cached_cleaned_data', {}).keys())}")
            print(f"Session keys: {list(request.session.keys())}")
            
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
                'user_question': user_message,
                # Add actual data for AI analysis
                'full_data': cached_data.get('full_data', []),
                'value_counts': cached_data.get('value_counts', {})
            }
            
            # Estimate tokens for the request
            estimated_input_tokens = estimate_tokens(str(context))
            
            # Check user token limits
            if not check_user_token_limit(request.user, estimated_input_tokens):
                return JsonResponse({
                    'error': 'You have reached your monthly token limit. Please upgrade your subscription or wait until next month.'
                }, status=429)
            
            # Generate AI response using Gemini
            ai_response = generate_gemini_response(context)
            
            # Track token usage
            input_tokens = estimated_input_tokens
            output_tokens = estimate_tokens(ai_response)
            
            # Get dataset name from session
            dataset_name = request.session.get('uploaded_file', {}).get('original_name', 'Unknown Dataset')
            
            # Track the usage
            usage = track_token_usage(
                user=request.user,
                session_id=request.session.session_key,
                dataset_name=dataset_name,
                question=user_message,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_used='gemini-2.5-flash-lite',
                response_text=ai_response
            )
            
            return JsonResponse({
                'success': True,
                'response': ai_response,
                'token_usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens,
                    'cost_usd': float(usage.cost_usd),
                    'cost_bdt': float(usage.cost_bdt)
                }
            })
            
        except Exception as e:
            print(f"AI Chat Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'error': f'Error processing chat request: {str(e)}'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def generate_gemini_response(context):
    """Generate AI response using Google Gemini API"""
    try:
        # Get user tier (default to basic for now)
        user_tier = 'basic'  # You can implement tier logic later
        
        # Select model based on tier
        if user_tier == 'basic':
            model_name = 'gemini-2.5-flash-lite'
        elif user_tier == 'premium':
            model_name = 'gemini-2.5-flash'
        else:
            model_name = 'gemini-2.5-pro'
        
        print(f"🤖 DEBUG - Using Gemini model: {model_name}")
        
        # Initialize Gemini model
        model = genai.GenerativeModel(model_name)
        
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
        
        # Send first 500 rows or entire dataset if smaller
        total_rows = len(full_data)
        rows_to_send = min(500, total_rows)
        data_sample = full_data[:rows_to_send] if full_data else []
        
        print(f"📊 DEBUG - Sending {rows_to_send} rows out of {total_rows} total rows to Gemini")
        
        # Format data sample clearly
        data_sample_str = ""
        if data_sample:
            data_sample_str = f"ACTUAL DATASET (first {rows_to_send} rows):\n"
            for i, row in enumerate(data_sample[:10]):  # Show first 10 rows in prompt
                data_sample_str += f"Row {i+1}: {row}\n"
            if len(data_sample) > 10:
                data_sample_str += f"... (showing first 10 rows out of {rows_to_send} rows sent)\n"
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
        
        # Create a domain-agnostic prompt with chart generation instructions
        prompt = f"""You are analyzing a REAL dataset. Here is the ACTUAL data:

DATASET INFO:
- Rows: {dataset_info['rows']} records
- Columns: {dataset_info['columns']}
- Column names: {', '.join(dataset_info['column_names'])}
- Missing values: {missing_count}
- Duplicate rows: {duplicate_count}
- Data sent to analysis: {rows_to_send} rows

{data_sample_str}

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

CHART GENERATION INSTRUCTIONS:
When appropriate, include ASCII charts and visualizations in your response. Use EXACTLY these formats:

BAR CHARTS (use █ characters):
Category Name: ████████████████████ (count)
Category Name: ████████████ (count)

LINE CHARTS (use ─ and │ characters):
Chart Title:
─────────────────────────────────────────
│    ●    ●    ●    ●    ●    ●    ●
│   ●   ●   ●   ●   ●   ●   ●   ●
│  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
│ ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●
└─────────────────────────────────────────

PIE CHARTS (use ▓ ▒ ░ characters):
Chart Title:
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ (50%)
▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ (30%)
░░░░░░░░░░ (20%)

BOX PLOTS (use ┌─┐│└─┘ characters):
Chart Title:
┌─────────────────────────────────────┐
│    ●    ●    ●    ●    ●    ●    ● │
│   ●   ●   ●   ●   ●   ●   ●   ●   │
│  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●     │
│ ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●   │
└─────────────────────────────────────┘

CRITICAL CHART FORMATTING RULES:
1. Use ONLY the ASCII characters shown above - no HTML, CSS, or styling
2. Do NOT include any font-family, background, padding, or border-radius instructions
3. Do NOT include any style attributes or HTML tags
4. Keep charts simple and clean - just the ASCII characters and text
5. Always include the count in parentheses for bar charts
6. Use consistent spacing and formatting
7. Make sure charts are readable in plain text format

Answer the user's question using ONLY the actual data provided above. Include relevant charts and visualizations when appropriate:"""

        # Debug: Print what data we're sending to AI
        print(f"DEBUG - Data being sent to Gemini:")
        print(f"Dataset rows: {dataset_info['rows']}")
        print(f"Dataset columns: {dataset_info['columns']}")
        print(f"Column names: {dataset_info['column_names']}")
        print(f"Value counts keys: {list(value_counts.keys())}")
        print(f"Sample data available: {len(full_data)} rows")
        print(f"Rows sent to Gemini: {rows_to_send}")
        print(f"User question: {context['user_question']}")
        
        # Debug: Print actual value counts to verify data
        print(f"DEBUG - Actual value counts:")
        for col, counts in value_counts.items():
            print(f"{col}: {dict(list(counts.items())[:3])}")  # Show first 3 values
        
        # Generate response using Gemini
        print(f"🔗 DEBUG - Sending request to Gemini API...")
        print(f"🤖 Model: {model_name}")
        print(f"📝 Prompt length: {len(prompt)} characters")
        
        try:
            response = model.generate_content(prompt)
            print(f"✅ DEBUG - Gemini API response successful!")
            
            ai_response = response.text
            print(f"🤖 DEBUG - AI Response length: {len(ai_response)} characters")
            print(f"📝 DEBUG - AI Response preview: {ai_response[:200]}...")
            
            return ai_response
            
        except Exception as e:
            print(f"❌ DEBUG - Gemini API error: {str(e)}")
            return f"Sorry, I'm having trouble connecting to the AI service. Error: {str(e)}"
            
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "Sorry, I encountered an error while processing your request. Please try again."

def token_usage_dashboard(request):
    """Display token usage dashboard for users"""
    try:
        # Get user usage summary
        usage_summary = get_user_usage_summary(
            user=request.user if request.user.is_authenticated else None,
            session_id=request.session.session_key if not request.user.is_authenticated else None,
            days=30
        )
        
        # Get daily chart data
        chart_data = get_daily_usage_chart_data(days=30)
        
        # Get subscription info if user is authenticated
        subscription_info = None
        if request.user.is_authenticated:
            from .models import UserSubscription
            subscription_info = UserSubscription.objects.filter(user=request.user).first()
        
        context = {
            'usage_summary': usage_summary,
            'chart_data': chart_data,
            'subscription_info': subscription_info,
            'subscription_tiers': get_subscription_tiers()
        }
        
        return render(request, 'token_dashboard.html', context)
        
    except Exception as e:
        print(f"Token Dashboard Error: {str(e)}")
        messages.error(request, f'Error loading token usage dashboard: {str(e)}')
        return render(request, 'token_dashboard.html', {'error': str(e)})

def api_token_usage(request):
    """API endpoint to get token usage data"""
    if request.method == 'GET':
        try:
            days = int(request.GET.get('days', 30))
            
            usage_summary = get_user_usage_summary(
                user=request.user if request.user.is_authenticated else None,
                session_id=request.session.session_key if not request.user.is_authenticated else None,
                days=days
            )
            
            return JsonResponse({
                'success': True,
                'usage_summary': usage_summary
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def api_daily_usage(request):
    """API endpoint to get daily usage chart data"""
    if request.method == 'GET':
        try:
            days = int(request.GET.get('days', 30))
            chart_data = get_daily_usage_chart_data(days=days)
            
            return JsonResponse({
                'success': True,
                'chart_data': chart_data
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)