import re
from decimal import Decimal
from django.utils import timezone
from .models import UserTokenUsage, TokenCost, UserSubscription, DailyTokenSummary

def estimate_tokens(text):
    """
    Estimate token count for text (rough approximation)
    Gemini uses a similar tokenization to GPT models
    """
    if not text:
        return 0
    
    # Rough estimation: 1 token ≈ 4 characters for English text
    # This is a simplified approximation
    return len(text) // 4

def calculate_token_costs(input_tokens, output_tokens, model_name='gemini-2.5-flash-lite'):
    """
    Calculate costs for input and output tokens
    """
    # Default Gemini 2.5 Flash Lite pricing (per 1M tokens)
    pricing = {
        'gemini-2.5-flash-lite': {
            'input_price_per_million': 0.1,  # USD
            'output_price_per_million': 0.4,  # USD
        },
        'gemini-2.5-flash': {
            'input_price_per_million': 0.3,  # USD
            'output_price_per_million': 2.5,  # USD
        },
        'gemini-2.5-pro': {
            'input_price_per_million': 1.25,  # USD
            'output_price_per_million': 10.0,  # USD
        }
    }
    
    model_pricing = pricing.get(model_name, pricing['gemini-2.5-flash-lite'])
    
    # Calculate costs
    input_cost_usd = (input_tokens / 1_000_000) * model_pricing['input_price_per_million']
    output_cost_usd = (output_tokens / 1_000_000) * model_pricing['output_price_per_million']
    total_cost_usd = input_cost_usd + output_cost_usd
    
    # Convert to BDT (current exchange rate ~117.50)
    exchange_rate = Decimal('117.50')
    total_cost_bdt = Decimal(str(total_cost_usd)) * exchange_rate
    
    return {
        'input_cost_usd': input_cost_usd,
        'output_cost_usd': output_cost_usd,
        'total_cost_usd': total_cost_usd,
        'total_cost_bdt': total_cost_bdt,
        'exchange_rate': exchange_rate
    }

def track_token_usage(user, session_id, dataset_name, question, input_tokens, output_tokens, 
                     model_used='gemini-2.5-flash-lite', response_text=''):
    """
    Track token usage and create UserTokenUsage record
    """
    total_tokens = input_tokens + output_tokens
    
    # Calculate costs
    costs = calculate_token_costs(input_tokens, output_tokens, model_used)
    
    # Create usage record
    usage = UserTokenUsage.objects.create(
        user=user,
        session_id=session_id,
        dataset_name=dataset_name,
        question=question,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        model_used=model_used,
        cost_usd=costs['total_cost_usd'],
        cost_bdt=costs['total_cost_bdt'],
        exchange_rate=costs['exchange_rate']
    )
    
    # Update user subscription if user is authenticated
    if user and user.is_authenticated:
        subscription, created = UserSubscription.objects.get_or_create(
            user=user,
            defaults={
                'monthly_token_limit': 10000,
                'monthly_cost_bdt': 0.0
            }
        )
        subscription.add_tokens_used(total_tokens)
    
    # Update daily summary
    update_daily_summary(input_tokens, output_tokens, costs)
    
    return usage

def update_daily_summary(input_tokens, output_tokens, costs):
    """
    Update daily token usage summary
    """
    today = timezone.now().date()
    
    summary, created = DailyTokenSummary.objects.get_or_create(
        date=today,
        defaults={
            'total_users': 0,
            'total_questions': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost_usd': 0.0,
            'total_cost_bdt': 0.0
        }
    )
    
    summary.total_questions += 1
    summary.total_input_tokens += input_tokens
    summary.total_output_tokens += output_tokens
    summary.total_cost_usd += Decimal(str(costs['total_cost_usd']))
    summary.total_cost_bdt += costs['total_cost_bdt']
    summary.save()

def get_user_usage_summary(user=None, session_id=None, days=30):
    """
    Get token usage summary for user or session
    """
    end_date = timezone.now()
    start_date = end_date - timezone.timedelta(days=days)
    
    if user:
        usages = UserTokenUsage.objects.filter(
            user=user,
            timestamp__range=(start_date, end_date)
        )
    elif session_id:
        usages = UserTokenUsage.objects.filter(
            session_id=session_id,
            timestamp__range=(start_date, end_date)
        )
    else:
        return None
    
    total_questions = usages.count()
    total_input_tokens = sum(u.input_tokens for u in usages)
    total_output_tokens = sum(u.output_tokens for u in usages)
    total_tokens = total_input_tokens + total_output_tokens
    total_cost_usd = sum(u.cost_usd for u in usages)
    total_cost_bdt = sum(u.cost_bdt for u in usages)
    
    return {
        'total_questions': total_questions,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_tokens,
        'total_cost_usd': total_cost_usd,
        'total_cost_bdt': total_cost_bdt,
        'average_tokens_per_question': total_tokens / total_questions if total_questions > 0 else 0,
        'average_cost_per_question_usd': total_cost_usd / total_questions if total_questions > 0 else 0,
        'average_cost_per_question_bdt': total_cost_bdt / total_questions if total_questions > 0 else 0,
    }

def get_daily_usage_chart_data(days=30):
    """
    Get daily usage data for charts
    """
    end_date = timezone.now()
    start_date = end_date - timezone.timedelta(days=days)
    
    summaries = DailyTokenSummary.objects.filter(
        date__range=(start_date, end_date)
    ).order_by('date')
    
    chart_data = {
        'dates': [],
        'questions': [],
        'tokens': [],
        'costs_usd': [],
        'costs_bdt': []
    }
    
    for summary in summaries:
        chart_data['dates'].append(summary.date.strftime('%Y-%m-%d'))
        chart_data['questions'].append(summary.total_questions)
        chart_data['tokens'].append(summary.total_input_tokens + summary.total_output_tokens)
        chart_data['costs_usd'].append(float(summary.total_cost_usd))
        chart_data['costs_bdt'].append(float(summary.total_cost_bdt))
    
    return chart_data

def check_user_token_limit(user, estimated_tokens):
    """
    Check if user can use estimated tokens
    """
    if not user or not user.is_authenticated:
        return True  # Anonymous users have no limits for now
    
    subscription, created = UserSubscription.objects.get_or_create(
        user=user,
        defaults={
            'monthly_token_limit': 10000,
            'monthly_cost_bdt': 0.0
        }
    )
    
    return subscription.can_use_tokens(estimated_tokens)

def get_subscription_tiers():
    """
    Get available subscription tiers and their limits
    """
    return {
        'free': {
            'name': 'Free',
            'monthly_token_limit': 10000,
            'monthly_cost_bdt': 0.0,
            'features': ['Basic AI analysis', 'Up to 10K tokens/month', 'Community support']
        },
        'basic': {
            'name': 'Basic',
            'monthly_token_limit': 50000,
            'monthly_cost_bdt': 500.0,
            'features': ['Enhanced AI analysis', 'Up to 50K tokens/month', 'Email support', 'Priority processing']
        },
        'premium': {
            'name': 'Premium',
            'monthly_token_limit': 200000,
            'monthly_cost_bdt': 1500.0,
            'features': ['Advanced AI analysis', 'Up to 200K tokens/month', 'Priority support', 'Custom datasets', 'Advanced charts']
        },
        'enterprise': {
            'name': 'Enterprise',
            'monthly_token_limit': 1000000,
            'monthly_cost_bdt': 5000.0,
            'features': ['Unlimited AI analysis', 'Up to 1M tokens/month', 'Dedicated support', 'Custom integrations', 'API access']
        }
    } 