from django.urls import path, include
from . import views

urlpatterns = [
    # Authentication URLs
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    
    # Subscription URLs
    path('subscription/plans/', views.subscription_plans, name='subscription_plans'),
    path('subscription/select/<int:plan_id>/', views.select_plan, name='select_plan'),
    path('subscription/my-subscription/', views.my_subscription, name='my_subscription'),
    
    # Main application URLs
    path('', views.home, name='home'),
    path('process-dataset/', views.process_dataset, name='process_dataset'),
    path('pass-to-ai/', views.pass_to_ai, name='pass_to_ai'),
    path('ai-chat/', views.ai_chat, name='ai_chat'),
    path('token-dashboard/', views.token_usage_dashboard, name='token_dashboard'),
    path('api/token-usage/', views.api_token_usage, name='api_token_usage'),
    path('api/daily-usage/', views.api_daily_usage, name='api_daily_usage'),
]