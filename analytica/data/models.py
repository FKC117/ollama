from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json

# Create your models here.

class SubscriptionPlan(models.Model):
    """Define available subscription plans"""
    name = models.CharField(max_length=50, unique=True)
    display_name = models.CharField(max_length=100)
    description = models.TextField()
    monthly_token_limit = models.IntegerField()
    monthly_cost_bdt = models.DecimalField(max_digits=10, decimal_places=2)
    features = models.JSONField(default=list)  # List of features included
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Subscription Plan'
        verbose_name_plural = 'Subscription Plans'
        ordering = ['monthly_cost_bdt']
    
    def __str__(self):
        return f"{self.display_name} - {self.monthly_cost_bdt} BDT/month"
    
    def get_features_list(self):
        """Return features as a list"""
        if isinstance(self.features, list):
            return self.features
        return []

class UserTokenUsage(models.Model):
    """Track token usage for each user interaction"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=255, null=True, blank=True)  # For anonymous users
    timestamp = models.DateTimeField(default=timezone.now)
    dataset_name = models.CharField(max_length=255, null=True, blank=True)
    question = models.TextField()
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    model_used = models.CharField(max_length=50, default='gemini-2.5-flash-lite')
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6, default=0.0)
    cost_bdt = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    exchange_rate = models.DecimalField(max_digits=10, decimal_places=4, default=122.50)  # USD to BDT
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'User Token Usage'
        verbose_name_plural = 'User Token Usages'
    
    def __str__(self):
        return f"{self.user or self.session_id} - {self.timestamp.strftime('%Y-%m-%d %H:%M')} - {self.total_tokens} tokens"
    
    def calculate_costs(self):
        """Calculate costs based on current Gemini pricing"""
        # Gemini 2.5 Flash Lite pricing (per 1M tokens)
        input_price_per_million = 0.1  # USD
        output_price_per_million = 0.4  # USD
        
        # Calculate costs
        input_cost_usd = (self.input_tokens / 1_000_000) * input_price_per_million
        output_cost_usd = (self.output_tokens / 1_000_000) * output_price_per_million
        total_cost_usd = input_cost_usd + output_cost_usd
        
        # Convert to BDT
        total_cost_bdt = total_cost_usd * self.exchange_rate
        
        self.cost_usd = total_cost_usd
        self.cost_bdt = total_cost_bdt
        return total_cost_usd, total_cost_bdt

class TokenCost(models.Model):
    """Store current token pricing for different models"""
    model_name = models.CharField(max_length=50, unique=True)
    input_price_per_million = models.DecimalField(max_digits=10, decimal_places=6)
    output_price_per_million = models.DecimalField(max_digits=10, decimal_places=6)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'Token Cost'
        verbose_name_plural = 'Token Costs'
    
    def __str__(self):
        return f"{self.model_name} - Input: ${self.input_price_per_million}/1M, Output: ${self.output_price_per_million}/1M"

class UserSubscription(models.Model):
    """Track user subscription tiers and limits"""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.SET_NULL, null=True, blank=True)
    monthly_token_limit = models.IntegerField(default=10000)  # Default 10K tokens for free tier
    monthly_cost_bdt = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    current_month_tokens_used = models.IntegerField(default=0)
    reset_date = models.DateField(default=timezone.now)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = 'User Subscription'
        verbose_name_plural = 'User Subscriptions'
    
    def __str__(self):
        plan_name = self.plan.display_name if self.plan else "No Plan"
        return f"{self.user.username} - {plan_name}"
    
    def get_usage_percentage(self):
        """Get current month usage as percentage"""
        if self.monthly_token_limit == 0:
            return 0
        return (self.current_month_tokens_used / self.monthly_token_limit) * 100
    
    def can_use_tokens(self, token_count):
        """Check if user can use specified number of tokens"""
        return (self.current_month_tokens_used + token_count) <= self.monthly_token_limit
    
    def add_tokens_used(self, token_count):
        """Add tokens to current month usage"""
        self.current_month_tokens_used += token_count
        self.save()

class DailyTokenSummary(models.Model):
    """Daily summary of token usage for analytics"""
    date = models.DateField(unique=True)
    total_users = models.IntegerField(default=0)
    total_questions = models.IntegerField(default=0)
    total_input_tokens = models.IntegerField(default=0)
    total_output_tokens = models.IntegerField(default=0)
    total_cost_usd = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    total_cost_bdt = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    
    class Meta:
        verbose_name = 'Daily Token Summary'
        verbose_name_plural = 'Daily Token Summaries'
        ordering = ['-date']
    
    def __str__(self):
        return f"{self.date} - {self.total_questions} questions, {self.total_input_tokens + self.total_output_tokens} tokens"
