from django.contrib import admin
from .models import UserTokenUsage, TokenCost, UserSubscription, DailyTokenSummary, SubscriptionPlan

@admin.register(SubscriptionPlan)
class SubscriptionPlanAdmin(admin.ModelAdmin):
    list_display = ('name', 'display_name', 'monthly_token_limit', 'monthly_cost_bdt', 'is_active')
    list_filter = ('is_active', 'created_at')
    search_fields = ('name', 'display_name')
    readonly_fields = ('created_at', 'updated_at')

@admin.register(UserTokenUsage)
class UserTokenUsageAdmin(admin.ModelAdmin):
    list_display = ('user', 'session_id', 'dataset_name', 'timestamp', 'total_tokens', 'cost_usd', 'cost_bdt', 'model_used')
    list_filter = ('model_used', 'timestamp', 'user')
    search_fields = ('user__username', 'session_id', 'dataset_name', 'question')
    readonly_fields = ('total_tokens', 'cost_usd', 'cost_bdt')
    date_hierarchy = 'timestamp'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')

@admin.register(TokenCost)
class TokenCostAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'input_price_per_million', 'output_price_per_million', 'is_active', 'updated_at')
    list_filter = ('is_active', 'created_at')
    search_fields = ('model_name',)

@admin.register(UserSubscription)
class UserSubscriptionAdmin(admin.ModelAdmin):
    list_display = ('user', 'plan', 'monthly_token_limit', 'current_month_tokens_used', 'get_usage_percentage', 'is_active')
    list_filter = ('plan', 'is_active', 'created_at')
    search_fields = ('user__username', 'user__email')
    readonly_fields = ('get_usage_percentage',)
    
    def get_usage_percentage(self, obj):
        return f"{obj.get_usage_percentage():.1f}%"
    get_usage_percentage.short_description = 'Usage %'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user', 'plan')

@admin.register(DailyTokenSummary)
class DailyTokenSummaryAdmin(admin.ModelAdmin):
    list_display = ('date', 'total_users', 'total_questions', 'total_input_tokens', 'total_output_tokens', 'total_cost_usd', 'total_cost_bdt')
    list_filter = ('date',)
    date_hierarchy = 'date'
    readonly_fields = ('total_users', 'total_questions', 'total_input_tokens', 'total_output_tokens', 'total_cost_usd', 'total_cost_bdt')
