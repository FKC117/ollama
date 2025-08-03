from django.core.management.base import BaseCommand
from data.models import SubscriptionPlan

class Command(BaseCommand):
    help = 'Set up initial subscription plans'

    def handle(self, *args, **options):
        plans = [
            {
                'name': 'free',
                'display_name': 'Free Plan',
                'description': 'Perfect for getting started with AI-powered data analysis',
                'monthly_token_limit': 10000,
                'monthly_cost_bdt': 0.00,
                'features': [
                    '10,000 tokens per month',
                    'Basic data analysis',
                    'AI chat support',
                    'Standard response time'
                ]
            },
            {
                'name': 'basic',
                'display_name': 'Basic Plan',
                'description': 'Great for regular data analysis needs',
                'monthly_token_limit': 50000,
                'monthly_cost_bdt': 500.00,
                'features': [
                    '50,000 tokens per month',
                    'Advanced data analysis',
                    'Priority AI chat support',
                    'Faster response time',
                    'Export analysis results'
                ]
            },
            {
                'name': 'premium',
                'display_name': 'Premium Plan',
                'description': 'For power users and small teams',
                'monthly_token_limit': 200000,
                'monthly_cost_bdt': 1500.00,
                'features': [
                    '200,000 tokens per month',
                    'Advanced analytics',
                    'Priority support',
                    'Fast response time',
                    'Export results',
                    'Custom analysis templates',
                    'Team collaboration features'
                ]
            },
            {
                'name': 'enterprise',
                'display_name': 'Enterprise Plan',
                'description': 'For large organizations with high-volume needs',
                'monthly_token_limit': 1000000,
                'monthly_cost_bdt': 5000.00,
                'features': [
                    '1,000,000 tokens per month',
                    'Enterprise-grade analytics',
                    'Dedicated support',
                    'Instant response time',
                    'Advanced exports',
                    'Custom integrations',
                    'White-label options',
                    'API access'
                ]
            }
        ]

        for plan_data in plans:
            plan, created = SubscriptionPlan.objects.get_or_create(
                name=plan_data['name'],
                defaults=plan_data
            )
            
            if created:
                self.stdout.write(
                    self.style.SUCCESS(f'Created plan: {plan.display_name}')
                )
            else:
                # Update existing plan
                for key, value in plan_data.items():
                    setattr(plan, key, value)
                plan.save()
                self.stdout.write(
                    self.style.WARNING(f'Updated plan: {plan.display_name}')
                )

        self.stdout.write(
            self.style.SUCCESS('Successfully set up subscription plans')
        ) 