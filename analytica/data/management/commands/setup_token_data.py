from django.core.management.base import BaseCommand
from django.utils import timezone
from data.models import TokenCost, UserTokenUsage, DailyTokenSummary
from datetime import timedelta
import random

class Command(BaseCommand):
    help = 'Set up initial token pricing and sample usage data'

    def handle(self, *args, **options):
        self.stdout.write('Setting up token pricing data...')
        
        # Create token pricing data
        pricing_data = [
            {
                'model_name': 'gemini-2.5-flash-lite',
                'input_price_per_million': 0.1,
                'output_price_per_million': 0.4,
                'is_active': True
            },
            {
                'model_name': 'gemini-2.5-flash',
                'input_price_per_million': 0.3,
                'output_price_per_million': 2.5,
                'is_active': True
            },
            {
                'model_name': 'gemini-2.5-pro',
                'input_price_per_million': 1.25,
                'output_price_per_million': 10.0,
                'is_active': True
            }
        ]
        
        for pricing in pricing_data:
            TokenCost.objects.get_or_create(
                model_name=pricing['model_name'],
                defaults=pricing
            )
            self.stdout.write(f'Created/updated pricing for {pricing["model_name"]}')
        
        # Create sample usage data for the last 7 days
        self.stdout.write('Creating sample usage data...')
        
        for i in range(7):
            date = timezone.now().date() - timedelta(days=i)
            
            # Create daily summary
            summary, created = DailyTokenSummary.objects.get_or_create(
                date=date,
                defaults={
                    'total_users': random.randint(5, 20),
                    'total_questions': random.randint(10, 50),
                    'total_input_tokens': random.randint(5000, 25000),
                    'total_output_tokens': random.randint(2000, 10000),
                    'total_cost_usd': random.uniform(0.01, 0.10),
                    'total_cost_bdt': random.uniform(1.0, 12.0)
                }
            )
            
            if created:
                self.stdout.write(f'Created daily summary for {date}')
        
        self.stdout.write(
            self.style.SUCCESS('Successfully set up token pricing and sample data!')
        ) 