#!/usr/bin/env python3
"""
Test script to verify chart generation is working properly
"""

import sys
import os

# Add the project to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'analytica'))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'analytica.settings')

import django
django.setup()

from data.views import generate_gemini_response

def test_chart_generation():
    """Test the chart generation with a simple dataset"""
    
    # Mock context with sample data
    context = {
        'user_question': 'provide me a bar chart for gender',
        'dataset_info': {
            'rows': 150,
            'columns': 5,
            'column_names': ['Gender', 'Age', 'Height', 'Weight', 'BMI']
        },
        'cleaning_report': {
            'cleaning_summary': {
                'numeric_columns': ['Age', 'Height', 'Weight', 'BMI'],
                'total_missing_values': 0,
                'total_duplicate_rows': 0
            }
        },
        'full_data': [
            {'Gender': 'Male', 'Age': 25, 'Height': 175, 'Weight': 70, 'BMI': 22.9},
            {'Gender': 'Female', 'Age': 30, 'Height': 165, 'Weight': 60, 'BMI': 22.0},
            {'Gender': 'Male', 'Age': 28, 'Height': 180, 'Weight': 75, 'BMI': 23.1},
            {'Gender': 'Female', 'Age': 35, 'Height': 160, 'Weight': 55, 'BMI': 21.5},
            {'Gender': 'Male', 'Age': 32, 'Height': 178, 'Weight': 72, 'BMI': 22.7}
        ],
        'value_counts': {
            'Gender': {'Male': 95, 'Female': 51}
        }
    }
    
    print("Testing chart generation...")
    print("=" * 50)
    
    try:
        response = generate_gemini_response(context)
        print("AI Response:")
        print(response)
        print("=" * 50)
        
        # Check if response contains clean ASCII charts
        if '█' in response and '(' in response:
            print("✅ Chart generation appears to be working!")
            print("Found ASCII bar chart characters (█) and count format")
        else:
            print("❌ Chart generation may not be working properly")
            print("No ASCII chart characters found in response")
            
    except Exception as e:
        print(f"❌ Error testing chart generation: {e}")

if __name__ == "__main__":
    test_chart_generation() 