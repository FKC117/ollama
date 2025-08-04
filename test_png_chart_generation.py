import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

def generate_chart_png(chart_type, data, title=None, x_label=None, y_label=None):
    """
    Generate a PNG chart and return it as a base64 encoded string
    
    Args:
        chart_type (str): Type of chart ('bar', 'pie', 'line', 'histogram', 'scatter')
        data (dict): Data for the chart
        title (str): Chart title
        x_label (str): X-axis label
        y_label (str): Y-axis label
    
    Returns:
        str: Base64 encoded PNG image
    """
    try:
        # Clear any existing plots
        plt.clf()
        
        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'bar':
            categories = list(data.keys())
            values = list(data.values())
            
            # Create bar chart
            bars = plt.bar(categories, values, color='skyblue', edgecolor='navy', alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        elif chart_type == 'pie':
            categories = list(data.keys())
            values = list(data.values())
            
            # Create pie chart
            plt.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
        
        elif chart_type == 'line':
            x_values = list(data.keys())
            y_values = list(data.values())
            
            # Create line chart
            plt.plot(x_values, y_values, marker='o', linewidth=2, markersize=6)
        
        elif chart_type == 'histogram':
            # For histogram, data should be a list of values
            plt.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
        
        elif chart_type == 'scatter':
            # For scatter, data should be {'x': [...], 'y': [...]}
            plt.scatter(data['x'], data['y'], alpha=0.6)
        
        # Set title and labels
        if title:
            plt.title(title, fontsize=14, fontweight='bold', pad=20)
        if x_label:
            plt.xlabel(x_label, fontsize=12)
        if y_label:
            plt.ylabel(y_label, fontsize=12)
        
        # Rotate x-axis labels if they're long
        if chart_type in ['bar', 'line']:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        
        # Clear the plot
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

def detect_chart_request(user_question, value_counts):
    """
    Detect if the user is requesting a chart and determine the type
    
    Args:
        user_question (str): User's question
        value_counts (dict): Available value counts data
    
    Returns:
        dict: Chart request info or None
    """
    question_lower = user_question.lower()
    
    # Chart type keywords
    chart_keywords = {
        'bar': ['bar chart', 'bar graph', 'bar plot', 'distribution', 'count'],
        'pie': ['pie chart', 'pie graph', 'percentage', 'proportion'],
        'line': ['line chart', 'line graph', 'trend', 'time series'],
        'histogram': ['histogram', 'frequency', 'distribution'],
        'scatter': ['scatter plot', 'scatter graph', 'correlation']
    }
    
    # Detect chart type
    detected_type = None
    for chart_type, keywords in chart_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            detected_type = chart_type
            break
    
    if not detected_type:
        return None
    
    # Find relevant data column
    relevant_column = None
    for col in value_counts.keys():
        if col.lower() in question_lower or any(word in col.lower() for word in question_lower.split()):
            relevant_column = col
            break
    
    # If no specific column found, use the first available
    if not relevant_column and value_counts:
        relevant_column = list(value_counts.keys())[0]
    
    if relevant_column and relevant_column in value_counts:
        return {
            'type': detected_type,
            'column': relevant_column,
            'data': value_counts[relevant_column]
        }
    
    return None

def test_png_chart_generation():
    """Test PNG chart generation with sample data"""
    
    print("Testing PNG Chart Generation")
    print("=" * 50)
    
    # Sample data
    sample_data = {
        'Male': 95,
        'Female': 51
    }
    
    # Test chart generation
    print("Testing bar chart generation...")
    chart_base64 = generate_chart_png(
        chart_type='bar',
        data=sample_data,
        title='Gender Distribution',
        x_label='Gender',
        y_label='Count'
    )
    
    if chart_base64:
        print("✅ Chart generated successfully!")
        print(f"Base64 length: {len(chart_base64)} characters")
        print(f"First 100 chars: {chart_base64[:100]}...")
        
        # Test pie chart
        print("\nTesting pie chart generation...")
        pie_chart_base64 = generate_chart_png(
            chart_type='pie',
            data=sample_data,
            title='Gender Distribution'
        )
        
        if pie_chart_base64:
            print("✅ Pie chart generated successfully!")
            print(f"Pie chart base64 length: {len(pie_chart_base64)} characters")
        else:
            print("❌ Pie chart generation failed!")
            
    else:
        print("❌ Chart generation failed!")
    
    # Test chart request detection
    print("\nTesting chart request detection...")
    test_questions = [
        "provide me a bar chart for gender",
        "show me a pie chart of the distribution",
        "what is the data like?",
        "create a bar graph for age"
    ]
    
    value_counts = {'Gender': sample_data, 'Age': {'20-30': 50, '31-40': 45}}
    
    for question in test_questions:
        result = detect_chart_request(question, value_counts)
        if result:
            print(f"✅ '{question}' -> {result['type']} chart for {result['column']}")
        else:
            print(f"❌ '{question}' -> No chart detected")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_png_chart_generation() 