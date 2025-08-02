# ===== COMPREHENSIVE DATA CLEANING SCRIPT =====
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple

def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Comprehensive data cleaning function
    
    Returns:
        Tuple[pd.DataFrame, Dict]: Cleaned dataframe and cleaning report
    """
    print("🧹 Starting comprehensive data cleaning...")
    
    # Store original info for comparison
    original_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict()
    }
    
    # 1. Clean column names
    df = clean_column_names(df)
    
    # 2. Clean whitespace in data
    df = clean_whitespace(df)
    
    # 3. Identify and categorize columns
    column_categories = categorize_columns(df)
    
    # 4. Detect and convert datetime columns
    datetime_columns = detect_datetime_columns(df)
    df = convert_datetime_columns(df, datetime_columns)
    
    # 5. Convert numeric columns
    df = convert_numeric_columns(df, column_categories['numeric'])
    
    # 6. Clean categorical columns
    df = clean_categorical_columns(df, column_categories['categorical'])
    
    # 7. Handle missing values (show but don't remove)
    missing_info = analyze_missing_values(df)
    
    # 8. Analyze duplicates (show but don't remove)
    duplicate_info = analyze_duplicates(df)
    
    # Update column categories after datetime conversion
    updated_column_categories = categorize_columns(df)
    
    # Generate cleaning report
    cleaning_report = generate_cleaning_report(df, original_info, updated_column_categories, missing_info, duplicate_info, datetime_columns)
    
    return df, cleaning_report

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names: remove whitespace, replace spaces with underscores, etc."""
    print("📝 Cleaning column names...")
    
    # Store original column names
    original_columns = list(df.columns)
    
    # Clean column names
    new_columns = []
    for col in original_columns:
        # Convert to string if not already
        col_str = str(col)
        
        # Remove leading/trailing whitespace
        col_str = col_str.strip()
        
        # Replace spaces with underscores
        col_str = re.sub(r'\s+', '_', col_str)
        
        # Remove special characters except underscores
        col_str = re.sub(r'[^\w_]', '', col_str)
        
        # Remove multiple underscores
        col_str = re.sub(r'_+', '_', col_str)
        
        # Remove leading/trailing underscores
        col_str = col_str.strip('_')
        
        # Ensure it's not empty
        if not col_str:
            col_str = f"column_{len(new_columns)}"
        
        # Ensure uniqueness
        if col_str in new_columns:
            counter = 1
            original_name = col_str
            while col_str in new_columns:
                col_str = f"{original_name}_{counter}"
                counter += 1
        
        new_columns.append(col_str)
    
    # Apply new column names
    df.columns = new_columns
    
    # Report changes
    changes = []
    for old, new in zip(original_columns, new_columns):
        if old != new:
            changes.append(f"'{old}' → '{new}'")
    
    if changes:
        print(f"✅ Column names cleaned: {len(changes)} changes")
        for change in changes[:5]:  # Show first 5 changes
            print(f"   {change}")
        if len(changes) > 5:
            print(f"   ... and {len(changes) - 5} more")
    else:
        print("✅ Column names already clean")
    
    return df

def clean_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Clean whitespace in string columns"""
    print("🧽 Cleaning whitespace in data...")
    
    cleaned_count = 0
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Store original values
            original_values = df[col].copy()
            
            # Clean whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Count changes
            changes = (original_values != df[col]).sum()
            if changes > 0:
                cleaned_count += changes
    
    print(f"✅ Whitespace cleaned: {cleaned_count} values updated")
    return df

def categorize_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify numeric and categorical columns"""
    print("🔍 Categorizing columns...")
    
    # Get all columns
    all_columns = list(df.columns)
    
    # Identify numeric columns (including those that can be converted)
    numeric_columns = []
    categorical_columns = []
    
    for col in all_columns:
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
            continue
        
        # Try to convert to numeric
        try:
            # Convert to numeric, coercing errors to NaN
            test_conversion = pd.to_numeric(df[col], errors='coerce')
            
            # If more than 50% of values can be converted, consider it numeric
            non_null_count = test_conversion.notna().sum()
            total_count = len(df[col])
            
            if non_null_count / total_count > 0.5:
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)
                
        except:
            categorical_columns.append(col)
    
    # Handle remaining columns
    remaining_columns = [col for col in all_columns if col not in numeric_columns + categorical_columns]
    categorical_columns.extend(remaining_columns)
    
    categories = {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "all": all_columns
    }
    
    print(f"✅ Column categorization complete:")
    print(f"   Numeric columns: {len(numeric_columns)}")
    print(f"   Categorical columns: {len(categorical_columns)}")
    
    return categories

def convert_numeric_columns(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """Convert numeric columns to appropriate types"""
    print("🔢 Converting numeric columns...")
    
    converted_count = 0
    
    for col in numeric_columns:
        try:
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert integers to float if they have decimal places
            if df[col].dtype == 'int64':
                # Check if there are any decimal values
                if df[col].notna().any():
                    # Convert to float for consistency
                    df[col] = df[col].astype(float)
            
            converted_count += 1
            
        except Exception as e:
            print(f"⚠️ Warning: Could not convert column '{col}': {e}")
    
    print(f"✅ Numeric conversion complete: {converted_count} columns converted")
    return df

def detect_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that contain date/time data"""
    print("📅 Detecting date/time columns...")
    
    datetime_columns = []
    datetime_patterns = [
        # Common date patterns
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY, MM/DD/YYYY
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY/MM/DD
        r'\d{1,2}\.\d{1,2}\.\d{2,4}',      # DD.MM.YYYY
        r'\d{4}\.\d{1,2}\.\d{1,2}',        # YYYY.MM.DD
        
        # Time patterns
        r'\d{1,2}:\d{2}(:\d{2})?',         # HH:MM, HH:MM:SS
        r'\d{1,2}:\d{2}(:\d{2})?\s*[AP]M', # HH:MM AM/PM
        
        # Combined date-time patterns
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s+\d{1,2}:\d{2}',  # Date + Time
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}\s+\d{1,2}:\d{2}',    # Date + Time
        
        # Month names
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}',
        r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',
        
        # ISO format
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
        
        # Unix timestamp (if reasonable range)
        r'^\d{10,13}$'  # 10-13 digits (Unix timestamp)
    ]
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Sample the column for pattern matching
            sample_values = df[col].dropna().astype(str).head(100)
            
            if len(sample_values) == 0:
                continue
                
            # Check for date patterns
            pattern_matches = 0
            for pattern in datetime_patterns:
                matches = sample_values.str.contains(pattern, regex=True, na=False)
                pattern_matches += matches.sum()
            
            # If more than 30% of sample values match date patterns, consider it datetime
            if pattern_matches / len(sample_values) > 0.3:
                datetime_columns.append(col)
                print(f"   📅 Detected datetime column: {col}")
    
    print(f"✅ Datetime detection complete: {len(datetime_columns)} datetime columns found")
    return datetime_columns

def convert_datetime_columns(df: pd.DataFrame, datetime_columns: List[str]) -> pd.DataFrame:
    """Convert datetime columns to pandas datetime format"""
    print("🕐 Converting datetime columns...")
    
    converted_count = 0
    conversion_errors = []
    
    for col in datetime_columns:
        try:
            print(f"   Converting {col}...")
            
            # Try different parsing strategies
            converted = False
            
            # Strategy 1: Try pandas to_datetime with infer_datetime_format
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                converted = True
                print(f"     ✅ Converted using pandas auto-detection")
            except:
                pass
            
            # Strategy 2: Try common date formats
            if not converted:
                common_formats = [
                    '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d',
                    '%m-%d-%Y', '%d-%m-%Y', '%Y-%m-%d %H:%M:%S',
                    '%m/%d/%Y %H:%M', '%d/%m/%Y %H:%M',
                    '%Y-%m-%d %H:%M', '%m-%d-%Y %H:%M', '%d-%m-%Y %H:%M',
                    '%b %d, %Y', '%d %b %Y', '%B %d, %Y', '%d %B %Y',
                    '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f'
                ]
                
                for fmt in common_formats:
                    try:
                        df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                        # Check if conversion was successful (not all NaN)
                        if df[col].notna().sum() > 0:
                            converted = True
                            print(f"     ✅ Converted using format: {fmt}")
                            break
                    except:
                        continue
            
            # Strategy 3: Try parsing with dayfirst parameter for DD/MM/YYYY
            if not converted:
                try:
                    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                    if df[col].notna().sum() > 0:
                        converted = True
                        print(f"     ✅ Converted using dayfirst=True")
                except:
                    pass
            
            # Strategy 4: Try parsing with yearfirst parameter for YYYY/MM/DD
            if not converted:
                try:
                    df[col] = pd.to_datetime(df[col], yearfirst=True, errors='coerce')
                    if df[col].notna().sum() > 0:
                        converted = True
                        print(f"     ✅ Converted using yearfirst=True")
                except:
                    pass
            
            if converted:
                converted_count += 1
                # Add datetime features
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_weekday'] = df[col].dt.weekday
                df[f'{col}_quarter'] = df[col].dt.quarter
                
                # Add time features if time component exists
                if df[col].dt.time.notna().any():
                    df[f'{col}_hour'] = df[col].dt.hour
                    df[f'{col}_minute'] = df[col].dt.minute
                    df[f'{col}_second'] = df[col].dt.second
                
                print(f"     📊 Added datetime features for {col}")
            else:
                conversion_errors.append(col)
                print(f"     ❌ Failed to convert {col}")
                
        except Exception as e:
            conversion_errors.append(col)
            print(f"     ❌ Error converting {col}: {e}")
    
    print(f"✅ Datetime conversion complete: {converted_count} columns converted")
    if conversion_errors:
        print(f"⚠️ Failed conversions: {conversion_errors}")
    
    return df

def clean_categorical_columns(df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """Clean categorical columns"""
    print("🏷️ Cleaning categorical columns...")
    
    cleaned_count = 0
    
    for col in categorical_columns:
        if df[col].dtype == 'object':
            # Convert to string and clean
            df[col] = df[col].astype(str)
            
            # Replace empty strings with NaN
            df[col] = df[col].replace(['', 'nan', 'None', 'NULL'], np.nan)
            
            # Clean whitespace
            df[col] = df[col].str.strip()
            
            cleaned_count += 1
    
    print(f"✅ Categorical cleaning complete: {cleaned_count} columns cleaned")
    return df

def analyze_missing_values(df: pd.DataFrame) -> Dict:
    """Analyze missing values without removing them"""
    print("🔍 Analyzing missing values...")
    
    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()
    
    # Find rows with missing values
    rows_with_missing = df[df.isnull().any(axis=1)]
    
    missing_info = {
        "total_missing_values": int(total_missing),
        "columns_with_missing": missing_summary[missing_summary > 0].to_dict(),
        "rows_with_missing": len(rows_with_missing),
        "missing_percentage": (total_missing / (len(df) * len(df.columns))) * 100,
        "missing_rows_sample": rows_with_missing.head(10).to_dict('records') if len(rows_with_missing) > 0 else []
    }
    
    if total_missing == 0:
        print("✅ No missing values found")
    else:
        print(f"📊 Missing values found: {total_missing} total missing values")
        print(f"📊 Rows with missing values: {len(rows_with_missing)}")
        print(f"📊 Missing percentage: {missing_info['missing_percentage']:.2f}%")
        
        # Show columns with missing values
        for col, missing_count in missing_summary[missing_summary > 0].items():
            percentage = (missing_count / len(df)) * 100
            print(f"   {col}: {missing_count} ({percentage:.1f}%)")
    
    return missing_info

def analyze_duplicates(df: pd.DataFrame) -> Dict:
    """Analyze duplicate rows without removing them"""
    print("🔄 Analyzing duplicate rows...")
    
    # Find duplicate rows
    duplicate_rows = df[df.duplicated(keep=False)]
    duplicate_count = len(duplicate_rows)
    
    # Count unique duplicate groups
    unique_duplicates = df[df.duplicated(keep='first')]
    unique_duplicate_count = len(unique_duplicates)
    
    duplicate_info = {
        "total_duplicate_rows": duplicate_count,
        "unique_duplicate_groups": unique_duplicate_count,
        "duplicate_percentage": (duplicate_count / len(df)) * 100 if len(df) > 0 else 0,
        "duplicate_rows_sample": duplicate_rows.head(10).to_dict('records') if duplicate_count > 0 else [],
        "duplicate_summary": duplicate_rows.value_counts().head(5).to_dict() if duplicate_count > 0 else {}
    }
    
    if duplicate_count == 0:
        print("✅ No duplicate rows found")
    else:
        print(f"📊 Duplicate rows found: {duplicate_count} total duplicate rows")
        print(f"📊 Unique duplicate groups: {unique_duplicate_count}")
        print(f"📊 Duplicate percentage: {duplicate_info['duplicate_percentage']:.2f}%")
        
        # Show most common duplicates
        if duplicate_info['duplicate_summary']:
            print("📊 Most common duplicate patterns:")
            for pattern, count in list(duplicate_info['duplicate_summary'].items())[:3]:
                print(f"   Pattern appears {count} times")
    
    return duplicate_info

def generate_cleaning_report(df: pd.DataFrame, original_info: Dict, column_categories: Dict, 
                           missing_info: Dict, duplicate_info: Dict, datetime_columns: List[str]) -> Dict:
    """Generate comprehensive cleaning report"""
    print("📋 Generating cleaning report...")
    
    # Get datetime feature columns (those ending with _year, _month, etc.)
    datetime_feature_columns = [col for col in df.columns if any(suffix in col for suffix in ['_year', '_month', '_day', '_weekday', '_quarter', '_hour', '_minute', '_second'])]
    
    report = {
        "original_shape": original_info["shape"],
        "final_shape": df.shape,
        "column_categories": column_categories,
        "data_types": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
        "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        "missing_values_analysis": missing_info,
        "duplicate_analysis": duplicate_info,
        "datetime_analysis": {
            "datetime_columns": datetime_columns,
            "datetime_feature_columns": datetime_feature_columns,
            "total_datetime_features": len(datetime_feature_columns)
        },
        "cleaning_summary": {
            "total_columns": len(df.columns),
            "numeric_columns": len(column_categories["numeric"]),
            "categorical_columns": len(column_categories["categorical"]),
            "datetime_columns": len(datetime_columns),
            "datetime_feature_columns": len(datetime_feature_columns),
            "total_missing_values": missing_info["total_missing_values"],
            "total_duplicate_rows": duplicate_info["total_duplicate_rows"],
            "columns_with_missing": list(missing_info["columns_with_missing"].keys())
        }
    }
    
    print("✅ Cleaning report generated")
    return report

def print_cleaning_summary(report: Dict):
    """Print a summary of the cleaning process"""
    print("\n" + "="*50)
    print("📊 CLEANING SUMMARY")
    print("="*50)
    
    print(f"📏 Dataset size: {report['original_shape'][0]} → {report['final_shape'][0]} rows")
    print(f"📋 Columns: {report['original_shape'][1]} → {report['final_shape'][1]} columns")
    print(f"🔢 Numeric columns: {report['cleaning_summary']['numeric_columns']}")
    print(f"🏷️ Categorical columns: {report['cleaning_summary']['categorical_columns']}")
    print(f"📅 Datetime columns: {report['cleaning_summary']['datetime_columns']}")
    print(f"📊 Datetime features: {report['cleaning_summary']['datetime_feature_columns']}")
    print(f"❓ Missing values: {report['cleaning_summary']['total_missing_values']}")
    print(f"🔄 Duplicate rows: {report['cleaning_summary']['total_duplicate_rows']}")
    print(f"💾 Memory usage: {report['memory_usage']:.2f} MB")
    
    # Show datetime details
    datetime_info = report['datetime_analysis']
    if datetime_info['datetime_columns']:
        print(f"\n📅 Datetime Processing Details:")
        print(f"   Datetime columns detected: {datetime_info['datetime_columns']}")
        print(f"   Datetime features created: {datetime_info['datetime_feature_columns']}")
        print(f"   Total datetime features: {datetime_info['total_datetime_features']}")
    
    # Show missing value details
    missing_info = report['missing_values_analysis']
    if missing_info['total_missing_values'] > 0:
        print(f"\n⚠️ Missing Values Details:")
        print(f"   Rows with missing values: {missing_info['rows_with_missing']}")
        print(f"   Missing percentage: {missing_info['missing_percentage']:.2f}%")
        print(f"   Columns with missing values: {len(missing_info['columns_with_missing'])}")
    
    # Show duplicate details
    duplicate_info = report['duplicate_analysis']
    if duplicate_info['total_duplicate_rows'] > 0:
        print(f"\n🔄 Duplicate Details:")
        print(f"   Unique duplicate groups: {duplicate_info['unique_duplicate_groups']}")
        print(f"   Duplicate percentage: {duplicate_info['duplicate_percentage']:.2f}%")
    
    print("="*50)

def show_data_quality_issues(df: pd.DataFrame, report: Dict):
    """Show detailed data quality issues to the user"""
    print("\n" + "="*50)
    print("🔍 DATA QUALITY ISSUES")
    print("="*50)
    
    # Show missing value rows
    missing_info = report['missing_values_analysis']
    if missing_info['rows_with_missing'] > 0:
        print(f"\n❓ ROWS WITH MISSING VALUES ({missing_info['rows_with_missing']} rows):")
        if missing_info['missing_rows_sample']:
            print("Sample of rows with missing values:")
            for i, row in enumerate(missing_info['missing_rows_sample'][:5], 1):
                print(f"   Row {i}: {row}")
            if len(missing_info['missing_rows_sample']) > 5:
                print(f"   ... and {len(missing_info['missing_rows_sample']) - 5} more rows")
    
    # Show duplicate rows
    duplicate_info = report['duplicate_analysis']
    if duplicate_info['total_duplicate_rows'] > 0:
        print(f"\n🔄 DUPLICATE ROWS ({duplicate_info['total_duplicate_rows']} rows):")
        if duplicate_info['duplicate_rows_sample']:
            print("Sample of duplicate rows:")
            for i, row in enumerate(duplicate_info['duplicate_rows_sample'][:5], 1):
                print(f"   Row {i}: {row}")
            if len(duplicate_info['duplicate_rows_sample']) > 5:
                print(f"   ... and {len(duplicate_info['duplicate_rows_sample']) - 5} more rows")
    
    print("="*50)

# ===== MAIN FUNCTION =====
def clean_and_prepare_dataset(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to load, clean, and prepare a dataset
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Cleaned dataframe and cleaning report
    """
    print("🚀 Starting dataset cleaning and preparation...")
    
    # Load dataset
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Use Excel (.xlsx) or CSV (.csv)")
        
        print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None, None
    
    # Clean dataset
    cleaned_df, cleaning_report = clean_dataset(df)
    
    # Print summary
    print_cleaning_summary(cleaning_report)
    
    # Show data quality issues
    show_data_quality_issues(cleaned_df, cleaning_report)
    
    return cleaned_df, cleaning_report

# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    # Example usage
    file_path = r"D:\Python code\ollama\GBM_V1.11.xlsx"
    
    cleaned_df, report = clean_and_prepare_dataset(file_path)
    
    if cleaned_df is not None:
        print("\n🎉 Dataset cleaning complete!")
        print(f"✅ Ready for analysis with {len(cleaned_df)} rows and {len(cleaned_df.columns)} columns")
        
        # Show first few rows
        print("\n📋 First 5 rows of cleaned dataset:")
        print(cleaned_df.head())
        
        # Show column info
        print("\n📋 Column information:")
        print(cleaned_df.info()) 