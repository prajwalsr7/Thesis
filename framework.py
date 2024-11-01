import pandas as pd
import requests
import json

# Sample dataset definition
data = {
    'patient_id': [1, 2, 3, 4, 5],
    'age': [30, 45, 60, 25, 50],
    'blood_pressure': [120, 130, 140, 110, 135],
    'cholesterol': [200, 220, 250, 180, 230],
    'height': [170, 165, 180, 175, 160],
    'weight': [70, 80, 90, 60, 75],
    'heart_rate': [72, 78, 85, 68, 80],
    'temperature': [36.6, 36.7, 36.8, 36.5, 36.6]
}

# Create DataFrame
df_sample = pd.DataFrame(data)

def df_to_dict(df):
    """Convert DataFrame to dictionary format for Lambda."""
    return {
        'columns': df.columns.tolist(),
        'data': df.values.tolist()
    }

def prepare_data_for_lambda(df, sensitive_columns, operation):
    """Prepare data for sending to the Lambda function."""
    return {
        'data': df_to_dict(df),
        'sensitive_columns': sensitive_columns,
        'operation': operation
    }

def process_data(df, sensitive_columns, operation):
    """Process data by sending requests to Lambda functions and combining results."""
    data = prepare_data_for_lambda(df, sensitive_columns, operation)
    non_sensitive_url = 'https://d1ci2kx43g.execute-api.eu-west-1.amazonaws.com/non_sensitive/'
    sensitive_url = 'https://1gjf7bj3f0.execute-api.eu-west-1.amazonaws.com/sensitive/'
    
    try:
        # Send request to process non-sensitive data
        response_non_sensitive = requests.post(non_sensitive_url, json=data)
        response_non_sensitive.raise_for_status()  # Raise an exception for HTTP errors
        non_sensitive_results = response_non_sensitive.json()
        
        # Debugging: Print the non-sensitive results
        print("Non-sensitive results:", non_sensitive_results)
        
        # Check if the response has the expected data
        non_sensitive_body = json.loads(non_sensitive_results.get('body', '{}'))
        if 'data' not in non_sensitive_body:
            raise ValueError("Non-sensitive response does not contain 'data'")
        
        # Process non-sensitive data
        non_sensitive_data = non_sensitive_body['data']
        if isinstance(non_sensitive_data, dict):
            # Convert the single row data into a DataFrame with the appropriate columns
            non_sensitive_df = pd.DataFrame([non_sensitive_data], columns=non_sensitive_data.keys())
        else:
            raise ValueError("Non-sensitive data is not in the expected format")
        
        # Send request to process sensitive data
        response_sensitive = requests.post(sensitive_url, json=data)
        response_sensitive.raise_for_status()  # Raise an exception for HTTP errors
        sensitive_results = response_sensitive.json()
        
        # Debugging: Print the sensitive results
        print("Sensitive results:", sensitive_results)
        
        # Check if the response has the expected data
        sensitive_body = json.loads(sensitive_results.get('body', '{}'))
        if 'data' not in sensitive_body:
            raise ValueError("Sensitive response does not contain 'data'")
        
        # Extract and process sensitive data
        sensitive_data = sensitive_body['data']
        sensitive_df = pd.DataFrame(
            sensitive_data['data'],
            columns=sensitive_data['columns'],
            index=sensitive_data.get('index', [])
        )
        
        # Combine non-sensitive and sensitive results
        combined_df = pd.concat([non_sensitive_df, sensitive_df], axis=1)
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]  # Remove duplicate columns

        return combined_df
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except ValueError as e:
        print(f"Data format error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    sensitive_columns = ['age', 'blood_pressure', 'cholesterol']
    operation = 'average'  # Choose the operation: 'average', 'addition', or 'multiplication'
    
    # Process data and handle the result
    processed_df = process_data(df_sample, sensitive_columns, operation)
    
    if processed_df is not None:
        print("Processed Data (first 5 rows):\n", processed_df.head())
    else:
        print("Data processing failed.")

