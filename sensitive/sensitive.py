import json
import tenseal as ts
import pandas as pd

def setup_tenseal():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

def encrypt_value(value, context):
    return ts.ckks_vector(context, [float(value)])

def decrypt_value(encrypted_value):
    return encrypted_value.decrypt()[0]

def process_sensitive_data(df, sensitive_columns, operation):
    context = setup_tenseal()
    results = df.copy()

    for column in sensitive_columns:
        if df[column].dtype in ['int64', 'float64']:
            if operation == 'average':
                encrypted_sum = encrypt_value(0, context)
                count = len(df)
                for i in range(count):
                    encrypted_value = encrypt_value(df[column].iloc[i], context)
                    encrypted_sum += encrypted_value
                encrypted_avg = encrypted_sum * (1.0 / count)
                results[column] = decrypt_value(encrypted_avg)
            elif operation == 'addition':
                encrypted_sum = encrypt_value(0, context)
                for i in range(len(df)):
                    encrypted_value = encrypt_value(df[column].iloc[i], context)
                    encrypted_sum += encrypted_value
                results[column] = decrypt_value(encrypted_sum)
            elif operation == 'multiplication':
                for i in range(len(df)):
                    encrypted_value = encrypt_value(df[column].iloc[i], context)
                    processed_value = encrypted_value * 2
                    results.at[df.index[i], column] = decrypt_value(processed_value)

    return results

def lambda_handler(event, context):
    try:
        # Extract and validate data
        data = event.get('data', {})
        if 'columns' not in data or 'data' not in data:
            raise ValueError("Invalid data format")

        columns = data['columns']
        rows = data['data']

        # Validate that all rows have the same length
        if not all(len(row) == len(columns) for row in rows):
            raise ValueError("Mismatch between number of columns and data length")

        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        # Extract operation and sensitive columns
        operation = event.get('operation', '')
        sensitive_columns = event.get('sensitive_columns', [])

        # Process sensitive data
        result_df = process_sensitive_data(df, sensitive_columns, operation)
        
        return {
            'statusCode': 200,
            'body': json.dumps({'data': result_df.to_dict(orient='split')})
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }

