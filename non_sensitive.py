import json
import pandas as pd

def lambda_handler(event, context):
    try:
        # Extract and validate the data
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

        # Perform operations
        operation = event.get('operation', '')
        if operation == 'average':
            result = df.mean().to_dict()
        elif operation == 'addition':
            result = df.sum().to_dict()
        elif operation == 'multiplication':
            result = df * 2
        else:
            result = {"error": "Invalid operation"}

        return {
            'statusCode': 200,
            'body': json.dumps({'data': result})
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
