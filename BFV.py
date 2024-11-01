import pandas as pd
import numpy as np
import tenseal as ts
import time
import psutil
from memory_profiler import memory_usage
import gc  # Import garbage collection module

# Sample healthcare dataset
def generate_large_dataset(num_rows):
    data = {
        'patient_id': range(1, num_rows + 1),
        'age': np.random.randint(18, 80, size=num_rows),
        'blood_pressure': np.random.randint(90, 150, size=num_rows),
        'cholesterol': np.random.randint(150, 300, size=num_rows),
        'height': np.random.randint(150, 200, size=num_rows),
        'weight': np.random.randint(50, 100, size=num_rows),
        'heart_rate': np.random.randint(60, 100, size=num_rows),
        'temperature': np.random.uniform(36.0, 37.5, size=num_rows)
    }
    return pd.DataFrame(data)

# Setup TenSEAL context
def setup_tenseal():
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=8192,
        plain_modulus=1032193
    )
    context.generate_galois_keys()
    return context

# Encryption and decryption functions
def encrypt_data(data, context):
    return ts.bfv_vector(context, [int(data)])

def decrypt_data(encrypted_data):
    return encrypted_data.decrypt()[0]

# Process data function
def process_data(df, operation):
    context = setup_tenseal()
    results = df.copy()
    
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            for i in range(len(df)):
                encrypted_value = encrypt_data(df[column][i], context)
                if operation == 'addition':
                    processed_value = encrypted_value + encrypted_value  # Example: doubling the value
                elif operation == 'multiplication':
                    processed_value = encrypted_value * 2  # Example: doubling the value
                results.at[i, column] = decrypt_data(processed_value)
    
    return results
def get_avg_memory_usage(interval=0.1, duration=1):
    mem_usage = memory_usage(interval=interval, timeout=duration)
    return np.mean(mem_usage)

if __name__ == "__main__":
    operation = 'addition'  # Example operation, can be 'multiplication' as well
    num_rows = 6000  # Number of rows in the dataset
    df = generate_large_dataset(num_rows)
    
    # Measure start time and CPU usage
    start_time = time.time()
    process = psutil.Process()
    start_cpu_times = process.cpu_times()
    start_memory = get_avg_memory_usage()
    start_ctx_switches = process.num_ctx_switches()
    
    # Perform garbage collection before starting the main task to minimize fluctuations
    gc.collect()
    
    # Perform processing on the dataset and measure peak memory usage
    peak_memory_usage = memory_usage((process_data, (df, operation)), interval=0.1, max_usage=True)
    processed_df = process_data(df, operation)
    
    # Measure end time and CPU usage
    end_time = time.time()
    end_cpu_times = process.cpu_times()
    end_memory = get_avg_memory_usage()
    end_ctx_switches = process.num_ctx_switches()
    
    # Calculate elapsed time and CPU usage
    elapsed_time = end_time - start_time
    user_cpu_time = end_cpu_times.user - start_cpu_times.user
    system_cpu_time = end_cpu_times.system - start_cpu_times.system
    total_cpu_time = user_cpu_time + system_cpu_time
    cpu_usage = (total_cpu_time / elapsed_time) * 100 if elapsed_time > 0 else 0
    
    memory_usage_diff = end_memory - start_memory
    peak_memory_usage_diff = peak_memory_usage - start_memory
    voluntary_ctx_switches = end_ctx_switches.voluntary - start_ctx_switches.voluntary
    involuntary_ctx_switches = end_ctx_switches.involuntary - start_ctx_switches.involuntary
    
    print("Processed Data:\n", processed_df)
    print(f"Time Elapsed: {elapsed_time:.2f} seconds")
    print(f"CPU Usage: {cpu_usage:.2f}%")
    print(f"Memory Usage: {memory_usage_diff:.2f} MB")
    print(f"Peak Memory Usage: {peak_memory_usage_diff:.2f} MB")
    print(f"Voluntary Context Switches: {voluntary_ctx_switches}")
    print(f"Involuntary Context Switches: {involuntary_ctx_switches}")
