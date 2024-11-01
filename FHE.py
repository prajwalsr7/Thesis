import pandas as pd
import numpy as np
import tenseal as ts
import time
import multiprocessing
import psutil
from memory_profiler import memory_usage
import gc  # Import garbage collection module

# Generate a large sample healthcare dataset
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
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

# Encryption and decryption functions
def encrypt_data(data, context):
    return ts.ckks_vector(context, [float(data)])

def decrypt_data(encrypted_data):
    return encrypted_data.decrypt()[0]

# Worker function for parallel processing of sensitive columns
def process_sensitive_data_worker(df_chunk, sensitive_columns, operation):
    context = setup_tenseal()
    results = df_chunk.copy()
    
    for column in sensitive_columns:
        if df_chunk[column].dtype in ['int64', 'float64']:
            for i in range(len(df_chunk)):
                encrypted_value = encrypt_data(df_chunk[column].iloc[i], context)
                if operation == 'addition':
                    processed_value = encrypted_value + encrypted_value  # Example: doubling the value
                elif operation == 'multiplication':
                    processed_value = encrypted_value * 2  # Example: doubling the value
                results.at[df_chunk.index[i], column] = decrypt_data(processed_value)
    
    return results

# Worker function for parallel processing of non-sensitive columns
def process_non_sensitive_data_worker(df_chunk, non_sensitive_columns, operation):
    results = df_chunk.copy()
    
    for column in non_sensitive_columns:
        if df_chunk[column].dtype in ['int64', 'float64']:
            if operation == 'addition':
                results[column] = df_chunk[column] + df_chunk[column]  # Example: doubling the value
            elif operation == 'multiplication':
                results[column] = df_chunk[column] * 2  # Example: doubling the value
    
    return results

# Main processing function
def process_data_parallel(df, sensitive_columns, operation):
    non_sensitive_columns = [col for col in df.columns if col not in sensitive_columns]

    num_chunks = multiprocessing.cpu_count()  # Number of chunks based on CPU cores
    chunk_size = len(df) // num_chunks
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    with multiprocessing.Pool(processes=num_chunks) as pool:
        sensitive_results = pool.starmap(process_sensitive_data_worker, [(chunk[sensitive_columns], sensitive_columns, operation) for chunk in chunks])
        non_sensitive_results = pool.starmap(process_non_sensitive_data_worker, [(chunk[non_sensitive_columns], non_sensitive_columns, operation) for chunk in chunks])
    
    sensitive_df = pd.concat(sensitive_results, ignore_index=True)
    non_sensitive_df = pd.concat(non_sensitive_results, ignore_index=True)
    
    processed_df = pd.concat([non_sensitive_df, sensitive_df], axis=1).sort_index(axis=1)
    
    return processed_df

def get_avg_memory_usage(interval=0.1, duration=1):
    mem_usage = memory_usage(interval=interval, timeout=duration)
    return np.mean(mem_usage)

if __name__ == "__main__":
    num_rows = 6000  # Number of rows in the dataset
    df = generate_large_dataset(num_rows)
    sensitive_columns = ['age', 'patient_id', 'height', 'weight']  # Specify sensitive columns here
    operation = 'addition'  # Example operation, can be 'multiplication' as well
    
    # Measure start time and CPU usage
    start_time = time.time()
    process = psutil.Process()
    start_cpu_times = process.cpu_times()
    start_memory = get_avg_memory_usage()
    start_ctx_switches = process.num_ctx_switches()

    # Perform garbage collection before starting the main task to minimize fluctuations
    gc.collect()

    # Perform parallel processing on the dataset and measure peak memory usage
    peak_memory_usage = memory_usage((process_data_parallel, (df, sensitive_columns, operation)), interval=0.1, max_usage=True)
    processed_df = process_data_parallel(df, sensitive_columns, operation)
    
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
    
    memory_usage_diff = start_memory - end_memory
    peak_memory_usage_diff = peak_memory_usage - end_memory
    voluntary_ctx_switches = end_ctx_switches.voluntary - start_ctx_switches.voluntary
    involuntary_ctx_switches = end_ctx_switches.involuntary - start_ctx_switches.involuntary
    
    print("Processed Data:\n", processed_df)
    print(f"Time Elapsed: {elapsed_time:.2f} seconds")
    print(f"CPU Usage: {cpu_usage:.2f}%")
    print(f"Memory Usage: {memory_usage_diff:.2f} MB")
    print(f"Peak Memory Usage: {peak_memory_usage_diff:.2f} MB")
    print(f"Voluntary Context Switches: {voluntary_ctx_switches}")
    print(f"Involuntary Context Switches: {involuntary_ctx_switches}")



