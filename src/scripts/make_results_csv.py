import os
import re
import pandas as pd
import glob

def parse_log_file(filepath):
    steps_data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Look for lines containing step completion info
                # Pattern: completed step: <step>, seconds: <sec>, TFLOP/s/device: <val>, Tokens/s/device: <val>
                if 'completed step:' in line and 'TFLOP/s/device:' in line:
                    try:
                        step_match = re.search(r'completed step:\s*(\d+)', line)
                        tflop_match = re.search(r'TFLOP/s/device:\s*([\d\.]+)', line)
                        token_match = re.search(r'Tokens/s/device:\s*([\d\.]+)', line)
                        
                        if step_match and tflop_match and token_match:
                            step = int(step_match.group(1))
                            if 5 <= step <= 9:
                                tflops = float(tflop_match.group(1))
                                tokens = float(token_match.group(1))
                                steps_data.append({
                                    'tflops': tflops,
                                    'tokens': tokens
                                })
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return steps_data

def main():
    # Define outputs directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, 'outputs')
    
    if not os.path.exists(outputs_dir):
        print(f"Directory not found: {outputs_dir}")
        return

    results = []
    
    # Iterate through all items in outputs directory
    for item in os.listdir(outputs_dir):
        item_path = os.path.join(outputs_dir, item)
        
        # Skip ignored directories
        if item in ['2025-12-12', 'ir_dump', 'profiles']:
            continue
            
        # Determine info from filename/dirname
        # Pattern: <implementation>-num_repeats_<repeats>[-multiprocess_<procs>]-<timestamp>
        match = re.match(r'^(.*)-num_repeats_(\d+)(?:-multiprocess_(\d+))?-.*', item)
        if not match:
            continue
            
        impl_raw = match.group(1)
        num_repeats = int(match.group(2))
        num_procs_str = match.group(3)
        
        # Determine num_processes and log file path
        log_file_path = None
        num_processes = 1
        
        if num_procs_str:
            num_processes = int(num_procs_str)
            if os.path.isdir(item_path):
                # For multiprocess, use the last process log file
                last_proc_idx = num_processes - 1
                candidate_log = os.path.join(item_path, f'process_{last_proc_idx}.log')
                if os.path.exists(candidate_log):
                    log_file_path = candidate_log
        else:
            if os.path.isfile(item_path) and item_path.endswith('.log'):
                log_file_path = item_path
                
        if not log_file_path:
            continue
            
        # Normalize implementation name
        implementation = impl_raw
        if implementation == 'spmd':
            implementation = 'spmd-gpipe'
            
        # Parse data
        data_points = parse_log_file(log_file_path)
        
        if not data_points:
            continue
            
        df_steps = pd.DataFrame(data_points)
        
        results.append({
            'implementation': implementation,
            'num_processes': num_processes,
            'num_repeats': num_repeats,
            'mean-TFLOP/sec/device': df_steps['tflops'].mean(),
            'stdv-TFLOP/sec/device': df_steps['tflops'].std(),
            'mean-Tokens/sec/device': df_steps['tokens'].mean(),
            'stdv-Tokens/sec/device': df_steps['tokens'].std(),
            'Notes': ''
        })
        
    if not results:
        print("No results found.")
        return

    df_results = pd.DataFrame(results)
    
    # Calculate ratios against SPMD-gpipe
    # Create lookup map: (num_processes, num_repeats) -> {tflops_mean, tokens_mean} for spmd-gpipe
    spmd_baselines = {}
    for _, row in df_results.iterrows():
        if row['implementation'] == 'spmd-gpipe':
            key = (row['num_processes'], row['num_repeats'])
            spmd_baselines[key] = {
                'tflops': row['mean-TFLOP/sec/device'],
                'tokens': row['mean-Tokens/sec/device']
            }
            
    df_results['mean_TFLOPs-percent_SPMD'] = df_results.apply(
        lambda x: x['mean-TFLOP/sec/device'] / spmd_baselines.get((x['num_processes'], x['num_repeats']), {}).get('tflops', float('nan')) 
        if spmd_baselines.get((x['num_processes'], x['num_repeats']), {}).get('tflops', 0) != 0 else float('nan'),
        axis=1
    )
    
    df_results['mean_tokens/sec-percent_SPMD'] = df_results.apply(
        lambda x: x['mean-Tokens/sec/device'] / spmd_baselines.get((x['num_processes'], x['num_repeats']), {}).get('tokens', float('nan')) 
        if spmd_baselines.get((x['num_processes'], x['num_repeats']), {}).get('tokens', 0) != 0 else float('nan'),
        axis=1
    )

    # Order columns
    columns = [
        'implementation',
        'num_processes',
        'num_repeats',
        'mean_TFLOPs-percent_SPMD',
        'mean_tokens/sec-percent_SPMD',
        'mean-TFLOP/sec/device',
        'stdv-TFLOP/sec/device',
        'mean-Tokens/sec/device',
        'stdv-Tokens/sec/device',
        'Notes'
    ]
    
    # Custom sort for implementation: spmd-gpipe < mpmd-gpipe < mpmd-1F1B
    impl_order = {'spmd-gpipe': 0, 'mpmd-gpipe': 1, 'mpmd-1F1B': 2}
    df_results['impl_rank'] = df_results['implementation'].map(impl_order)
    
    # Sort by (num_processes, implementation rank, num_repeats)
    df_results = df_results.sort_values(by=['num_processes', 'impl_rank', 'num_repeats'])
    
    # Select final columns
    df_results = df_results[columns]
    
    csv_path = os.path.join(script_dir, 'results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
