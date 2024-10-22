import json
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_folder = Path("../results")


# Function to extract QPS from the test name
def extract_qps(test_name):
    match = re.search(r'qps_([^_]+)', test_name)
    if match:
        qps_str = match.group(1)
        if qps_str == 'inf':
            return float('inf')
        else:
            try:
                return float(qps_str)
            except ValueError:
                return None
    else:
        return None


# Function to extract Input Length from the test name
def extract_input_length(test_name):
    match = re.search(r'in_(\d+)', test_name)
    if match:
        return int(match.group(1))
    else:
        return None

# Function to determine if 'prefix' is in the test name
def is_prefix_in_test_name(test_name):
    return 'prefix' in test_name

# Read vllm.json files
vllm_files = list(results_folder.glob('2*vllm*.json'))
if not vllm_files:
    raise FileNotFoundError("No file containing 'vllm' found in the directory.")
vllm_file = vllm_files[0]
with open(vllm_file, 'r') as f:
    vllm_data = json.load(f)

# Read sglang.json files
sglang_files = list(results_folder.glob('2*sglang*.json'))
if not sglang_files:
    raise FileNotFoundError("No file containing 'sglang' found in the directory.")
sglang_file = sglang_files[0]
with open(sglang_file, 'r') as f:
    sglang_data = json.load(f)

# Combine the data into a single list
all_data = vllm_data + sglang_data

# Initialize an empty list to collect data
data_list = []

# Parse the data and collect metrics
for entry in all_data:
    test_name = entry.get("Test name", "")
    engine = entry.get("Engine", "")

    qps = extract_qps(test_name)
    input_length = entry.get("Input Length", None)
    if input_length is None:
        input_length = extract_input_length(test_name)

    is_prefix = is_prefix_in_test_name(test_name)

    if qps is not None and input_length is not None:
        # Collect data
        ttft = entry.get("Mean TTFT (ms)", 0)
        tpot = entry.get("Mean TPOT (ms)", 0)
        output_tput_tok_s = entry.get("Output Tput (tok/s)", 0)
        data_list.append({
            "Engine": engine,
            "QPS": qps,
            "Input Length": input_length,
            "TTFT (ms)": ttft,
            "TPOT (ms)": tpot,
            "Output Throughput (tok/s)": output_tput_tok_s,
            "Prefix": 'Prefix' if is_prefix else 'Non-Prefix'  # Use string for better labels
        })
    else:
        print(f"QPS or Input Length not found in test name: {test_name}")

# Convert data_list into a DataFrame
df = pd.DataFrame(data_list)

# Convert 'Prefix' to string for plotting purposes
df['Prefix'] = df['Prefix'].astype(str)

# Filtering out entries with QPS == inf for certain plots
df_filtered = df[df['QPS'] != float('inf')]

# Drop entries with missing QPS values
df_filtered = df_filtered.dropna(subset=['QPS'])

# Convert QPS to float for plotting purposes
df_filtered['QPS'] = df_filtered['QPS'].astype(float)

sns.set(style='whitegrid')

unique_qps_values = sorted(df_filtered['QPS'].unique())

# Plotting TTFT vs Input Length for each QPS, comparing Prefix and Non-Prefix
for qps in unique_qps_values:
    df_qps = df_filtered[df_filtered['QPS'] == qps]
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_qps,
        x='Input Length',
        y='TTFT (ms)',
        hue='Prefix',
        style='Engine',
        markers=True,
        dashes=False,
        
    )
    plt.title(f'TTFT vs Input Length at QPS {qps}')
    plt.xlabel('Input Length')
    plt.ylabel('TTFT (ms)')
    plt.legend(title='Prefix / Engine')
    plt.savefig(f'ttft_vs_input_length_qps_{qps}.png')
    plt.clf()

# Plotting TPOT vs Input Length, comparing Prefix and Non-Prefix
for qps in unique_qps_values:
    df_qps = df_filtered[df_filtered['QPS'] == qps]
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_qps,
        x='Input Length',
        y='TPOT (ms)',
        hue='Prefix',
        style='Engine',
        markers=True,
        dashes=False,
        
    )
    plt.title(f'TPOT vs Input Length at QPS {qps}')
    plt.xlabel('Input Length')
    plt.ylabel('TPOT (ms)')
    plt.legend(title='Prefix / Engine')
    plt.savefig(f'tpot_vs_input_length_qps_{qps}.png')
    plt.clf()
    

# Plotting Output Throughput vs Input Length for each QPS, comparing Prefix and Non-Prefix
for qps in unique_qps_values:
    df_qps = df_filtered[df_filtered['QPS'] == qps]
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_qps,
        x='Input Length',
        y='Output Throughput (tok/s)',
        hue='Prefix',
        style='Engine',
        markers=True,
        dashes=False,
        
    )
    plt.title(f'Output Throughput vs Input Length at QPS {qps}')
    plt.xlabel('Input Length')
    plt.ylabel('Output Throughput (tok/s)')
    plt.legend(title='Prefix / Engine')
    plt.savefig(f'output_throughput_vs_input_length_qps_{qps}.png')
    plt.clf()

# Additionally, plot metrics vs QPS for each Input Length, comparing Prefix and Non-Prefix
input_lengths = sorted(df_filtered['Input Length'].unique())
for input_length in input_lengths:
    df_il = df_filtered[df_filtered['Input Length'] == input_length]
    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_il,
        x='QPS', y='TTFT (ms)', hue='Prefix', style='Engine', markers=True, dashes=False
    )
    plt.title(f'TTFT vs QPS at Input Length {input_length}')
    plt.savefig(f'ttft_vs_qps_input_length_{input_length}_prefix_comparison.png')
    plt.clf()

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_il,
        x='QPS', y='TPOT (ms)', hue='Prefix', style='Engine', markers=True, dashes=False
    )
    plt.title(f'TPOT vs QPS at Input Length {input_length}')
    plt.savefig(f'tpot_vs_qps_input_length_{input_length}_prefix_comparison.png')
    plt.clf()

    plt.figure(figsize=(12, 8))
    sns.lineplot(
        data=df_il,
        x='QPS', y='Output Throughput (tok/s)', hue='Prefix', style='Engine', markers=True, dashes=False
    )
    plt.title(f'Output Throughput vs QPS at Input Length {input_length}')
    plt.savefig(f'output_throughput_vs_qps_input_length_{input_length}_prefix_comparison.png')
    plt.clf()