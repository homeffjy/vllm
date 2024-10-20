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
    
    if qps is not None and input_length is not None:
        # Collect data
        ttft = entry.get("Mean TTFT (ms)", 0)
        tpot = entry.get("Mean TPOT (ms)", 0)
        tput_req_s = entry.get("Tput (req/s)", 0)
        data_list.append({
            "Engine": engine,
            "QPS": qps,
            "Input Length": input_length,
            "TTFT (ms)": ttft,
            "TPOT (ms)": tpot,
            "Throughput (req/s)": tput_req_s
        })
    else:
        print(f"QPS or Input Length not found in test name: {test_name}")
        # You can choose to skip this entry or handle it accordingly

# Convert data_list into a DataFrame
df = pd.DataFrame(data_list)

# Filtering out entries with QPS == inf for certain plots
df_filtered = df[df['QPS'] != float('inf')]

# Convert QPS to int for plotting purposes (exclude inf)
df_filtered['QPS'] = df_filtered['QPS'].astype(int)

sns.set(style='whitegrid')

# Plotting TTFT vs Input Length for each Engine, grouped by QPS
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_filtered,
    x='Input Length', y='TTFT (ms)', hue='Engine', style='QPS',
    markers=True, dashes=False
)
plt.title('TTFT vs Input Length')
plt.savefig('ttft_vs_input_length.png')
plt.clf()

# Plotting TPOT vs Input Length for each Engine, grouped by QPS
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_filtered,
    x='Input Length', y='TPOT (ms)', hue='Engine', style='QPS',
    markers=True, dashes=False
)
plt.title('TPOT vs Input Length')
plt.savefig('tpot_vs_input_length.png')
plt.clf()

# Plotting Throughput vs Input Length for each Engine, grouped by QPS
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_filtered,
    x='Input Length', y='Throughput (req/s)', hue='Engine', style='QPS',
    markers=True, dashes=False
)
plt.title('Throughput vs Input Length')
plt.savefig('throughput_vs_input_length.png')
plt.clf()

# Handling QPS == inf separately
df_inf_qps = df[df['QPS'] == float('inf')]
if not df_inf_qps.empty:
    # TTFT vs Input Length at QPS == inf
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_inf_qps,
        x='Input Length', y='TTFT (ms)', hue='Engine',
        markers=True, dashes=False
    )
    plt.title('TTFT vs Input Length at QPS == inf')
    plt.savefig('ttft_vs_input_length_qps_inf.png')
    plt.clf()

    # TPOT vs Input Length at QPS == inf
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_inf_qps,
        x='Input Length', y='TPOT (ms)', hue='Engine',
        markers=True, dashes=False
    )
    plt.title('TPOT vs Input Length at QPS == inf')
    plt.savefig('tpot_vs_input_length_qps_inf.png')
    plt.clf()

    # Throughput vs Input Length at QPS == inf
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_inf_qps,
        x='Input Length', y='Throughput (req/s)', hue='Engine',
        markers=True, dashes=False
    )
    plt.title('Throughput vs Input Length at QPS == inf')
    plt.savefig('throughput_vs_input_length_qps_inf.png')
    plt.clf()

# Additionally, you can plot metrics vs QPS for each Input Length
input_lengths = df_filtered['Input Length'].unique()
for input_length in input_lengths:
    df_il = df_filtered[df_filtered['Input Length'] == input_length]
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_il,
        x='QPS', y='TTFT (ms)', hue='Engine',
        markers=True, dashes=False
    )
    plt.title(f'TTFT vs QPS at Input Length {input_length}')
    plt.savefig(f'ttft_vs_qps_input_length_{input_length}.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_il,
        x='QPS', y='TPOT (ms)', hue='Engine',
        markers=True, dashes=False
    )
    plt.title(f'TPOT vs QPS at Input Length {input_length}')
    plt.savefig(f'tpot_vs_qps_input_length_{input_length}.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_il,
        x='QPS', y='Throughput (req/s)', hue='Engine',
        markers=True, dashes=False
    )
    plt.title(f'Throughput vs QPS at Input Length {input_length}')
    plt.savefig(f'throughput_vs_qps_input_length_{input_length}.png')
    plt.clf()