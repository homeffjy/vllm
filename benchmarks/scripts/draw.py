import json

import matplotlib.pyplot as plt


# Function to extract QPS from the test name
def extract_qps(test_name):
    # The QPS is indicated at the end of the test name like 'qps_2' or 'qps_inf'
    parts = test_name.split('_')
    for i, part in enumerate(parts):
        if part == 'qps':
            try:
                qps = parts[i + 1]
                if qps == 'inf':
                    return 'inf'
                else:
                    return int(qps)
            except IndexError:
                pass
    # If 'qps' not found or invalid, return None
    return None


# Read vllm.json
with open('../results/2024-09-04_00-46-47_vllm_055_nightly_results.json',
          'r') as f:
    vllm_data = json.load(f)

# Read sglang.json
# with open('sglang.json', 'r') as f:
#     sglang_data = json.load(f)

# Initialize results dictionaries
vllm_results = {}
sglang_results = {}

data_list = [
    (vllm_data, vllm_results),
    #  (sglang_data, sglang_results)
]

# Parse the data and collect metrics
for data, results in data_list:
    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]
    for entry in data:
        test_name = entry.get("Test name", "")
        qps = extract_qps(test_name)
        if qps is not None:
            # Collect data
            ttft = entry.get("Mean TTFT (ms)", 0)
            tpot = entry.get("Mean TPOT (ms)", 0)
            tput_req_s = entry.get("Tput (req/s)", 0)
            qps_str = str(qps)  # Use QPS as string for consistent keys
            if qps_str not in results:
                # Initialize dict for this QPS
                results[qps_str] = {"TTFT": [], "TPOT": [], "Throughput": []}
            # Append data
            results[qps_str]["TTFT"].append(ttft)
            results[qps_str]["TPOT"].append(tpot)
            results[qps_str]["Throughput"].append(tput_req_s)
        else:
            print(f"QPS not found in test name: {test_name}")

# List of QPS values as strings
qps_values = set(list(vllm_results.keys()) + list(sglang_results.keys()))


# Convert numerical QPS strings to integers for sorting, keep 'inf' as is
def qps_sort_key(qps):
    return float('inf') if qps == 'inf' else int(qps)


# Sort QPS values
qps_values_sorted = sorted(qps_values, key=qps_sort_key)

# Prepare data for plotting
vllm_ttft = []
vllm_tpot = []
vllm_tput = []
# sglang_ttft = []
# sglang_tpot = []

for qps in qps_values_sorted:
    # vLLM
    if qps in vllm_results:
        v_ttft_values = vllm_results[qps]["TTFT"]
        v_tpot_values = vllm_results[qps]["TPOT"]
        v_tput_values = vllm_results[qps]["Throughput"]
        vllm_ttft.append(sum(v_ttft_values) / len(v_ttft_values))
        vllm_tpot.append(sum(v_tpot_values) / len(v_tpot_values))
        vllm_tput.append(sum(v_tput_values) / len(v_tput_values))
    else:
        vllm_ttft.append(None)
        vllm_tpot.append(None)
        vllm_tput.append(None)
    # SGLang
    # if qps in sglang_results:
    #     s_ttft_values = sglang_results[qps]["TTFT"]
    #     s_tpot_values = sglang_results[qps]["TPOT"]
    #     sglang_ttft.append(sum(s_ttft_values)/len(s_ttft_values))
    #     sglang_tpot.append(sum(s_tpot_values)/len(s_tpot_values))
    # else:
    #     sglang_ttft.append(None)
    #     sglang_tpot.append(None)

# Figure 1: TTFT vs QPS
plt.figure()
plt.title('ShareGPT, TTFT')
plt.xlabel('QPS')
plt.ylabel('TTFT (ms)')
x_positions = range(len(qps_values_sorted))  # Positions for x-axis
plt.plot(x_positions, vllm_ttft, label='vLLM', marker='o')
# plt.plot(x_positions, sglang_ttft, label='SGLang', marker='x')
plt.xticks(x_positions, qps_values_sorted)
plt.legend()
plt.grid(True)
plt.savefig('ttft_vs_qps.png')

# Figure 2: TPOT vs QPS
plt.figure()
plt.title('ShareGPT, TPOT')
plt.xlabel('QPS')
plt.ylabel('TPOT (ms)')
plt.plot(x_positions, vllm_tpot, label='vLLM', marker='o')
# plt.plot(x_positions, sglang_tpot, label='SGLang', marker='x')
plt.xticks(x_positions, qps_values_sorted)
plt.legend()
plt.grid(True)
plt.savefig('tpot_vs_qps.png')

# Figure 3: TPUT vs QPS
plt.figure()
plt.title('ShareGPT, Thoughput')
plt.xlabel('QPS')
plt.ylabel('TPUT (req/s)')
plt.plot(x_positions, vllm_tput, label='vLLM', marker='o')
# plt.plot(x_positions, sglang_tpot, label='SGLang', marker='x')
plt.xticks(x_positions, qps_values_sorted)
plt.legend()
plt.grid(True)
plt.savefig('tput_vs_qps.png')
