#!/bin/bash

set -o pipefail
set -x

launch_vllm_server() {

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  model=$(echo "$common_params" | jq -r '.model')
  tp=$(echo "$common_params" | jq -r '.tp')
  dataset_name=$(echo "$common_params" | jq -r '.dataset_name')
  dataset_path=$(echo "$common_params" | jq -r '.dataset_path')
  port=$(echo "$common_params" | jq -r '.port')
  num_prompts=$(echo "$common_params" | jq -r '.num_prompts')
  server_args=$(json2args "$server_params")

  if echo "$common_params" | jq -e 'has("fp8")' >/dev/null; then
    echo "Key 'fp8' exists in common params. Use neuralmagic fp8 model for convenience."
    model=$(echo "$common_params" | jq -r '.neuralmagic_quantized_model')
    server_command="python3 \
        -m vllm.entrypoints.openai.api_server \
        -tp $tp \
        --model $model \
        --port $port \
        $server_args"
  else
    echo "Key 'fp8' does not exist in common params."
    server_command="python3 \
        -m vllm.entrypoints.openai.api_server \
        -tp $tp \
        --model $model \
        --port $port \
        $server_args"
  fi

  # run the server
  echo "Server command: $server_command"
  eval "$server_command" &
}

check_gpus() {
  # check the number of GPUs and GPU type.
  declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)
  if [[ $gpu_count -gt 0 ]]; then
    echo "GPU found."
  else
    echo "Need at least 1 GPU to run benchmarking."
    exit 1
  fi
  declare -g gpu_type=$(echo $(nvidia-smi --query-gpu=name --format=csv,noheader) | awk '{print $2}')
  echo "GPU type is $gpu_type"
}

json2args() {
  # transforms the JSON string to command line args, and '_' is replaced to '-'
  # example:
  # input: { "model": "meta-llama/Llama-2-7b-chat-hf", "tensor_parallel_size": 1 }
  # output: --model meta-llama/Llama-2-7b-chat-hf --tensor-parallel-size 1
  local json_string=$1
  local args=$(
    echo "$json_string" | jq -r '
      to_entries |
      map("--" + (.key | gsub("_"; "-")) + " " + (.value | tostring)) |
      join(" ")
    '
  )
  echo "$args"
}

kill_gpu_processes() {
  pkill -f python
  pkill -f python3
  pkill -f tritonserver
  pkill -f pt_main_thread
  pkill -f text-generation
  pkill -f lmdeploy

  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1) -ge 1000 ]; do
    sleep 1
  done
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  timeout 1200 bash -c '
    until curl -s localhost:8000/v1/completions > /dev/null; do
      sleep 1
    done' && return 0 || return 1
}

ensure_installed() {
  # Ensure that the given command is installed by apt-get
  local cmd=$1
  if ! which $cmd >/dev/null; then
    apt-get update && apt-get install -y $cmd
  fi
}

run_serving_tests() {
  # run serving tests using `benchmark_serving.py`
  # $1: a json file specifying serving test cases

  local serving_test_file
  serving_test_file=$1

  # Iterate over serving tests
  jq -c '.[]' "$serving_test_file" | while read -r params; do
    # get the test name, and append the GPU type back to it.
    test_name=$(echo "$params" | jq -r '.test_name')

    # if TEST_SELECTOR is set, only run the test cases that match the selector
    if [[ -n "$TEST_SELECTOR" ]] && [[ ! "$test_name" =~ $TEST_SELECTOR ]]; then
      echo "Skip test case $test_name."
      continue
    fi

    # prepend the current serving engine to the test name
    test_name=${CURRENT_LLM_SERVING_ENGINE}_${test_name}

    # get common parameters
    common_params=$(echo "$params" | jq -r '.common_parameters')
    model=$(echo "$common_params" | jq -r '.model')
    tp=$(echo "$common_params" | jq -r '.tp')
    dataset_name=$(echo "$common_params" | jq -r '.dataset_name')
    dataset_path=$(echo "$common_params" | jq -r '.dataset_path')
    port=$(echo "$common_params" | jq -r '.port')
    num_prompts=$(echo "$common_params" | jq -r '.num_prompts')
    reuse_server=$(echo "$common_params" | jq -r '.reuse_server')

    # get client and server arguments
    server_params=$(echo "$params" | jq -r ".${CURRENT_LLM_SERVING_ENGINE}_server_parameters")
    client_params=$(echo "$params" | jq -r ".${CURRENT_LLM_SERVING_ENGINE}_client_parameters")
    client_args=$(json2args "$client_params")
    qps_list=$(echo "$params" | jq -r '.qps_list')
    qps_list=$(echo "$qps_list" | jq -r '.[] | @sh')
    echo "Running over qps list $qps_list"

    # check if there is enough GPU to run the test
    if [[ $gpu_count -lt $tp ]]; then
      echo "Required num-shard $tp but only $gpu_count GPU found. Skip testcase $test_name."
      continue
    fi

    if [[ $reuse_server == "true" ]]; then
      echo "Reuse previous server for test case $test_name"
    else
      kill_gpu_processes
      
      launch_vllm_server # can be others
    fi

    wait_for_server

    if [ $? -eq 0 ]; then
      echo ""
      echo "$CURRENT_LLM_SERVING_ENGINE server is up and running."
    else
      echo ""
      echo "$CURRENT_LLM_SERVING_ENGINE failed to start within the timeout period."
      break
    fi

    cd $VLLM_SOURCE_CODE_LOC/benchmarks

    # iterate over different QPS
    for qps in $qps_list; do
      # remove the surrounding single quote from qps
      if [[ "$qps" == *"inf"* ]]; then
        echo "qps was $qps"
        qps="inf"
        echo "now qps is $qps"
      fi

      new_test_name=$test_name"_qps_"$qps

      backend=$CURRENT_LLM_SERVING_ENGINE

      if [[ "$backend" == *"vllm"* ]]; then
        backend="vllm"
      fi

      if [[ "$dataset_name" = "sharegpt" ]]; then

        client_command="python3 benchmark_serving.py \
          --backend $backend \
          --tokenizer /tokenizer_cache \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --num-prompts $num_prompts \
          --port $port \
          --save-result \
          --result-dir $RESULTS_FOLDER \
          --result-filename ${new_test_name}.json \
          --request-rate $qps \
          --ignore-eos \
          $client_args"

      elif [[ "$dataset_name" = "ib" ]]; then

        !!!!!!!!!!!!!!!!!!!!

      else
  
        echo "The dataset name must be either 'sharegpt' or 'ib' (InfiniteBench). Got $dataset_name."
        exit 1

      fi

        

      echo "Running test case $test_name with qps $qps"
      echo "Client command: $client_command"

      eval "$client_command"

      server_command="None"

      # record the benchmarking commands
      jq_output=$(jq -n \
        --arg server "$server_command" \
        --arg client "$client_command" \
        --arg gpu "$gpu_type" \
        --arg engine "$CURRENT_LLM_SERVING_ENGINE" \
        '{
          server_command: $server,
          client_command: $client,
          gpu_type: $gpu,
          engine: $engine
        }')
      echo "$jq_output" >"$RESULTS_FOLDER/${new_test_name}.commands"

    done

  done

  kill_gpu_processes
}

main() {

  # check if the environment variable is successfully injected from yaml

  check_gpus
  export CURRENT_LLM_SERVING_ENGINE=vllm
  export VLLM_SOURCE_CODE_LOC=/sgl-workspace/vllm

  pip install -U transformers

  # check storage
  df -h

  ensure_installed wget
  ensure_installed curl
  ensure_installed jq

  cd $VLLM_SOURCE_CODE_LOC/benchmarks
  declare -g RESULTS_FOLDER=results/
  mkdir -p $RESULTS_FOLDER
  BENCHMARK_ROOT=$VLLM_SOURCE_CODE_LOC/benchmarks/

  # run the test
  run_serving_tests $BENCHMARK_ROOT/scripts/tests.json

  python3 $BENCHMARK_ROOT/scripts/summary-nightly-results.py
}

main "$@"
