# 运行步骤
1. 选择调整 `test.json` 中的参数
2. 切换到对应的 conda 环境中分别运行
```
./run-benchmark.sh vllm
./run-benchmark.sh sglang
```
3. 生成结果在 `../results` 文件夹内，运行 `python summary.py` 综合数据
4. 运行 `python draw.py` 生成图片
# 部分参数
## vllm
- gpu_memory_utilization: Fraction of GPU memory to use for the vLLM execution
- max_num_seqs: Maximum number of sequences to be processed in a single iteration.
- enable_chunked_prefill: If True, prefill requests can be chunked based on the remaining max_num_batched_tokens.
- max_num_batched_tokens: Maximum number of tokens to be processed in a single iteration.
- swap_space: The size (GiB) of CPU memory per GPU to use as swap space. This can be used for temporarily storing the states of the requests when their `best_of` sampling parameters are larger than 1. If all requests will have `best_of=1`, you can safely set this to 0. Otherwise, too small values may cause out-of-memory (OOM) errors.
- dtype:  Data type for model weights and activations. The "auto" option will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.
## sglang
- mem_fraction_static: The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.
- max_running_request: The maximum number of running requests.
- chunked_prefill_size: The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill