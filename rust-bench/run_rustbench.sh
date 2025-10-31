python -m batchbench.online \
    --jsonl data/requests_count-2000_tokens-1024_prefix-0p10_tokenizer-Qwen-Qwen3-30B-A3B-Instruct-2507-FP8.jsonl \
    --model Qwen/Qwen3-4B \
    --host https://35vpm6rpjj4gvk-8080.proxy.runpod.net \
    --requests-per-user 1 \
    --users 8 \
    --request-timeout-secs 1000000 \
    --output-tokens 10 \
    --output-vary 1
