cargo run --bin batchbench -- \
  --jsonl data/requests_count-2000_tokens-512_prefix-0p10_tokenizer-Qwen-Qwen3-30B-A3B-Instruct-2507-FP8.jsonl \
  --users 200 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --host http://89.25.97.3:11651 \
  --requests-per-user 1 \
  --request-timeout-secs 100000 \
  --output-tokens 2048
