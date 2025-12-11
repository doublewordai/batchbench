# python -m batchbench.generate \
#   --output data/cosine_sim.jsonl \
#   --count 1000 \
#   --prefix-overlap 0.0 \
#   --dist-mode lognormal \
#   --dist-median 15000 \
#   --dist-sigma 1.0 \
#   --dist-max 90000 \
#   --tokenizer-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

python -m batchbench.online \
  --jsonl data/cosine_sim.jsonl \
  --users 32 \
  --requests-per-user 10 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host http://192.222.52.121:8000 \
  --random-requests \
  --output-tokens 300 \
  --output-vary 200 \
  --request-timeout-secs 10000
