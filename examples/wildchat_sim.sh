python -m batchbench.generate \
  --output data/wildchat_sim.jsonl \
  --count 10000 \
  --prefix-overlap 0.0 \
  --dist-mode lognormal \
  --dist-median 1000 \
  --dist-sigma 1.68 \
  --dist-max 128000 \
  --tokenizer-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

python -m batchbench.online \
  --jsonl data/wildchat_sim.jsonl \
  --users 32 \
  --requests-per-user 10 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --host http://localhost:8000 \
  --random-requests \
  --output-lognorm-mu 6 \
  --output-lognorm-sigma 0.7 \
  --output-lognorm-max 2000 \
  --request-timeout-secs 10000