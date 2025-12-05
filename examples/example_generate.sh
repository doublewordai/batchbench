python -m batchbench.generate \
  --output data/wildchat.jsonl \
  --count 10 \
  --prefix-overlap 0.0 \
  --dist-mode lognormal \
  --dist-median 1000 \
  --dist-sigma 1.68 \
  --dist-max 128000 \
  --tokenizer-model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8