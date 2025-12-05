#!/usr/bin/env python3
"""
Append ' /no_think' to the end of every user message in wildchat-50k-input.jsonl
and save to wildchat-50k-input-nothink.jsonl
"""
import json
from tqdm import tqdm

input_file = "wildchat-50k-input.jsonl"
output_file = "wildchat-50k-input-nothink.jsonl"

# First, count total lines for progress bar
print("Counting lines...")
with open(input_file, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

print(f"Processing {total_lines} records...")

# Process the file
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in tqdm(infile, total=total_lines, desc="Appending /no_think"):
        record = json.loads(line.strip())
        messages = record["body"]
        
        # Handle both old and new JSONL formats
        # New OpenAI format: modify user messages
        for message in messages['messages']:
            if message.get('role') == 'user':
                message['content'] = message['content'] + ' /no_think'
        
        record["body"] = messages

        # Write modified record
        outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"\nDone! Output saved to {output_file}")
