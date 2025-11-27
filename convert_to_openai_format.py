#!/usr/bin/env python3
"""Convert legacy JSONL files to OpenAI Batch API compliant format.

This script converts JSONL files from the old format:
    {"text": "..."}

To the new OpenAI Batch API compliant format:
    {"messages": [{"role": "user", "content": "..."}], "model": "..."}
"""

import argparse
import json
import sys
from pathlib import Path


def convert_file(input_path: Path, output_path: Path, model: str | None = None, overwrite: bool = False) -> None:
    """Convert a single JSONL file to OpenAI Batch API format.
    
    Args:
        input_path: Path to the input JSONL file
        output_path: Path to the output JSONL file
        model: Optional model identifier to include in each record
        overwrite: If True, overwrite existing output file
    """
    if output_path.exists() and not overwrite:
        print(f"Error: Output file {output_path} already exists. Use --overwrite to replace it.", file=sys.stderr)
        sys.exit(1)
    
    converted_count = 0
    skipped_count = 0
    
    with input_path.open('r', encoding='utf-8') as infile, \
         output_path.open('w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON, skipping: {e}", file=sys.stderr)
                skipped_count += 1
                continue
            
            # Check if already in new format
            if "messages" in record:
                # Already in OpenAI format, just pass through (optionally update model)
                if model and "model" not in record:
                    record["model"] = model
                json.dump(record, outfile)
                outfile.write('\n')
                converted_count += 1
                continue
            
            # Convert from old format
            if "text" not in record:
                print(f"Warning: Line {line_num} has neither 'text' nor 'messages' field, skipping", file=sys.stderr)
                skipped_count += 1
                continue
            
            text_content = record["text"]
            
            # Build new record
            new_record = {
                "messages": [
                    {
                        "role": "user",
                        "content": text_content
                    }
                ]
            }
            
            # Add model if specified
            if model:
                new_record["model"] = model
            
            json.dump(new_record, outfile)
            outfile.write('\n')
            converted_count += 1
    
    print(f"Converted {converted_count} records from {input_path} to {output_path}", file=sys.stderr)
    if skipped_count > 0:
        print(f"Skipped {skipped_count} invalid records", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL file or directory containing JSONL files"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSONL file or directory (defaults to input with '_openai' suffix)"
    )
    parser.add_argument(
        "-m", "--model",
        help="Model identifier to include in output records (optional)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process all .jsonl files in directory recursively"
    )
    
    args = parser.parse_args()
    
    input_path = args.input
    
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Handle directory processing
    if input_path.is_dir():
        if not args.recursive:
            print("Error: Input is a directory. Use --recursive to process all JSONL files in it.", file=sys.stderr)
            sys.exit(1)
        
        # Find all JSONL files
        jsonl_files = list(input_path.glob("**/*.jsonl") if args.recursive else input_path.glob("*.jsonl"))
        
        if not jsonl_files:
            print(f"No .jsonl files found in {input_path}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(jsonl_files)} JSONL files to convert", file=sys.stderr)
        
        # Determine output directory
        output_dir = args.output if args.output else input_path / "openai_format"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for jsonl_file in jsonl_files:
            # Preserve directory structure relative to input_path
            relative_path = jsonl_file.relative_to(input_path)
            output_file = output_dir / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"\nConverting {jsonl_file} -> {output_file}", file=sys.stderr)
            convert_file(jsonl_file, output_file, args.model, args.overwrite)
    
    else:
        # Single file processing
        if args.output:
            output_path = args.output
        else:
            # Default: add '_openai' suffix before extension
            output_path = input_path.parent / f"{input_path.stem}_openai{input_path.suffix}"
        
        convert_file(input_path, output_path, args.model, args.overwrite)
    
    print("\nConversion complete!", file=sys.stderr)


if __name__ == "__main__":
    main()
