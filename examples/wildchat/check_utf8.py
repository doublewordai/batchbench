#!/usr/bin/env python3
"""
Check a JSONL file for invalid UTF-8 encoding and identify problematic lines.

This script reads a file and checks for UTF-8 encoding errors, reporting:
- Line numbers with invalid UTF-8
- The specific byte sequences causing issues
- Context around the errors
"""

import argparse
import sys
from pathlib import Path


def check_utf8(file_path: Path, verbose: bool = False):
    """
    Check a file for UTF-8 encoding errors.
    
    Args:
        file_path: Path to the file to check
        verbose: If True, show more detailed error information
    """
    print(f"Checking {file_path} for UTF-8 encoding errors...\n")
    
    errors_found = []
    total_lines = 0
    
    try:
        # First, try to read the entire file as UTF-8
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines = line_num
                if verbose and line_num % 10000 == 0:
                    print(f"Checked {line_num} lines...")
    except UnicodeDecodeError as e:
        print(f"✗ UTF-8 decode error found!")
        print(f"  Error: {e}")
        print(f"  Position: byte {e.start}-{e.end}")
        print(f"  Reason: {e.reason}\n")
        
        # Now check line by line with binary mode to find all errors
        print("Scanning file line-by-line for all errors...\n")
        
        with open(file_path, 'rb') as f:
            for line_num, line_bytes in enumerate(f, 1):
                try:
                    line_bytes.decode('utf-8')
                    if verbose and line_num % 10000 == 0:
                        print(f"Checked {line_num} lines...")
                except UnicodeDecodeError as line_error:
                    errors_found.append({
                        'line_num': line_num,
                        'error': line_error,
                        'line_bytes': line_bytes
                    })
        
        # Report all errors
        print(f"\n{'='*70}")
        print(f"SUMMARY: Found {len(errors_found)} lines with UTF-8 errors")
        print(f"{'='*70}\n")
        
        for i, error_info in enumerate(errors_found, 1):
            line_num = error_info['line_num']
            error = error_info['error']
            line_bytes = error_info['line_bytes']
            
            print(f"Error {i}/{len(errors_found)}:")
            print(f"  Line number: {line_num}")
            print(f"  Error: {error.reason}")
            print(f"  Byte position: {error.start}-{error.end} in line")
            
            # Show the problematic bytes
            bad_bytes = line_bytes[error.start:error.end]
            print(f"  Invalid bytes: {bad_bytes.hex()} ({bad_bytes})")
            
            # Show context around the error
            context_start = max(0, error.start - 50)
            context_end = min(len(line_bytes), error.end + 50)
            context = line_bytes[context_start:context_end]
            
            print(f"  Context (50 chars before/after):")
            try:
                # Try to decode the context, replacing errors
                context_str = context.decode('utf-8', errors='replace')
                print(f"    {repr(context_str)}")
            except:
                print(f"    {context.hex()}")
            
            # Show first 200 chars of the line
            print(f"  Line preview (first 200 bytes):")
            preview = line_bytes[:200].decode('utf-8', errors='replace')
            print(f"    {repr(preview)}")
            print()
        
        return False
    
    print(f"✓ File is valid UTF-8!")
    print(f"  Total lines checked: {total_lines}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the JSONL file to check"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose progress information"
    )
    
    args = parser.parse_args()
    
    if not args.file.exists():
        print(f"Error: File {args.file} does not exist", file=sys.stderr)
        sys.exit(1)
    
    is_valid = check_utf8(args.file, args.verbose)
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
