#!/usr/bin/env python3
"""
Script to fix whitespace issues on blank lines (W293).

This script removes trailing whitespace from lines that contain only whitespace,
which is a common linting issue (ruff W293: blank line contains whitespace).

Usage:
    python scripts/fix_whitespace.py [--check] [--verbose] [path]

    --check: Only check for issues, don't fix them
    --verbose: Show detailed output
    path: Directory or file to process (default: current directory)
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


def fix_whitespace_in_file(filepath: Path, check_only: bool = False, verbose: bool = False) -> Tuple[bool, int]:
    """
    Fix whitespace issues in a single file.

    Args:
        filepath: Path to the file to process
        check_only: If True, only check for issues without fixing
        verbose: If True, print detailed information

    Returns:
        Tuple of (has_issues, num_issues_found)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except (UnicodeDecodeError, IOError) as e:
        if verbose:
            print(f"  ⚠️  Skipping {filepath}: {e}")
        return False, 0

    original_lines = lines.copy()
    issues_found = 0

    # Pattern for lines with only whitespace (spaces, tabs)
    whitespace_only_pattern = re.compile(r'^[ \t]+$')

    for i, line in enumerate(lines):
        if whitespace_only_pattern.match(line):
            issues_found += 1
            if verbose:
                print(f"  Line {i+1}: Found whitespace-only line")
            if not check_only:
                # Replace with empty line (just newline)
                lines[i] = '\n' if line.endswith('\n') else ''

    if issues_found > 0:
        if check_only:
            print(f"❌ {filepath}: {issues_found} whitespace issue(s) found")
        else:
            # Write the fixed content back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"✅ {filepath}: Fixed {issues_found} whitespace issue(s)")
        return True, issues_found
    elif verbose:
        print(f"✓ {filepath}: No whitespace issues")

    return False, 0


def should_process_file(filepath: Path) -> bool:
    """
    Determine if a file should be processed.

    Args:
        filepath: Path to check

    Returns:
        True if the file should be processed
    """
    # Skip hidden files and directories
    if any(part.startswith('.') for part in filepath.parts):
        return False

    # Skip common directories
    skip_dirs = {
        '__pycache__', 'node_modules', 'venv', '.venv', 
        'env', '.env', 'build', 'dist', '.git', '.pytest_cache',
        '.mypy_cache', '.ruff_cache', 'htmlcov', '.tox',
        'docs/_build', 'docs/.jupyter_cache'
    }
    if any(part in skip_dirs for part in filepath.parts):
        return False

    # Only process Python files
    return filepath.suffix == '.py'


def process_directory(path: Path, check_only: bool = False, verbose: bool = False) -> Tuple[int, int]:
    """
    Process all Python files in a directory recursively.

    Args:
        path: Directory path to process
        check_only: If True, only check for issues without fixing
        verbose: If True, print detailed information

    Returns:
        Tuple of (files_with_issues, total_issues)
    """
    files_with_issues = 0
    total_issues = 0

    if path.is_file():
        if should_process_file(path):
            has_issues, num_issues = fix_whitespace_in_file(path, check_only, verbose)
            if has_issues:
                files_with_issues += 1
                total_issues += num_issues
    else:
        # Process directory recursively
        for filepath in sorted(path.rglob('*.py')):
            if should_process_file(filepath):
                has_issues, num_issues = fix_whitespace_in_file(filepath, check_only, verbose)
                if has_issues:
                    files_with_issues += 1
                    total_issues += num_issues

    return files_with_issues, total_issues


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fix whitespace issues on blank lines (W293)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix all whitespace issues in current directory
  python scripts/fix_whitespace.py

  # Check for issues without fixing them
  python scripts/fix_whitespace.py --check

  # Fix issues in a specific file
  python scripts/fix_whitespace.py path/to/file.py

  # Fix issues with verbose output
  python scripts/fix_whitespace.py --verbose
        """
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Path to file or directory to process (default: current directory)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Only check for issues without fixing them'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path '{path}' does not exist")
        sys.exit(1)

    mode = "Checking" if args.check else "Fixing"
    print(f"{mode} whitespace issues in: {path}")
    print("-" * 50)

    files_with_issues, total_issues = process_directory(path, args.check, args.verbose)

    print("-" * 50)
    if args.check:
        if files_with_issues > 0:
            print(f"❌ Found {total_issues} whitespace issue(s) in {files_with_issues} file(s)")
            print("Run without --check to fix these issues automatically")
            sys.exit(1)
        else:
            print("✅ No whitespace issues found")
    else:
        if files_with_issues > 0:
            print(f"✅ Fixed {total_issues} whitespace issue(s) in {files_with_issues} file(s)")
        else:
            print("✅ No whitespace issues found")

    sys.exit(0)


if __name__ == '__main__':
    main()