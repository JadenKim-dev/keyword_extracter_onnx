#!/usr/bin/env python3
"""
Script 3: Validate Tokenizer Consistency with Browser Implementation

Compares Python transformers tokenizer outputs with browser tokenization
to ensure numerical equivalence.

Usage:
    uv run python -m scripts.3_validate_tokenizer
    uv run python -m scripts.3_validate_tokenizer --verbose
    uv run python -m scripts.3_validate_tokenizer --filter basic
    uv run python -m scripts.3_validate_tokenizer --test-name simple_text
"""

import os
import json
import argparse
from datetime import datetime
from transformers import AutoTokenizer

from scripts.utils.display import print_kv_block, log


# Constants
MODEL_DIR = "./public/models/keyword_model"
OUTPUT_DIR = "./tests/validation"
DEFAULT_OUTPUT_FILE = "tokenizer_validation_python.json"
MAX_LENGTH = 512


# Test cases matching browser test suite
TEST_CASES = [
    {
        "name": "simple_text",
        "category": "basic",
        "input": "Hello world",
        "description": "Simple text with special tokens"
    },
    {
        "name": "single_char",
        "category": "padding",
        "input": "a",
        "description": "Single character - maximum padding"
    },
    {
        "name": "short_word",
        "category": "padding",
        "input": "Hi",
        "description": "Short word - extensive padding"
    },
    {
        "name": "very_long_text",
        "category": "truncation",
        "input": "word " * 1000,
        "description": "Very long text exceeding 512 tokens"
    },
    {
        "name": "unicode_text",
        "category": "edge_case",
        "input": "Hello 世界 🌍",
        "description": "Unicode characters including Chinese and emoji"
    },
    {
        "name": "special_characters",
        "category": "edge_case",
        "input": "Test: @#$%^&*()",
        "description": "Special characters and symbols"
    },
    {
        "name": "whitespace_text",
        "category": "edge_case",
        "input": "Line 1\nLine 2\tTab",
        "description": "Text with newlines and tabs"
    },
]


def load_tokenizer(model_dir: str = MODEL_DIR, verbose: bool = False) -> AutoTokenizer:
    """
    Load tokenizer from same model files used by browser.

    Args:
        model_dir: Path to model directory
        verbose: Show detailed output

    Returns:
        Loaded tokenizer

    Raises:
        AssertionError: If tokenizer configuration is incorrect
    """
    log(f"Loading tokenizer from: {model_dir}", verbose=verbose, level="DETAIL")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Verify configuration matches browser expectations
    assert tokenizer.model_max_length == MAX_LENGTH, \
        f"Expected max_length={MAX_LENGTH}, got {tokenizer.model_max_length}"

    log(f"Tokenizer loaded: {tokenizer.__class__.__name__}", verbose=verbose, level="DETAIL")

    return tokenizer


def tokenize_with_config(tokenizer: AutoTokenizer, text: str) -> dict:
    """
    Tokenize with exact same configuration as browser.

    Configuration matches lib/tokenizer.ts:110-115:
    - padding: 'max_length'
    - truncation: True
    - max_length: 512
    - return_tensors: None (returns Python lists)

    Args:
        tokenizer: Loaded tokenizer
        text: Input text to tokenize

    Returns:
        Dictionary with input_ids, attention_mask, and tokens
    """
    # Call tokenizer with exact browser configuration
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None  # Return Python lists, not tensors
    )

    # Extract arrays (already Python lists)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # Decode each ID to get token strings
    # Matches browser: tokenizer.decode([id], skip_special_tokens=False)
    tokens = [
        tokenizer.decode([token_id], skip_special_tokens=False)
        for token_id in input_ids
    ]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'tokens': tokens
    }


def run_test_case(tokenizer: AutoTokenizer, test_case: dict, verbose: bool = False) -> dict:
    """
    Run a single test case with validation.

    Args:
        tokenizer: Loaded tokenizer
        test_case: Test case dictionary
        verbose: Show detailed output

    Returns:
        Result dictionary with success/error information
    """
    result = {
        'name': test_case['name'],
        'category': test_case['category'],
        'input': test_case['input'],
        'description': test_case['description'],
        'timestamp': datetime.now().isoformat()
    }

    try:
        log(f"  Tokenizing: {test_case['input'][:50]}...", verbose=verbose, level="DETAIL")

        output = tokenize_with_config(tokenizer, test_case['input'])
        result['success'] = True
        result['output'] = output

        # Add validation metadata
        result['validation'] = {
            'length_check': len(output['input_ids']) == MAX_LENGTH,
            'arrays_length_match': (
                len(output['input_ids']) ==
                len(output['attention_mask']) ==
                len(output['tokens']) == MAX_LENGTH
            ),
            'has_cls_token': output['tokens'][0] == '[CLS]',
            'real_token_count': sum(output['attention_mask']),
            'padding_token_count': output['attention_mask'].count(0)
        }

        log(f"  Real tokens: {result['validation']['real_token_count']}, "
            f"Padding: {result['validation']['padding_token_count']}",
            verbose=verbose, level="DETAIL")

    except Exception as e:
        result['success'] = False
        result['error'] = str(e)

        if verbose:
            import traceback
            result['traceback'] = traceback.format_exc()
            log(f"  Error: {e}", verbose=verbose, level="DETAIL")

    return result


def get_filtered_test_cases(
    test_cases: list,
    filter_category: str = None,
    test_name: str = None
) -> list:
    """
    Filter test cases based on criteria.

    Args:
        test_cases: All test cases
        filter_category: Category filter
        test_name: Specific test name

    Returns:
        Filtered list of test cases
    """
    if test_name:
        filtered = [tc for tc in test_cases if tc['name'] == test_name]
        if not filtered:
            raise ValueError(f"Test case '{test_name}' not found")
        return filtered

    if filter_category:
        filtered = [tc for tc in test_cases if tc['category'] == filter_category]
        if not filtered:
            raise ValueError(f"No test cases found for category '{filter_category}'")
        return filtered

    # Default: return all test cases
    return test_cases


def validate_tokenizer(
    *,
    verbose: bool = False,
    output_file: str = DEFAULT_OUTPUT_FILE,
    filter_category: str = None,
    test_name: str = None
) -> dict:
    """
    Main validation function.

    Args:
        verbose: Show detailed output
        output_file: Path to output JSON file
        filter_category: Run only tests in this category
        test_name: Run only this specific test

    Returns:
        Complete results dictionary
    """
    print("🧪 Python Tokenizer Validation Script")
    print("=" * 60)

    # 1. Load tokenizer
    print(f"\n📦 Loading tokenizer from: {MODEL_DIR}")
    tokenizer = load_tokenizer(MODEL_DIR, verbose=verbose)

    print_kv_block("Tokenizer Info", {
        "Model type": tokenizer.__class__.__name__,
        "Vocab size": tokenizer.vocab_size,
        "Max length": tokenizer.model_max_length,
        "CLS token": f"{tokenizer.cls_token} (ID: {tokenizer.cls_token_id})",
        "SEP token": f"{tokenizer.sep_token} (ID: {tokenizer.sep_token_id})",
        "PAD token": f"{tokenizer.pad_token} (ID: {tokenizer.pad_token_id})"
    })

    # 2. Filter test cases
    test_cases = get_filtered_test_cases(
        TEST_CASES,
        filter_category=filter_category,
        test_name=test_name
    )

    print(f"\n🔬 Running {len(test_cases)} test cases...")

    # 3. Run all tests
    test_results = []
    for i, test_case in enumerate(test_cases, 1):
        log(f"\n[{i}/{len(test_cases)}] {test_case['name']}", verbose=verbose, level="INFO")
        result = run_test_case(tokenizer, test_case, verbose=verbose)
        test_results.append(result)

        # Show status
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {test_case['name']}: {test_case['description']}")

    # 4. Compile results
    successful = sum(1 for r in test_results if r['success'])
    failed = len(test_results) - successful

    results = {
        'metadata': {
            'script_version': '1.0.0',
            'generation_date': datetime.now().isoformat(),
            'model_directory': os.path.abspath(MODEL_DIR),
            'tokenizer_config': {
                'model_max_length': tokenizer.model_max_length,
                'padding': 'max_length',
                'truncation': True,
                'max_length': MAX_LENGTH,
                'special_tokens': {
                    'cls_token': tokenizer.cls_token,
                    'sep_token': tokenizer.sep_token,
                    'pad_token': tokenizer.pad_token,
                    'cls_token_id': tokenizer.cls_token_id,
                    'sep_token_id': tokenizer.sep_token_id,
                    'pad_token_id': tokenizer.pad_token_id
                }
            },
            'total_test_cases': len(test_results),
            'successful_tests': successful,
            'failed_tests': failed
        },
        'test_results': test_results
    }

    # 5. Save results
    output_path = os.path.join(OUTPUT_DIR, output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    file_size_kb = os.path.getsize(output_path) / 1024

    print_kv_block("Output", {
        "File path": output_path,
        "File size": f"{file_size_kb:.2f} KB",
        "Total tests": results['metadata']['total_test_cases'],
        "Successful": results['metadata']['successful_tests'],
        "Failed": results['metadata']['failed_tests']
    })

    # 6. Summary
    print("\n" + "=" * 60)
    print(f"✓ Validation complete: {successful}/{len(test_results)} tests passed")

    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Validate tokenizer consistency between Python and browser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  uv run python -m scripts.3_validate_tokenizer

  # Run with verbose output
  uv run python -m scripts.3_validate_tokenizer --verbose

  # Run only edge case tests
  uv run python -m scripts.3_validate_tokenizer --filter edge_case

  # Run single test
  uv run python -m scripts.3_validate_tokenizer --test-name simple_text

  # Custom output file
  uv run python -m scripts.3_validate_tokenizer --output my_results.json
        """
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSON filename (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--filter",
        choices=['basic', 'padding', 'truncation', 'edge_case'],
        help="Run only tests in this category"
    )
    parser.add_argument(
        "--test-name",
        help="Run only this specific test"
    )

    args = parser.parse_args()

    try:
        validate_tokenizer(
            verbose=args.verbose,
            output_file=args.output,
            filter_category=args.filter,
            test_name=args.test_name
        )
        print("\n✓ Script completed successfully!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        raise SystemExit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
