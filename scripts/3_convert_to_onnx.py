#!/usr/bin/env python3
"""
Script 3: Convert PyTorch Model to ONNX Format with Optimization

This script converts the adapted keyword extraction model to ONNX format
with both FP32 and INT8 quantized versions for browser deployment.

Usage:
    uv run python -m scripts.3_convert_to_onnx
    uv run python -m scripts.3_convert_to_onnx --verbose
    uv run python -m scripts.3_convert_to_onnx --force
    uv run python -m scripts.3_convert_to_onnx --skip-quantization
"""

import os
import sys
import json
import argparse
import shutil
from datetime import datetime
import numpy as np
import torch
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

from scripts.utils.display import print_kv_block, log, fmt_millions

# Directory paths
PYTORCH_MODEL_DIR = "./models/pytorch/keyword_model"
ONNX_OUTPUT_DIR = "./models/onnx"
PUBLIC_MODELS_DIR = "./public/models"

# Model names
FP32_MODEL_NAME = "keyword_model_fp32.onnx"
INT8_MODEL_NAME = "keyword_model_int8.onnx"


def check_prerequisites(verbose=False):
    """Check that prerequisites are met before conversion."""
    log("📦 Checking prerequisites...", verbose=verbose, level="INFO")

    if not os.path.exists(PYTORCH_MODEL_DIR):
        raise FileNotFoundError(
            f"PyTorch model not found at {PYTORCH_MODEL_DIR}.\n"
            f"Run 'uv run python -m scripts.2_adapt_model' first."
        )

    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    for filename in required_files:
        filepath = os.path.join(PYTORCH_MODEL_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file missing: {filepath}")

    log("✓ All prerequisites met", verbose=verbose, level="INFO")


def export_to_onnx_fp32(output_dir, verbose=False):
    """
    Export PyTorch model to ONNX format (FP32) using Optimum.

    Args:
        output_dir: Directory to save ONNX model
        verbose: Show detailed output

    Returns:
        Path to exported ONNX model
    """
    print("\n🔧 Exporting to ONNX format (FP32) using Optimum...")

    # Export using Optimum - this handles everything including proper ONNX format
    ort_model = ORTModelForTokenClassification.from_pretrained(
        PYTORCH_MODEL_DIR,
        export=True,
        provider="CPUExecutionProvider"
    )

    # Save to output directory
    ort_model.save_pretrained(output_dir)

    # Optimum always saves as model.onnx
    onnx_file = os.path.join(output_dir, "model.onnx")

    # Verify ONNX model
    log("\n[DETAIL] Verifying ONNX model...", verbose=verbose, level="DETAIL")
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)

    file_size_mb = os.path.getsize(onnx_file) / (1024 ** 2)

    print_kv_block("FP32 Export", {
        "Output path": onnx_file,
        "File size": f"{file_size_mb:.2f} MB",
        "Method": "Optimum ORTModelForTokenClassification",
        "Dynamic axes": "Handled by Optimum"
    })

    return onnx_file


def quantize_to_int8(output_dir, verbose=False):
    """
    Apply INT8 dynamic quantization to ONNX model using Optimum.

    Args:
        output_dir: Directory containing ONNX model
        verbose: Show detailed output

    Returns:
        Path to quantized model
    """
    print("\n🔧 Applying INT8 dynamic quantization using Optimum...")

    log("  Quantization config: ARM64 (dynamic INT8)", verbose=verbose, level="DETAIL")

    # Create quantizer - specify model.onnx explicitly
    quantizer = ORTQuantizer.from_pretrained(output_dir, file_name="model.onnx")

    # Use dynamic quantization configuration optimized for ARM64
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

    # Apply quantization
    quantizer.quantize(
        save_dir=output_dir,
        quantization_config=qconfig,
        file_suffix="int8"
    )

    # Find the quantized model
    int8_path = os.path.join(output_dir, "model_int8.onnx")
    fp32_path = os.path.join(output_dir, "model.onnx")

    if not os.path.exists(int8_path):
        raise FileNotFoundError(f"INT8 model not found at {int8_path}")

    fp32_size_mb = os.path.getsize(fp32_path) / (1024 ** 2)
    int8_size_mb = os.path.getsize(int8_path) / (1024 ** 2)
    size_reduction = (1 - int8_size_mb / fp32_size_mb) * 100

    print_kv_block("INT8 Quantization", {
        "Output path": int8_path,
        "FP32 size": f"{fp32_size_mb:.2f} MB",
        "INT8 size": f"{int8_size_mb:.2f} MB",
        "Size reduction": f"{size_reduction:.1f}%",
        "Target met": "✓ Yes" if int8_size_mb < 100 else "✗ No (>100MB)"
    })

    return int8_path


def copy_to_public_dir(verbose=False):
    """
    Copy ONNX models and tokenizer files to Next.js public directory.

    Args:
        verbose: Show detailed output
    """
    print("\n📂 Copying models to public directory...")

    os.makedirs(PUBLIC_MODELS_DIR, exist_ok=True)

    # Files to copy
    files_to_copy = [
        (os.path.join(ONNX_OUTPUT_DIR, FP32_MODEL_NAME), os.path.join(PUBLIC_MODELS_DIR, FP32_MODEL_NAME)),
        (os.path.join(ONNX_OUTPUT_DIR, INT8_MODEL_NAME), os.path.join(PUBLIC_MODELS_DIR, INT8_MODEL_NAME)),
        (os.path.join(PYTORCH_MODEL_DIR, "config.json"), os.path.join(PUBLIC_MODELS_DIR, "config.json")),
        (os.path.join(PYTORCH_MODEL_DIR, "tokenizer.json"), os.path.join(PUBLIC_MODELS_DIR, "tokenizer.json")),
        (os.path.join(PYTORCH_MODEL_DIR, "vocab.txt"), os.path.join(PUBLIC_MODELS_DIR, "vocab.txt")),
        (os.path.join(PYTORCH_MODEL_DIR, "special_tokens_map.json"), os.path.join(PUBLIC_MODELS_DIR, "special_tokens_map.json")),
        (os.path.join(PYTORCH_MODEL_DIR, "tokenizer_config.json"), os.path.join(PUBLIC_MODELS_DIR, "tokenizer_config.json")),
    ]

    copied_count = 0
    total_size_mb = 0

    for src, dst in files_to_copy:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            file_size_mb = os.path.getsize(dst) / (1024 ** 2)
            total_size_mb += file_size_mb
            copied_count += 1
            log(f"\n[DETAIL] Copied: {os.path.basename(src)} ({file_size_mb:.2f} MB)", verbose=verbose, level="DETAIL")
        else:
            log(f"\n[WARNING] File not found: {src}", verbose=verbose, level="WARNING")

    print_kv_block("Copy Results", {
        "Target directory": PUBLIC_MODELS_DIR,
        "Files copied": f"{copied_count}/{len(files_to_copy)}",
        "Total size": f"{total_size_mb:.2f} MB"
    })


def save_conversion_metadata(fp32_path, int8_path, verbose=False):
    """Save conversion metadata to JSON file."""

    metadata = {
        "conversion_date": datetime.now().isoformat(),
        "pytorch_model": PYTORCH_MODEL_DIR,
        "onnx_output_dir": ONNX_OUTPUT_DIR,
        "opset_version": 17,
        "models": {
            "fp32": {
                "path": fp32_path,
                "size_mb": round(os.path.getsize(fp32_path) / (1024 ** 2), 2)
            },
            "int8": {
                "path": int8_path,
                "size_mb": round(os.path.getsize(int8_path) / (1024 ** 2), 2)
            }
        },
        "validation": {
            "note": "Run 'uv run pytest tests/test_onnx_model.py -v' to validate models"
        },
        "dynamic_axes": ["batch", "seq_len"],
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"]
    }

    metadata_path = os.path.join(ONNX_OUTPUT_DIR, "conversion_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"\n[DETAIL] Metadata saved to: {metadata_path}", verbose=verbose, level="DETAIL")


def convert_to_onnx(*, verbose=False, force=False, skip_quantization=False):
    """
    Main conversion function.

    Args:
        verbose: Show detailed output
        force: Overwrite existing files without prompting
        skip_quantization: Skip INT8 quantization
    """
    # Check prerequisites
    check_prerequisites(verbose=verbose)

    # Check output directory
    if os.path.exists(ONNX_OUTPUT_DIR) and os.listdir(ONNX_OUTPUT_DIR):
        if not force:
            print(f"\n⚠️  Warning: {ONNX_OUTPUT_DIR} already exists and is not empty.")
            response = input("Overwrite existing ONNX models? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
        log(f"Overwriting {ONNX_OUTPUT_DIR}...", verbose=verbose, level="INFO")

    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)

    # Load PyTorch model for validation
    print(f"\n📦 Loading PyTorch model: {PYTORCH_MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(PYTORCH_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(PYTORCH_MODEL_DIR)

    print_kv_block("Model Info", {
        "Architecture": model.config.architectures[0],
        "Num labels": model.config.num_labels,
        "Hidden size": model.config.hidden_size,
        "Parameters": fmt_millions(sum(p.numel() for p in model.parameters()))
    })

    # Export to ONNX (FP32) using Optimum
    fp32_path = export_to_onnx_fp32(ONNX_OUTPUT_DIR, verbose=verbose)

    # Rename to standard name
    fp32_output = os.path.join(ONNX_OUTPUT_DIR, FP32_MODEL_NAME)
    if fp32_path != fp32_output:
        shutil.copy2(fp32_path, fp32_output)
        fp32_path = fp32_output

    # FP32 model created successfully
    print("\n✓ FP32 ONNX model created successfully")
    print(f"  Run 'uv run pytest tests/test_onnx_model.py -v -k fp32' to validate")

    # Quantize to INT8
    int8_path = None
    if not skip_quantization:
        int8_path = quantize_to_int8(ONNX_OUTPUT_DIR, verbose=verbose)

        # Rename to standard name
        int8_output = os.path.join(ONNX_OUTPUT_DIR, INT8_MODEL_NAME)
        if int8_path != int8_output:
            shutil.copy2(int8_path, int8_output)
            int8_path = int8_output

        # INT8 model created successfully
        print("\n✓ INT8 ONNX model created successfully")
        print(f"  Run 'uv run pytest tests/test_onnx_model.py -v -k int8' to validate")
        print("  Note: INT8 quantization may have some accuracy loss compared to FP32")

    # Save metadata
    if int8_path:
        save_conversion_metadata(fp32_path, int8_path, verbose=verbose)

    # Copy to public directory
    copy_to_public_dir(verbose=verbose)

    print("\n✓ ONNX conversion completed successfully!")
    print(f"✓ Models saved to: {os.path.abspath(ONNX_OUTPUT_DIR)}")
    print(f"✓ Public models copied to: {os.path.abspath(PUBLIC_MODELS_DIR)}")
    print("\n✓ Next steps:")
    print("  1. Run tests: uv run pytest tests/test_onnx_model.py -v")
    print("  2. Test in browser with ONNX Runtime Web (Task 5)")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to ONNX format with quantization"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite output directory without prompting"
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip INT8 quantization (FP32 only)"
    )

    args = parser.parse_args()

    try:
        convert_to_onnx(
            verbose=args.verbose,
            force=args.force,
            skip_quantization=args.skip_quantization
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
