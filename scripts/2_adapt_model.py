#!/usr/bin/env python3
"""
Script 2: Adapt Base NER Model to Keyword Extraction (No Training)

This script replaces the 9-label NER classification head with a new
3-label keyword classification head (O, B-KEY, I-KEY).

Usage:
    python scripts/2_adapt_model.py
    python scripts/2_adapt_model.py --verbose
    python scripts/2_adapt_model.py --force  # Overwrite without prompt
"""

import os
import sys
import json
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification

from scripts.utils.display import print_kv_block, log, fmt_millions

MODEL_NAME = "dslim/distilbert-NER"
CACHE_DIR = "./models/pytorch/base_ner"
OUTPUT_DIR = "./models/pytorch/keyword_model"

NEW_LABELS = {
    0: "O",      # Outside (not a keyword)
    1: "B-KEY",  # Begin keyword
    2: "I-KEY"   # Inside keyword
}


def adapt_model(*, verbose: bool = False, force: bool = False):
    """
    Main adaptation logic: Replace 9-label NER head with 3-label keyword head.

    Args:
        verbose: Show detailed output
        force: Overwrite output directory without prompting

    Returns:
        Tuple of (adapted_model, tokenizer, metadata)
    """
    # 1. Check prerequisites
    log("📦 Checking prerequisites...", verbose=verbose, level="INFO")

    if not os.path.exists(CACHE_DIR):
        raise FileNotFoundError(
            f"Base model cache not found at {CACHE_DIR}.\n"
            f"Run 'python scripts/1_load_model.py' first to download the base model."
        )

    # 2. Check output directory
    if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        if not force:
            print(f"\n⚠️  Warning: {OUTPUT_DIR} already exists and is not empty.")
            response = input("Overwrite existing model? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
        log(f"Overwriting {OUTPUT_DIR}...", verbose=verbose, level="INFO")

    # 3. Load base model
    print(f"\n📦 Loading base model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    base_model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR
    )

    # 4. Inspect base model
    log("\n[Base Model]", verbose=verbose, level="DETAIL")
    log(f"- Architecture          : {base_model.config.architectures[0]}",
        verbose=verbose, level="DETAIL")
    log(f"- Num labels (original) : {base_model.config.num_labels}", verbose=verbose, level="DETAIL")
    log(f"- Classifier shape      : {base_model.classifier}", verbose=verbose, level="DETAIL")

    # 5. Create new classification head
    print("\n🔧 Adapting model for keyword extraction...")

    hidden_size = base_model.config.hidden_size
    num_labels = len(NEW_LABELS)

    new_classifier = nn.Linear(hidden_size, num_labels, bias=True)
    nn.init.xavier_uniform_(new_classifier.weight)
    nn.init.zeros_(new_classifier.bias)

    log(f"\n[New Classifier]", verbose=verbose, level="DETAIL")
    log(f"- Input features        : {hidden_size}", verbose=verbose, level="DETAIL")
    log(f"- Output features       : {num_labels}", verbose=verbose, level="DETAIL")
    log(f"- Initialization        : Xavier uniform (weights), zeros (bias)", verbose=verbose, level="DETAIL")

    # 6. Replace classification head
    base_model.classifier = new_classifier

    # 7. Update model configuration
    base_model.config.num_labels = num_labels
    base_model.config.id2label = NEW_LABELS
    base_model.config.label2id = {v: k for k, v in NEW_LABELS.items()}

    print_kv_block("Adaptation", {
        "Method": "Classification head replacement",
        "New labels": f"{num_labels} (O, B-KEY, I-KEY)",
        "Initialization": "Xavier uniform",
        "New classifier shape": f"Linear({hidden_size} → {num_labels})"
    })

    # 8. Save adapted model
    print("\n💾 Saving adapted model...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    total_params = sum(p.numel() for p in base_model.parameters())
    model_size_mb = sum(p.numel() * 4 for p in base_model.parameters()) / (1024 ** 2)

    metadata = {
        "base_model": MODEL_NAME,
        "adaptation_date": datetime.now().isoformat(),
        "adaptation_method": "classification_head_replacement",
        "initialization": "xavier_uniform",
        "num_labels": num_labels,
        "label_scheme": NEW_LABELS,
        "architecture": base_model.config.architectures[0],
        "hidden_size": hidden_size,
        "num_layers": base_model.config.num_hidden_layers,
        "num_attention_heads": base_model.config.num_attention_heads,
        "total_parameters": total_params,
        "model_size_mb": round(model_size_mb, 2),
        "vocab_size": tokenizer.vocab_size,
        "max_length": tokenizer.model_max_length
    }

    metadata_path = os.path.join(OUTPUT_DIR, "adaptation_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    saved_files = [f for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]

    print_kv_block("Output", {
        "Directory": OUTPUT_DIR,
        "Files saved": f"{len(saved_files)} (config, model, tokenizer, metadata)",
        "Total size": f"{model_size_mb:.1f} MB",
        "Parameters": fmt_millions(total_params)
    })

    print("\n✓ Model adapted successfully!")
    print("✓ Run tests to validate: pytest tests/ -v")

    return base_model, tokenizer, metadata


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Adapt DistilBERT NER model to keyword extraction (no training)"
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

    args = parser.parse_args()

    try:
        adapt_model(verbose=args.verbose, force=args.force)
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
