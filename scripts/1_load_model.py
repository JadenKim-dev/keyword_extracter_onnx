#!/usr/bin/env python3
"""
Script 1: Load and Inspect Base DistilBERT NER Model (Readable Output)

Usage:
    python scripts/1_load_model.py
    python scripts/1_load_model.py --verbose
    python scripts/1_load_model.py --no-infer
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Add project root to Python path

from scripts.utils.display import (
    kv_block,
    log,
    fmt_millions,
    print_label_mapping,
    print_predictions,
)


MODEL_NAME = "dslim/distilbert-NER"
CACHE_DIR = "./models/pytorch/base_ner"


def load_and_inspect(*, verbose: bool = False, run_infer: bool = True):
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"📦 Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    config = model.config

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Summary blocks (default-visible)
    kv_block("Model", {
        "Architecture": (config.architectures[0] if config.architectures else type(model).__name__),
        "Num labels": config.num_labels,
        "Hidden size": config.hidden_size,
        "Layers": config.num_hidden_layers,
        "Heads": config.num_attention_heads,
        "Params": fmt_millions(total_params),
        "Trainable": fmt_millions(trainable_params),
    })

    kv_block("Tokenizer", {
        "Vocab size": len(tokenizer),
        "Max length": tokenizer.model_max_length,
        "Special tokens": ", ".join(tokenizer.special_tokens_map.keys()),
    })

    # Optional details
    if hasattr(config, "intermediate_size"):
        log(f"\n[DETAIL] Intermediate size: {config.intermediate_size}", verbose=verbose, level="DETAIL")

    print_label_mapping(config.id2label, verbose=verbose)

    # Inference (optional)
    if run_infer:
        test_text = "Google was founded by Larry Page and Sergey Brin in California"
        print(f"\n🧪 Test inference: \"{test_text}\"")

        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)

        log("\n[DETAIL] Tensor shapes", verbose=verbose, level="DETAIL")
        log(f"  input_ids:      {tuple(inputs['input_ids'].shape)}", verbose=verbose, level="DETAIL")
        log(f"  attention_mask: {tuple(inputs['attention_mask'].shape)}", verbose=verbose, level="DETAIL")

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        log(f"\n[DETAIL] Tokens ({len(tokens)})", verbose=verbose, level="DETAIL")
        if verbose:
            log(f"  {tokens}", verbose=verbose, level="DETAIL")

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        log("\n[DETAIL] Logits shape", verbose=verbose, level="DETAIL")
        log(f"  logits: {tuple(logits.shape)}", verbose=verbose, level="DETAIL")

        pred_ids = torch.argmax(logits, dim=-1)[0]
        pred_labels = [config.id2label[i.item()] for i in pred_ids]

        print_predictions(
            tokens,
            pred_labels,
            special_tokens_set=set(tokenizer.all_special_tokens),
            max_items=12,
        )

    # Final short summary
    print(f"\n✓ Cached at: {os.path.abspath(CACHE_DIR)}")
    print("✓ Ready for adaptation to keyword extraction (next script)")

    return tokenizer, model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", action="store_true", help="Show detailed output")
    p.add_argument("--no-infer", action="store_true", help="Skip test inference")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        load_and_inspect(verbose=args.verbose, run_infer=(not args.no_infer))
        print("\nScript completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
