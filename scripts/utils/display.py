#!/usr/bin/env python3
"""
Display and logging utilities for model scripts.

This module provides reusable functions for formatted output,
logging, and displaying model/tokenizer information.
"""


def print_kv_block(title: str, data: dict) -> None:
    """
    Print a formatted key-value block.

    Args:
        title: Block title to display
        data: Dictionary of key-value pairs to display

    Example:
        kv_block("Model Info", {"Architecture": "BERT", "Layers": 12})
    """
    print(f"\n[{title}]")
    for k, v in data.items():
        print(f"- {k:<22}: {v}")


def log(msg: str, *, verbose: bool = True, level: str = "INFO") -> None:
    """
    Simple log gate: DETAIL messages only show on verbose.

    Args:
        msg: Message to log
        verbose: Whether to show DETAIL level messages
        level: Log level ("INFO" or "DETAIL")

    Example:
        log("Loading model...", verbose=True, level="INFO")
        log("Hidden size: 768", verbose=False, level="DETAIL")  # Won't print
    """
    if level == "DETAIL" and not verbose:
        return
    print(msg)


def fmt_millions(n: int) -> str:
    """
    Format a number in millions with 2 decimal places.

    Args:
        n: Number to format

    Returns:
        Formatted string (e.g., "67.58M")

    Example:
        fmt_millions(67584321)  # Returns "67.58M"
    """
    return f"{n/1e6:.2f}M"


def print_label_mapping(id2label: dict, *, verbose: bool) -> None:
    """
    Print label ID to label name mapping.

    Args:
        id2label: Dictionary mapping label IDs to label names
        verbose: Whether to show detailed output

    Example:
        print_label_mapping({0: "O", 1: "B-PER", 2: "I-PER"}, verbose=True)
    """
    log("\n[Label mapping]", verbose=verbose, level="DETAIL")
    if not verbose:
        return
    for idx in sorted(id2label.keys()):
        log(f"  {idx}: {id2label[idx]}", verbose=verbose, level="DETAIL")


def print_predictions(
    tokens: list,
    labels: list,
    *,
    special_tokens_set: set,
    max_items: int = 12
) -> None:
    """
    Print token-to-label predictions in a formatted way.

    Args:
        tokens: List of tokens
        labels: List of predicted labels (same length as tokens)
        special_tokens_set: Set of special tokens to skip
        max_items: Maximum number of items to display

    Example:
        print_predictions(
            ["hello", "world"],
            ["O", "O"],
            special_tokens_set={"[CLS]", "[SEP]"},
            max_items=10
        )
    """
    print("\n[Sample prediction]")
    shown = 0
    for t, l in zip(tokens, labels):
        if t in special_tokens_set:
            continue
        print(f"- {t:<15} → {l}")
        shown += 1
        if shown >= max_items:
            break
