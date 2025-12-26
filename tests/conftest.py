"""
Pytest configuration and fixtures for model validation tests.
"""

import os
import pytest
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "./models/pytorch/keyword_model"


@pytest.fixture(scope="session")
def model_dir():
    """Provide model directory path."""
    if not os.path.exists(MODEL_DIR):
        pytest.skip(
            f"Adapted model not found at {MODEL_DIR}. "
            f"Run 'python scripts/2_adapt_model.py' first."
        )
    return MODEL_DIR


@pytest.fixture(scope="session")
def tokenizer(model_dir):
    """Load tokenizer once per test session."""
    return AutoTokenizer.from_pretrained(model_dir)


@pytest.fixture(scope="session")
def model(model_dir):
    """Load model once per test session."""
    return AutoModelForTokenClassification.from_pretrained(model_dir)
