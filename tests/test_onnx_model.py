"""
Test suite for ONNX model conversion and functionality.

Run with:
    uv run pytest tests/test_onnx_model.py
    uv run pytest tests/test_onnx_model.py -v
    uv run pytest tests/test_onnx_model.py -v -k "fp32"
"""

import os
import pytest
import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


# Directory paths
ONNX_DIR = "./models/onnx"
PUBLIC_DIR = "./public/models"
PYTORCH_DIR = "./models/pytorch/keyword_model"


class TestONNXConversion:
    """Test ONNX model conversion and basic functionality."""

    def test_onnx_models_exist(self):
        """Test that ONNX models were created."""
        fp32_path = os.path.join(ONNX_DIR, "keyword_model_fp32.onnx")
        int8_path = os.path.join(ONNX_DIR, "keyword_model_int8.onnx")

        assert os.path.exists(fp32_path), "FP32 ONNX model not found"
        assert os.path.exists(int8_path), "INT8 ONNX model not found"

    def test_onnx_runtime_sessions(self):
        """Test that ONNX models can be loaded and inference sessions created."""
        fp32_path = os.path.join(ONNX_DIR, "keyword_model_fp32.onnx")
        int8_path = os.path.join(ONNX_DIR, "keyword_model_int8.onnx")

        # Load and verify FP32 model
        onnx_model_fp32 = onnx.load(fp32_path)
        onnx.checker.check_model(onnx_model_fp32)
        session_fp32 = ort.InferenceSession(fp32_path)
        assert session_fp32 is not None

        # Load and verify INT8 model
        onnx_model_int8 = onnx.load(int8_path)
        onnx.checker.check_model(onnx_model_int8)
        session_int8 = ort.InferenceSession(int8_path)
        assert session_int8 is not None

    def test_model_file_sizes(self):
        """Test that model files are within expected size ranges."""
        fp32_path = os.path.join(ONNX_DIR, "keyword_model_fp32.onnx")
        int8_path = os.path.join(ONNX_DIR, "keyword_model_int8.onnx")

        fp32_size_mb = os.path.getsize(fp32_path) / (1024 ** 2)
        int8_size_mb = os.path.getsize(int8_path) / (1024 ** 2)

        # FP32 should be reasonable size (100-300 MB)
        assert 100 < fp32_size_mb < 300, f"FP32 model size {fp32_size_mb:.2f} MB is outside expected range"

        # INT8 should be smaller than FP32
        assert int8_size_mb < fp32_size_mb, "INT8 model should be smaller than FP32"

        # INT8 should be reasonable size (< 100 MB)
        assert int8_size_mb < 100, f"INT8 model size {int8_size_mb:.2f} MB exceeds 100 MB"

    def test_conversion_metadata_exists(self):
        """Test that conversion metadata file exists."""
        metadata_path = os.path.join(ONNX_DIR, "conversion_metadata.json")
        assert os.path.exists(metadata_path), "Conversion metadata not found"


class TestONNXOutputShape:
    """Test ONNX model output shapes and structure."""

    @pytest.fixture(scope="class")
    def fp32_session(self):
        """Load FP32 ONNX session once for all tests."""
        fp32_path = os.path.join(ONNX_DIR, "keyword_model_fp32.onnx")
        return ort.InferenceSession(fp32_path)

    @pytest.fixture(scope="class")
    def int8_session(self):
        """Load INT8 ONNX session once for all tests."""
        int8_path = os.path.join(ONNX_DIR, "keyword_model_int8.onnx")
        return ort.InferenceSession(int8_path)

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load tokenizer once for all tests."""
        return AutoTokenizer.from_pretrained(PYTORCH_DIR)

    @pytest.mark.parametrize("model_type", ["fp32", "int8"])
    def test_output_shape(self, model_type, tokenizer, request):
        """Test ONNX model output shape for both FP32 and INT8."""
        session_fixture = f"{model_type}_session"
        session = request.getfixturevalue(session_fixture)
        test_text = "Sample text for testing"
        inputs = tokenizer(test_text, return_tensors="np", padding=True, truncation=True)

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        outputs = session.run(None, onnx_inputs)

        # Should have 1 output (logits)
        assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"

        logits = outputs[0]
        batch_size, seq_len, num_labels = logits.shape

        # Check dimensions
        assert batch_size == 1, f"Expected batch_size=1, got {batch_size}"
        assert num_labels == 3, f"Expected num_labels=3, got {num_labels}"
        assert seq_len == inputs["input_ids"].shape[1], "Sequence length mismatch"

    @pytest.mark.parametrize("seq_len", [10, 128, 512])
    def test_dynamic_sequence_lengths(self, fp32_session, tokenizer, seq_len):
        """Test that model handles different sequence lengths (dynamic axes)."""
        # Create text with approximate desired length
        text = " ".join([f"word{i}" for i in range(seq_len // 2)])
        inputs = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=seq_len)

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        outputs = fp32_session.run(None, onnx_inputs)

        logits = outputs[0]
        assert logits.shape[1] == seq_len, f"Expected seq_len={seq_len}, got {logits.shape[1]}"


class TestONNXNumericalAccuracy:
    """Test ONNX model numerical accuracy against PyTorch."""

    @pytest.fixture(scope="class")
    def pytorch_model(self):
        """Load PyTorch model once for all tests."""
        model = AutoModelForTokenClassification.from_pretrained(PYTORCH_DIR)
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load tokenizer once for all tests."""
        return AutoTokenizer.from_pretrained(PYTORCH_DIR)

    @pytest.fixture(scope="class")
    def fp32_session(self):
        """Load FP32 ONNX session once for all tests."""
        fp32_path = os.path.join(ONNX_DIR, "keyword_model_fp32.onnx")
        return ort.InferenceSession(fp32_path)

    def test_fp32_numerical_equivalence(self, pytorch_model, fp32_session: ort.InferenceSession, tokenizer):
        """Test FP32 ONNX matches PyTorch output within tolerance."""
        test_text = "Machine learning and artificial intelligence are transforming technology"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)

        # PyTorch inference
        with torch.no_grad():
            pytorch_outputs = pytorch_model(**inputs)
            pytorch_logits = pytorch_outputs.logits.numpy()

        # ONNX inference
        onnx_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy()
        }
        onnx_logits = fp32_session.run(None, onnx_inputs)[0]

        # Check numerical equivalence
        max_diff = np.abs(pytorch_logits - onnx_logits).max()
        mean_diff = np.abs(pytorch_logits - onnx_logits).mean()

        assert max_diff < 1e-4, f"Max difference {max_diff:.2e} exceeds tolerance 1e-4"
        assert mean_diff < 1e-5, f"Mean difference {mean_diff:.2e} exceeds tolerance 1e-5"

    def test_no_nan_or_inf_values(self, fp32_session: ort.InferenceSession, tokenizer):
        """Test that outputs contain no NaN or Inf values."""
        test_text = "Testing for NaN and Inf values in model outputs"
        inputs = tokenizer(test_text, return_tensors="np", padding=True, truncation=True)

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        outputs = fp32_session.run(None, onnx_inputs)

        logits = outputs[0]
        assert not np.isnan(logits).any(), "NaN values detected in outputs"
        assert not np.isinf(logits).any(), "Inf values detected in outputs"


class TestPublicModels:
    """Test models in public directory for Next.js deployment."""

    def test_public_tokenizer_files_exist(self):
        """Test tokenizer files exist in public/models/ directory."""
        required_files = [
            "config.json",
            "tokenizer.json",
            "vocab.txt",
            "special_tokens_map.json",
            "tokenizer_config.json"
        ]

        for filename in required_files:
            filepath = os.path.join(PUBLIC_DIR, filename)
            assert os.path.exists(filepath), f"Required file {filename} not found in public directory"

    def test_public_models_loadable(self):
        """Test that public ONNX models exist and can be loaded."""
        fp32_path = os.path.join(PUBLIC_DIR, "keyword_model_fp32.onnx")
        int8_path = os.path.join(PUBLIC_DIR, "keyword_model_int8.onnx")

        # Check existence
        assert os.path.exists(fp32_path), "FP32 model not found in public directory"
        assert os.path.exists(int8_path), "INT8 model not found in public directory"

        # Should be able to create inference sessions
        session_fp32 = ort.InferenceSession(fp32_path)
        session_int8 = ort.InferenceSession(int8_path)

        assert session_fp32 is not None
        assert session_int8 is not None

    def test_public_total_size(self):
        """Test total size of public models directory."""
        total_size_mb = 0
        for filename in os.listdir(PUBLIC_DIR):
            filepath = os.path.join(PUBLIC_DIR, filename)
            if os.path.isfile(filepath):
                total_size_mb += os.path.getsize(filepath) / (1024 ** 2)

        # Should be reasonable for deployment (< 400 MB)
        assert total_size_mb < 400, f"Public models directory size {total_size_mb:.2f} MB exceeds 400 MB"
