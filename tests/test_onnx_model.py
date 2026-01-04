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
PUBLIC_DIR = "./public/models/keyword_model"
PYTORCH_DIR = "./models/pytorch/keyword_model"


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer once for entire test module."""
    return AutoTokenizer.from_pretrained(PYTORCH_DIR)


@pytest.fixture(scope="module")
def pytorch_model():
    """Load PyTorch model once for entire test module."""
    model = AutoModelForTokenClassification.from_pretrained(PYTORCH_DIR)
    model.eval()
    return model


@pytest.fixture(scope="module")
def fp32_session():
    """Load FP32 ONNX session once for entire test module."""
    fp32_path = os.path.join(ONNX_DIR, "keyword_model_fp32.onnx")
    return ort.InferenceSession(fp32_path)


@pytest.fixture(scope="module")
def int8_session():
    """Load INT8 ONNX session once for entire test module."""
    int8_path = os.path.join(ONNX_DIR, "keyword_model_int8.onnx")
    return ort.InferenceSession(int8_path)


@pytest.fixture(scope="module")
def public_fp32_session():
    """Load public FP32 ONNX session for web deployment testing."""
    fp32_path = os.path.join(PUBLIC_DIR, "keyword_model_fp32.onnx")
    return ort.InferenceSession(fp32_path)


@pytest.fixture(scope="module")
def public_int8_session():
    """Load public INT8 ONNX session for web deployment testing."""
    int8_path = os.path.join(PUBLIC_DIR, "keyword_model_int8.onnx")
    return ort.InferenceSession(int8_path)


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


class TestBIOLabelPredictions:
    """Test that ONNX model correctly predicts all BIO label types."""

    def _get_predicted_labels(self, logits: np.ndarray) -> np.ndarray:
        """
        Extract predicted labels from logits using argmax.

        Args:
            logits: Shape [batch_size, seq_len, num_classes]

        Returns:
            predicted_labels: Shape [batch_size, seq_len] with values in [0, 1, 2]
        """
        return np.argmax(logits, axis=-1)

    def _analyze_predictions(self, predicted_labels: np.ndarray, attention_mask: np.ndarray) -> dict:
        """
        Analyze predicted label distribution (excluding padding).

        Returns:
            Dictionary with label counts and percentages
        """
        # Filter out padding tokens (where attention_mask == 0)
        valid_predictions = predicted_labels[attention_mask == 1]

        counts = {
            'O': np.sum(valid_predictions == 0),
            'B_KEY': np.sum(valid_predictions == 1),
            'I_KEY': np.sum(valid_predictions == 2),
        }

        total = len(valid_predictions)
        percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in counts.items()}

        return {
            'counts': counts,
            'percentages': percentages,
            'total_tokens': total
        }

    @pytest.mark.parametrize("model_type,session_fixture", [
        ("fp32", "fp32_session"),
        ("int8", "int8_session"),
        ("public_fp32", "public_fp32_session"),
        ("public_int8", "public_int8_session"),
    ])
    def test_all_bio_labels_present(self, model_type, session_fixture, tokenizer, request):
        """
        Test that model predicts all three BIO labels (O, B-KEY, I-KEY).

        This is critical for proper keyword extraction. Without B-KEY labels,
        the post-processing pipeline cannot identify keyword boundaries.
        """
        session = request.getfixturevalue(session_fixture)

        # Use keyword-rich text that should trigger all label types
        test_text = (
            "Machine learning and artificial intelligence are transforming "
            "natural language processing and computer vision technologies. "
            "Deep learning models and neural networks enable advanced "
            "data analysis and predictive analytics applications."
        )

        inputs = tokenizer(test_text, return_tensors="np", padding=True, truncation=True)

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        outputs = session.run(None, onnx_inputs)
        logits = outputs[0]

        # Get predicted labels
        predicted_labels = self._get_predicted_labels(logits)

        # Analyze predictions
        analysis = self._analyze_predictions(predicted_labels, inputs["attention_mask"])

        # Print detailed analysis for debugging
        print(f"\n{model_type} Prediction Analysis:")
        print(f"  Total valid tokens: {analysis['total_tokens']}")
        print(f"  O (Outside):     {analysis['counts']['O']:3d} tokens ({analysis['percentages']['O']:5.1f}%)")
        print(f"  B-KEY (Begin):   {analysis['counts']['B_KEY']:3d} tokens ({analysis['percentages']['B_KEY']:5.1f}%)")
        print(f"  I-KEY (Inside):  {analysis['counts']['I_KEY']:3d} tokens ({analysis['percentages']['I_KEY']:5.1f}%)")

        # Critical assertions
        assert analysis['counts']['O'] > 0, (
            f"{model_type}: No O (Outside) labels predicted! Model may be broken."
        )
        assert analysis['counts']['B_KEY'] > 0, (
            f"{model_type}: No B-KEY labels predicted! This will break keyword extraction. "
            f"Got {analysis['counts']['O']} O labels and {analysis['counts']['I_KEY']} I-KEY labels only."
        )
        assert analysis['counts']['I_KEY'] > 0, (
            f"{model_type}: No I-KEY labels predicted! Model may not detect multi-token keywords."
        )

        # Sanity check: B-KEY should be less frequent than I-KEY (keywords span multiple tokens)
        # This is a soft warning, not a hard failure
        if analysis['counts']['B_KEY'] > analysis['counts']['I_KEY']:
            print(f"  WARNING: More B-KEY than I-KEY labels - unusual pattern detected")

    @pytest.mark.parametrize("model_type,session_fixture", [
        ("fp32", "fp32_session"),
        ("int8", "int8_session"),
    ])
    def test_bio_label_consistency_with_pytorch(self, model_type, session_fixture, tokenizer, pytorch_model, request):
        """
        Test that ONNX predictions match PyTorch label distribution.

        This ensures the conversion didn't break the classification behavior.
        """
        session = request.getfixturevalue(session_fixture)

        test_text = "Machine learning and deep learning enable artificial intelligence applications"

        # PyTorch inference
        inputs_pt = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            pytorch_outputs = pytorch_model(**inputs_pt)
            pytorch_logits = pytorch_outputs.logits.numpy()

        pytorch_labels = self._get_predicted_labels(pytorch_logits)
        pytorch_analysis = self._analyze_predictions(pytorch_labels, inputs_pt["attention_mask"].numpy())

        # ONNX inference
        inputs_onnx = tokenizer(test_text, return_tensors="np", padding=True, truncation=True)
        onnx_inputs = {
            "input_ids": inputs_onnx["input_ids"],
            "attention_mask": inputs_onnx["attention_mask"]
        }
        onnx_logits = session.run(None, onnx_inputs)[0]
        onnx_labels = self._get_predicted_labels(onnx_logits)
        onnx_analysis = self._analyze_predictions(onnx_labels, inputs_onnx["attention_mask"])

        print(f"\n{model_type} vs PyTorch Label Distribution:")
        print(f"  PyTorch - O: {pytorch_analysis['counts']['O']}, B-KEY: {pytorch_analysis['counts']['B_KEY']}, I-KEY: {pytorch_analysis['counts']['I_KEY']}")
        print(f"  ONNX    - O: {onnx_analysis['counts']['O']}, B-KEY: {onnx_analysis['counts']['B_KEY']}, I-KEY: {onnx_analysis['counts']['I_KEY']}")

        # Check that predictions mostly match (allow small quantization differences for INT8)
        tolerance = 0.1 if model_type == "int8" else 0.05  # 10% tolerance for INT8, 5% for FP32

        for label in ['O', 'B_KEY', 'I_KEY']:
            pytorch_pct = pytorch_analysis['percentages'][label]
            onnx_pct = onnx_analysis['percentages'][label]
            diff_pct = abs(pytorch_pct - onnx_pct)

            assert diff_pct < (tolerance * 100), (
                f"{model_type}: {label} label distribution differs too much. "
                f"PyTorch: {pytorch_pct:.1f}%, ONNX: {onnx_pct:.1f}%, Diff: {diff_pct:.1f}%"
            )


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
