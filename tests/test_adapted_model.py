"""
Pytest-based validation tests for adapted keyword extraction model.

Run with:
    pytest tests/
    pytest tests/test_adapted_model.py -v
    pytest tests/test_adapted_model.py -v -k "structure"
"""

import time
import torch
import pytest


# Test cases for parametrized testing
TEST_CASES = [
    pytest.param("Machine learning algorithms", id="short_technical"),
    pytest.param(
        "Python is a high-level programming language widely used for data science and web development.",
        id="medium_paragraph"
    ),
    pytest.param(
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers. "
        "These neural networks attempt to simulate the behavior of the human brain by learning from large "
        "amounts of data. While a neural network with a single layer can still make approximate predictions, "
        "additional hidden layers can help optimize the accuracy. Deep learning is used in many applications "
        "including computer vision, natural language processing, speech recognition, and recommendation systems. "
        "Popular frameworks for deep learning include TensorFlow, PyTorch, and Keras. These tools make it easier "
        "to build and train complex neural network architectures. The field has seen rapid advancement in recent "
        "years thanks to improvements in computational power, particularly GPUs, and the availability of large "
        "datasets for training models. Companies like Google, Facebook, and OpenAI have made significant "
        "contributions to the field through research and open-source tools.",
        id="long_text"
    ),
    pytest.param("", id="empty_string"),
    pytest.param("C++ @username #hashtag <html> & $ % ^", id="special_chars"),
    pytest.param("Python 3.11 released in 2023 with performance improvements", id="numbers"),
    pytest.param(
        "Apple Inc. CEO Tim Cook announced iPhone 15 at an event in California on September 12, 2023.",
        id="entity_heavy"
    ),
    pytest.param(
        "Neural networks are powerful. They can learn complex patterns. The training process requires data.",
        id="multi_sentence"
    ),
    pytest.param(
        "def train_model(X_train, y_train): return model.fit(X_train, y_train)",
        id="code_like"
    ),
    pytest.param(
        "BERT vs bert-base-uncased: DistilBERT achieves 97% of BERT performance",
        id="mixed_case"
    ),
]


class TestStructuralValidation:
    """Test model structure and configuration."""

    def test_output_shape(self, model, tokenizer):
        """Test that model produces correct output shape."""
        test_text = "Sample text for validation"
        inputs = tokenizer(test_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        batch_size, seq_len, num_labels = logits.shape
        assert num_labels == 3, f"Expected 3 labels, got {num_labels}"

    def test_logits_validity(self, model, tokenizer):
        """Test that logits contain no NaN or Inf values."""
        test_text = "Sample text for validation"
        inputs = tokenizer(test_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        assert not torch.isnan(logits).any(), "NaN detected in logits"
        assert not torch.isinf(logits).any(), "Inf detected in logits"

    def test_config_labels(self, model):
        """Test that config has correct number of labels and label mapping."""
        assert model.config.num_labels == 3, f"Expected 3 labels, got {model.config.num_labels}"
        
        expected_id2label = {0: "O", 1: "B-KEY", 2: "I-KEY"}
        assert model.config.id2label == expected_id2label, f"Unexpected id2label: {model.config.id2label}"


class TestPredictions:
    """Test model predictions on diverse inputs."""

    @pytest.mark.parametrize("text", TEST_CASES)
    def test_prediction_on_case(self, model, tokenizer, text):
        """Test that model produces valid predictions for various inputs."""
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Check output shape
        batch_size, seq_len, num_labels = logits.shape
        assert batch_size == 1, f"Expected batch_size=1, got {batch_size}"
        assert num_labels == 3, f"Expected 3 labels, got {num_labels}"

        # Check predictions are valid
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(logits, dim=-1)[0]

        # All predictions should be in valid range
        assert torch.all(pred_ids >= 0), "Invalid prediction: negative ID"
        assert torch.all(pred_ids < 3), "Invalid prediction: ID >= 3"

        # Probabilities should sum to ~1.0
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
            "Probabilities don't sum to 1.0"

    def test_long_text_truncation(self, model, tokenizer):
        """Test that text longer than max_length is handled correctly."""
        # Create text with >512 words
        text = " ".join([f"word{i}" for i in range(1000)])

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )

        # Should be truncated to max_length
        assert inputs["input_ids"].shape[1] == 512, "Text not truncated to max_length"

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        assert logits.shape[1] == 512, "Output not correct length"


class TestONNXReadiness:
    """Test that model is ready for ONNX conversion."""

    def test_standard_hf_format(self, model_dir):
        """Test that model is saved in standard Hugging Face format."""
        import os
        required_files = ["config.json", "tokenizer.json"]

        for filename in required_files:
            filepath = os.path.join(model_dir, filename)
            assert os.path.exists(filepath), f"Required file missing: {filename}"

    def test_config_has_num_labels(self, model):
        """Test that config has num_labels attribute (required for ONNX)."""
        assert hasattr(model.config, "num_labels"), "Config missing num_labels"
        assert model.config.num_labels == 3, "num_labels should be 3"

    def test_input_format_compatible(self, model, tokenizer):
        """Test that model accepts standard input format (input_ids, attention_mask)."""
        text = "Test input"
        inputs = tokenizer(text, return_tensors="pt")

        # Should accept input_ids and attention_mask
        assert "input_ids" in inputs, "Tokenizer doesn't produce input_ids"
        assert "attention_mask" in inputs, "Tokenizer doesn't produce attention_mask"

        # Model should accept these inputs
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        assert hasattr(outputs, "logits"), "Model doesn't return logits"
        assert isinstance(outputs.logits, torch.Tensor), "Logits is not a tensor"

    def test_dynamic_sequence_length(self, model, tokenizer):
        """Test that model handles different sequence lengths (dynamic axes)."""
        texts = [
            "Short",
            "This is a medium length sentence with more words.",
            "This is a much longer sentence that contains many more words to test dynamic sequence length handling."
        ]

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Should produce output matching input length
            assert logits.shape[1] == inputs["input_ids"].shape[1], \
                f"Output length {logits.shape[1]} doesn't match input length {inputs['input_ids'].shape[1]}"


class TestModelMetadata:
    """Test model metadata and configuration."""

    def test_metadata_file_exists(self, model_dir):
        """Test that adaptation metadata file exists."""
        import os
        metadata_path = os.path.join(model_dir, "adaptation_metadata.json")
        assert os.path.exists(metadata_path), "adaptation_metadata.json not found"

    def test_metadata_content(self, model_dir):
        """Test that metadata contains required information."""
        import os
        import json

        metadata_path = os.path.join(model_dir, "adaptation_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        required_keys = [
            "base_model",
            "adaptation_date",
            "adaptation_method",
            "num_labels",
            "label_scheme"
        ]

        for key in required_keys:
            assert key in metadata, f"Metadata missing required key: {key}"

        assert metadata["num_labels"] == 3, "Metadata num_labels should be 3"
        assert metadata["adaptation_method"] == "classification_head_replacement", \
            "Unexpected adaptation method"
