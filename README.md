# Keyword Extraction Model - ONNX Converter

A project for converting PyTorch-based keyword extraction models to ONNX format.

## Prerequisites

- Python 3.8+
- uv (Python package manager)

## Installation

Install project dependencies:

```bash
uv sync
```

Install with test dependencies:

```bash
uv sync --extra dev
```

## Usage

### 1. Load Pre-trained Model

Load the pre-trained keyphrase extraction model:

```bash
uv run python -m scripts.1_load_model
```

The model will be saved to the `models/pytorch/keyword_model/` directory.

### 2. Convert to ONNX

Convert the PyTorch model to ONNX format (FP32 + INT8 quantized):

```bash
uv run python -m scripts.2_convert_to_onnx
```

Conversion options:
- `--verbose`, `-v`: Verbose output
- `--force`, `-f`: Overwrite existing files without prompting
- `--skip-quantization`: Skip INT8 quantization (generate FP32 only)

Converted models:
- `models/onnx/keyword_model_fp32.onnx` (253 MB) - **Recommended for browser deployment**
- `models/onnx/keyword_model_int8.onnx` (64 MB) - Python only (not supported in browsers)
- Automatically copied to `public/models/` directory (for Next.js deployment)
- `lib/model-version.ts` - Auto-generated version file (for browser cache control)

**⚠️ INT8 Browser Limitation:**
Dynamic INT8 quantization works in Python onnxruntime, but cannot produce accurate inference results in onnxruntime-web (browser) due to limited support for the `DynamicQuantizeLinear` operator. We recommend using the FP32 model for browser deployment.

**Caching Mechanism:**
The conversion script generates a timestamp-based version and saves it to `lib/model-version.ts`. Browsers use this version for model caching, and the version is automatically updated each time the model is regenerated.

### 3. Validate Tokenizer

Compare and validate Python transformers and browser tokenizer outputs:

```bash
uv run python -m scripts.3_validate_tokenizer --verbose
```

Validation options:
- `--verbose`, `-v`: Verbose output
- `--filter`: Test specific categories only (basic, padding, truncation, edge_case)
- `--test-name`: Run specific test only

Validation results:
- `tests/validation/tokenizer_validation_python.json` - Python tokenizer output
- Browser comparison: `node scripts/compare_tokenizer_outputs.mjs`

## Testing

### Python Tests

Run all tests:

```bash
uv run pytest
```

Run specific tests:

```bash
uv run pytest tests/test_adapted_model.py::TestStructuralValidation
uv run pytest tests/test_onnx_model.py -v
```

### Browser Tokenizer Tests

Test browser tokenizer with Vitest:

```bash
npm test                           # All tests
npm test -- lib/__tests__/tokenizer.test.ts   # Tokenizer tests only
```

## Project Structure

```
.
├── scripts/
│   ├── 1_load_model.py         # Load pre-trained model
│   ├── 2_convert_to_onnx.py    # ONNX conversion and quantization
│   ├── 3_validate_tokenizer.py # Tokenizer validation
│   └── compare_tokenizer_outputs.mjs  # Browser output comparison
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── test_adapted_model.py   # Model validation tests
│   ├── test_onnx_model.py      # ONNX model tests
│   └── validation/             # Tokenizer validation results
│       └── tokenizer_validation_python.json
├── lib/__tests__/
│   └── tokenizer.test.ts       # Browser tokenizer tests
├── models/
│   ├── pytorch/
│   │   └── keyword_model/      # Adapted PyTorch model
│   └── onnx/
│       ├── keyword_model_fp32.onnx  # FP32 ONNX model
│       ├── keyword_model_int8.onnx  # INT8 quantized model
│       └── README.md                # ONNX model documentation
├── public/models/              # Models for Next.js deployment
└── pyproject.toml              # Project configuration
```

## Development Workflow

1. Load pre-trained model: `uv run python -m scripts.1_load_model`
2. Run tests: `uv run pytest tests/test_adapted_model.py -v`
3. Convert to ONNX: `uv run python -m scripts.2_convert_to_onnx --force`
4. Test ONNX: `uv run pytest tests/test_onnx_model.py -v`
5. Validate tokenizer: `uv run python -m scripts.3_validate_tokenizer --verbose`
6. Browser tests: `npm test`

## Model Information

**Base Model:** ml6team/keyphrase-extraction-distilbert-inspec

- **Architecture:** DistilBERT (6 layers, 768 hidden size, 12 attention heads)
- **Training:** Pre-trained on Inspec dataset (academic paper abstracts)
- **Task:** Token classification with BIO tagging for keyphrase extraction
- **Labels:** `{0: "B-KEY", 1: "I-KEY", 2: "O"}`
- **Size:** ~67M parameters (~249 MB FP32, ~63 MB INT8 quantized)
- **License:** MIT
- **Paper:** [Simple Unsupervised Keyphrase Extraction using Sentence Embeddings](https://arxiv.org/abs/2112.08547)
