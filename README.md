# Keyword Extraction Model - ONNX Converter

PyTorch 기반 키워드 추출 모델을 ONNX 포맷으로 변환하는 프로젝트입니다.

## Prerequisites

- Python 3.8+
- uv (Python package manager)

## Installation

프로젝트 의존성 설치:

```bash
uv sync
```

테스트 의존성 포함 설치:

```bash
uv sync --extra dev
```

## Usage

### 1. Load Base Model

DistilBERT 베이스 모델을 로드합니다:

```bash
uv run python -m scripts.1_load_model
```

### 2. Adapt Model

키워드 추출을 위한 모델 아키텍처를 적용합니다:

```bash
uv run python -m scripts.2_adapt_model
```

적용된 모델은 `models/pytorch/keyword_model/` 디렉토리에 저장됩니다.

### 3. Convert to ONNX

PyTorch 모델을 ONNX 포맷으로 변환합니다 (FP32 + INT8 quantized):

```bash
uv run python -m scripts.3_convert_to_onnx
```

변환 옵션:
- `--verbose`, `-v`: 상세 출력
- `--force`, `-f`: 기존 파일 덮어쓰기 (프롬프트 없음)
- `--skip-quantization`: INT8 양자화 건너뛰기 (FP32만 생성)

변환된 모델:
- `models/onnx/keyword_model_fp32.onnx` (248.8 MB) - GPU/WebGL용
- `models/onnx/keyword_model_int8.onnx` (62.6 MB) - CPU/WASM용
- `public/models/` 디렉토리에 자동 복사 (Next.js 배포용)

### 4. Validate Tokenizer

Python transformers와 브라우저 토크나이저 출력을 비교 검증합니다:

```bash
uv run python -m scripts.4_validate_tokenizer --verbose
```

검증 옵션:
- `--verbose`, `-v`: 상세 출력
- `--filter`: 특정 카테고리만 테스트 (basic, padding, truncation, edge_case)
- `--test-name`: 특정 테스트만 실행

검증 결과:
- `tests/validation/tokenizer_validation_python.json` - Python 토크나이저 출력
- 브라우저 비교: `node scripts/compare_tokenizer_outputs.mjs`

## Testing

### Python Tests

전체 테스트 실행:

```bash
uv run pytest
```

특정 테스트 실행:

```bash
uv run pytest tests/test_adapted_model.py::TestStructuralValidation
uv run pytest tests/test_onnx_model.py -v
```

### Browser Tokenizer Tests

Vitest로 브라우저 토크나이저 테스트:

```bash
npm test                           # 전체 테스트
npm test -- lib/__tests__/tokenizer.test.ts   # 토크나이저 테스트만
```

## Project Structure

```
.
├── scripts/
│   ├── 1_load_model.py         # 베이스 모델 로드
│   ├── 2_adapt_model.py        # 모델 아키텍처 적용
│   ├── 3_convert_to_onnx.py    # ONNX 변환 및 양자화
│   ├── 4_validate_tokenizer.py # 토크나이저 검증
│   └── compare_tokenizer_outputs.mjs  # 브라우저 출력 비교
├── tests/
│   ├── conftest.py             # Pytest fixtures
│   ├── test_adapted_model.py   # 모델 검증 테스트
│   ├── test_onnx_model.py      # ONNX 모델 테스트
│   └── validation/             # 토크나이저 검증 결과
│       └── tokenizer_validation_python.json
├── lib/__tests__/
│   └── tokenizer.test.ts       # 브라우저 토크나이저 테스트
├── models/
│   ├── pytorch/
│   │   └── keyword_model/      # 적용된 PyTorch 모델
│   └── onnx/
│       ├── keyword_model_fp32.onnx  # FP32 ONNX 모델
│       ├── keyword_model_int8.onnx  # INT8 양자화 모델
│       └── README.md                # ONNX 모델 문서
├── public/models/              # Next.js 배포용 모델
└── pyproject.toml              # 프로젝트 설정
```

## Development Workflow

1. 베이스 모델 로드: `uv run python -m scripts.1_load_model`
2. 모델 적용: `uv run python -m scripts.2_adapt_model`
3. 테스트 실행: `uv run pytest tests/test_adapted_model.py -v`
4. ONNX 변환: `uv run python -m scripts.3_convert_to_onnx --force`
5. ONNX 테스트: `uv run pytest tests/test_onnx_model.py -v`
6. 토크나이저 검증: `uv run python -m scripts.4_validate_tokenizer --verbose`
7. 브라우저 테스트: `npm test`

## Features

- ✅ DistilBERT 모델 로드 및 검증
- ✅ 키워드 추출용 모델 아키텍처 적용 (3-label classification)
- ✅ ONNX 포맷 변환 (Hugging Face Optimum)
- ✅ INT8 동적 양자화 (74.9% 크기 감소)
- ✅ PyTorch vs ONNX 출력 검증
- ✅ 포괄적인 테스트 스위트
- ✅ 브라우저 토크나이저 구현 (DistilBERT)
- ✅ Python vs 브라우저 토크나이저 검증 (100% 일치)
- 🔄 ONNX Runtime Web 추론 엔진 (Task 5)
- 🔄 키워드 후처리 파이프라인 (Task 6)
- 🔄 Next.js UI 컴포넌트 (Task 7)
