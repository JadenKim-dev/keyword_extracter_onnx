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

## Testing

### Run Tests

전체 테스트 실행:

```bash
uv run pytest
```

특정 테스트 실행:

```bash
uv run pytest tests/test_adapted_model.py::TestStructuralValidation
```

## Project Structure

```
.
├── scripts/
│   ├── 1_load_model.py      # 베이스 모델 로드
│   └── 2_adapt_model.py     # 모델 아키텍처 적용
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   └── test_adapted_model.py # 모델 검증 테스트
├── models/
│   └── pytorch/
│       └── keyword_model/   # 적용된 모델 저장 위치
└── pyproject.toml          # 프로젝트 설정
```

## Development Workflow

1. 베이스 모델 로드: `uv run python -m scripts.1_load_model`
2. 모델 적용: `uv run python -m scripts.2_adapt_model`
3. 테스트 실행: `pytest tests/ -v`
4. ONNX 변환 (Task 3 - 예정)
