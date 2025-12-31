/**
 * Type definitions for ONNX inference module
 */

/**
 * Supported ONNX execution providers in order of preference
 */
export type ExecutionProvider = 'webgpu' | 'webgl' | 'wasm';

/**
 * Available model variants
 */
export type ModelVariant = 'fp32' | 'int8';

/**
 * Inference input format (from tokenizer)
 */
export interface InferenceInput {
  /** Array of token IDs (integers) */
  input_ids: number[];
  /** Array indicating which tokens should be attended to (1) vs padding (0) */
  attention_mask: number[];
}

/**
 * Raw ONNX model output (logits)
 */
export interface InferenceOutput {
  /** Model output logits as Float32Array */
  logits: Float32Array;
  /** Shape of logits tensor [batch, seq_len, num_labels] */
  shape: [number, number, number];
}

/**
 * Session initialization options
 */
export interface SessionOptions {
  /** Preferred model variant (default: INT8) */
  modelVariant?: ModelVariant;
  /** Execution providers to use (default: ['webgpu', 'webgl', 'wasm']) */
  executionProviders?: ExecutionProvider[];
  /** Graph optimization level (default: 'all') */
  graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
  /** Log severity level (0=verbose, 4=fatal, default: 2=warning) */
  logSeverityLevel?: 0 | 1 | 2 | 3 | 4;
}

/**
 * Error codes for inference operations
 */
export enum InferenceErrorCode {
  /** Session not initialized - call warmup() first */
  NOT_INITIALIZED = 'NOT_INITIALIZED',
  /** Failed to load model file */
  MODEL_LOAD_FAILED = 'MODEL_LOAD_FAILED',
  /** No compatible execution backend available */
  BACKEND_NOT_AVAILABLE = 'BACKEND_NOT_AVAILABLE',
  /** Invalid input dimensions or types */
  INVALID_INPUT = 'INVALID_INPUT',
  /** Runtime error during inference execution */
  INFERENCE_FAILED = 'INFERENCE_FAILED',
  /** Inference called in SSR environment */
  SSR_ERROR = 'SSR_ERROR',
}

/**
 * Custom error class for inference operations
 */
export class InferenceError extends Error {
  constructor(
    message: string,
    public readonly code: InferenceErrorCode,
    public readonly originalError?: Error
  ) {
    super(message, { cause: originalError });
    this.name = 'InferenceError';
  }
}
