import * as ort from 'onnxruntime-web';
import { isFunction } from 'es-toolkit';
import type {
  ExecutionProvider,
  ModelVariant,
  InferenceInput,
  InferenceOutput,
  SessionOptions,
} from './types/inference';
import { InferenceError, InferenceErrorCode } from './types/inference';

// Constants
const MODEL_BASE_PATH = '/models/keyword_model';
const MODEL_FILES: Record<ModelVariant, string> = {
  fp32: 'keyword_model_fp32.onnx',
  int8: 'keyword_model_int8.onnx',
};
const DEFAULT_EXECUTION_PROVIDERS: ExecutionProvider[] = ['webgpu', 'wasm'];
const WARMUP_SEQUENCE_LENGTH = 128;
const MAX_SEQUENCE_LENGTH = 512;

// Error messages
const NOT_INITIALIZED_ERROR = 'Inference session not initialized. Call warmup() first.';
const SSR_ERROR = 'Inference can only be used in browser environment';
const INVALID_INPUT_LENGTH = 'input_ids and attention_mask must have same length';
const SEQUENCE_TOO_LONG = `Input sequence length exceeds maximum (${MAX_SEQUENCE_LENGTH})`;

/**
 * ONNX Inference Session Manager
 *
 * Manages an ONNX Runtime Web session for keyword extraction inference.
 * Supports multi-backend execution (WebGPU → WebGL → WASM) with automatic fallback.
 */
class OnnxInferenceSession {
  private session: ort.InferenceSession | null = null;
  private loadingPromise: Promise<void> | null = null;

  /**
   * Initialize the ONNX inference session (lazy loading)
   * Uses promise caching to prevent race conditions
   */
  private async initializeSession(options: SessionOptions): Promise<void> {
    // Already loaded
    if (this.session) return;

    // Currently loading - wait for existing promise
    if (this.loadingPromise) {
      await this.loadingPromise;
      return;
    }

    // Start loading
    this.loadingPromise = (async () => {
      try {
        const modelVariant = options.modelVariant || 'int8';
        // Step 1: Load model as ArrayBuffer
        const modelBuffer = await this.loadModel(modelVariant);

        // Step 2: Configure session options
        const sessionConfig: ort.InferenceSession.SessionOptions = {
          executionProviders: (options.executionProviders || DEFAULT_EXECUTION_PROVIDERS),
          graphOptimizationLevel: options.graphOptimizationLevel || 'all',
          logSeverityLevel: options.logSeverityLevel ?? 2,
        };

        // Step 3: Create InferenceSession
        this.session = await ort.InferenceSession.create(modelBuffer, sessionConfig);
      } catch (error) {
        const err = error as Error;
        throw new InferenceError(
          `Failed to initialize ONNX session: ${err.message}`,
          InferenceErrorCode.MODEL_LOAD_FAILED,
          err
        );
      } finally {
        this.loadingPromise = null;
      }
    })();

    await this.loadingPromise;
  }

  /**
   * Load ONNX model file as ArrayBuffer
   */
  private async loadModel(variant: ModelVariant): Promise<ArrayBuffer> {
    const modelPath = `${MODEL_BASE_PATH}/${MODEL_FILES[variant]}`;

    const response = await fetch(modelPath);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const buffer = await response.arrayBuffer();
    if (buffer.byteLength === 0) {
      throw new Error('Downloaded model file is empty');
    }

    return buffer;
  }

  /**
   * Ensure session is ready for inference
   */
  private ensureReady(): void {
    if (!this.session) {
      throw new InferenceError(NOT_INITIALIZED_ERROR, InferenceErrorCode.NOT_INITIALIZED);
    }
  }

  /**
   * Run inference on tokenized input
   *
   * @param input - Tokenized input with input_ids and attention_mask
   * @returns Promise with logits and shape
   *
   * @example
   * ```typescript
   * const tokenized = await tokenizer.tokenize("Hello world");
   * const output = await inference.runInference({
   *   input_ids: tokenized.input_ids,
   *   attention_mask: tokenized.attention_mask,
   * });
   * console.log(output.shape); // [1, 512, 3]
   * ```
   */
  public async runInference(input: InferenceInput): Promise<InferenceOutput> {
    this.ensureReady();

    // Validate input dimensions
    if (input.input_ids.length !== input.attention_mask.length) {
      throw new InferenceError(INVALID_INPUT_LENGTH, InferenceErrorCode.INVALID_INPUT);
    }

    if (input.input_ids.length > MAX_SEQUENCE_LENGTH) {
      throw new InferenceError(SEQUENCE_TOO_LONG, InferenceErrorCode.INVALID_INPUT);
    }

    try {
      const batchSize = 1;
      const seqLen = input.input_ids.length;

      // Create input tensors (ONNX Runtime Web requires int64 as BigInt64Array)
      const inputIdsTensor = new ort.Tensor(
        'int64',
        new BigInt64Array(input.input_ids.map((id) => BigInt(id))),
        [batchSize, seqLen]
      );

      const attentionMaskTensor = new ort.Tensor(
        'int64',
        new BigInt64Array(input.attention_mask.map((m) => BigInt(m))),
        [batchSize, seqLen]
      );

      // Build feeds object with correct input names
      const feeds: Record<string, ort.Tensor> = {
        input_ids: inputIdsTensor,
        attention_mask: attentionMaskTensor,
      };

      // Run inference
      const results = await this.session!.run(feeds);

      // Extract output
      const logitsTensor = results.logits;
      const logits = logitsTensor.data as Float32Array;
      const shape = logitsTensor.dims as [number, number, number];

      return { logits, shape };
    } catch (error) {
      const err = error as Error;
      throw new InferenceError(
        `Inference execution failed: ${err.message}`,
        InferenceErrorCode.INFERENCE_FAILED,
        err
      );
    }
  }

  /**
   * Pre-warm the inference engine by loading the model and running a dummy inference
   * This triggers JIT compilation and shader caching for better first-use performance
   *
   * @param options - Session initialization options
   *
   * @example
   * ```typescript
   * // Warmup during page load
   * useEffect(() => {
   *   inference.warmup({ modelVariant: 'int8' });
   * }, []);
   * ```
   */
  public async warmup(options?: SessionOptions): Promise<void> {
    // Initialize session if not ready
    await this.initializeSession(options || {});

    // Run dummy inference to JIT-compile kernels
    const dummyInput: InferenceInput = {
      input_ids: new Array(WARMUP_SEQUENCE_LENGTH).fill(101), // [CLS] token
      attention_mask: new Array(WARMUP_SEQUENCE_LENGTH).fill(1),
    };

    try {
      await this.runInference(dummyInput);
    } catch (error) {
      // Warmup failure is non-critical, log but don't throw
      console.warn('Warmup inference failed:', error);
    }
  }

  /**
   * Check if inference session is ready to use
   */
  public isReady(): boolean {
    return this.session !== null;
  }
}

// Singleton instance with SSR safety
let instance: OnnxInferenceSession | null = null;

/**
 * Get the singleton inference session instance
 *
 * @throws InferenceError if called in non-browser environment (SSR)
 */
function getInferenceSession(): OnnxInferenceSession {
  if (typeof window === 'undefined') {
    throw new InferenceError(SSR_ERROR, InferenceErrorCode.SSR_ERROR);
  }
  return (instance ??= new OnnxInferenceSession());
}

/**
 * ONNX Inference Engine
 *
 * Singleton instance for running keyword extraction inference in the browser.
 * Automatically handles model loading, backend selection, and session management.
 *
 * @example
 * ```typescript
 * // Warmup during initialization
 * await inference.warmup();
 *
 * // Run inference
 * const tokenized = await tokenizer.tokenize("Machine learning");
 * const output = await inference.runInference({
 *   input_ids: tokenized.input_ids,
 *   attention_mask: tokenized.attention_mask,
 * });
 * ```
 */
export const inference = new Proxy({} as OnnxInferenceSession, {
  get(_, prop: string) {
    const instance = getInferenceSession();
    const value = instance[prop as keyof OnnxInferenceSession];
    return isFunction(value) ? value.bind(instance) : value;
  },
});

// Export types
export type { InferenceInput, InferenceOutput, SessionOptions };
export { InferenceError, InferenceErrorCode };
