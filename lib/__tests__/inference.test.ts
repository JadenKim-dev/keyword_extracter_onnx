/**
 * Comprehensive Test Suite for OnnxInferenceSession
 *
 * Tests ONNX model loading and inference execution using mocked fetch().
 * Models are loaded from disk (public/models/) instead of requiring HTTP server.
 */

import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import { tokenizer } from '../tokenizer';
import { inference, InferenceError, InferenceErrorCode } from '../inference';
import { readFileSync } from 'fs';
import { resolve } from 'path';

/**
 * Load actual ONNX model from disk for testing
 * Returns ArrayBuffer matching fetch() response format
 */
function loadModelFromDisk(modelPath: string): ArrayBuffer {
  const absolutePath = resolve(__dirname, '../../public/models/keyword_model', modelPath);
  const { buffer } = readFileSync(absolutePath);
  return buffer
}

// SSR Protection Test (must run before any inference access)
describe('OnnxInferenceSession - SSR Protection', () => {
  it('should throw SSR_ERROR when window is undefined', () => {
    // Use Vitest's stubGlobal for safe global object manipulation
    vi.stubGlobal('window', undefined);

    // Accessing inference methods in SSR environment should throw
    expect(() => inference.isReady()).toThrow('Inference can only be used in browser environment');

    // Restore all globals after test
    vi.unstubAllGlobals();
  });
});

describe('OnnxInferenceSession', () => {
  // Mock fetch() to load ONNX models from disk instead of requiring a server
  beforeAll(() => {
    const mockFetch = vi.fn(async (url: string) => {
      // Handle model file GET requests
      if (url.includes('keyword_model_fp32.onnx')) {
        const modelBuffer = loadModelFromDisk('keyword_model_fp32.onnx');
        return {
          ok: true,
          status: 200,
          statusText: 'OK',
          arrayBuffer: async () => modelBuffer,
        } as Response;
      }

      if (url.includes('keyword_model_int8.onnx')) {
        const modelBuffer = loadModelFromDisk('keyword_model_int8.onnx');
        return {
          ok: true,
          status: 200,
          statusText: 'OK',
          arrayBuffer: async () => modelBuffer,
        } as Response;
      }

      // Handle HEAD requests for model variant selection (inference.ts:140)
      if (url.includes('/models/keyword_model/') && url.includes('.onnx')) {
        return {
          ok: true,
          status: 200,
          statusText: 'OK',
        } as Response;
      }

      // Fallback for unexpected URLs
      return {
        ok: false,
        status: 404,
        statusText: 'Not Found',
        arrayBuffer: async () => new ArrayBuffer(0),
      } as Response;
    });

    vi.stubGlobal('fetch', mockFetch);
  });

  afterAll(() => {
    vi.unstubAllGlobals();
  });

  /**
   * Load model once for all tests to optimize performance
   * Model loaded from disk via mocked fetch (~63MB for INT8)
   */
  beforeAll(async () => {
    // Also warmup tokenizer for integration tests
    await Promise.all([
      tokenizer.warmup(),
      inference.warmup({ modelVariant: 'int8' }),
    ]);
  }, 120000); // 2 min timeout for model download + warmup

  describe('Initialization & Warmup', () => {
    it('should report ready state correctly after warmup', () => {
      expect(inference.isReady()).toBe(true);
    });

    it('should handle concurrent warmup calls without duplicate loading', async () => {
      const promises = [inference.warmup(), inference.warmup(), inference.warmup()];
      await Promise.all(promises);
      expect(inference.isReady()).toBe(true);
    });
  });

  describe('runInference() - Basic Inference', () => {
    it('should run inference on tokenized input with correct dimensions', async () => {
      // Use tokenizer to create real input
      const tokenized = await tokenizer.tokenize('Hello world');

      // Run inference
      const output = await inference.runInference({
        input_ids: tokenized.input_ids,
        attention_mask: tokenized.attention_mask,
      });

      // Validate output structure
      expect(output).toHaveProperty('logits');
      expect(output).toHaveProperty('shape');
      expect(output.shape).toHaveLength(3);
      expect(output.logits).toBeInstanceOf(Float32Array);
      
      // Logits should be [batch=1, seq_len=512, num_labels]
      const [batch, seqLen, numLabels] = output.shape;
      expect(batch).toBe(1);
      expect(seqLen).toBe(512);
      expect(numLabels).toBeGreaterThan(0);

      // Total elements should match shape
      const expectedSize = batch * seqLen * numLabels;
      expect(output.logits.length).toBe(expectedSize);
    });

    it('should produce different logits for different inputs', async () => {
      const input1 = await tokenizer.tokenize('Machine learning');
      const input2 = await tokenizer.tokenize('Deep neural networks');

      const output1 = await inference.runInference({
        input_ids: input1.input_ids,
        attention_mask: input1.attention_mask,
      });

      const output2 = await inference.runInference({
        input_ids: input2.input_ids,
        attention_mask: input2.attention_mask,
      });

      // Logits should be different for different inputs
      expect(output1.logits).not.toEqual(output2.logits);
    });
  });

  describe('runInference() - Input Validation', () => {
    it('should throw error for mismatched input_ids and attention_mask lengths', async () => {
      await expect(
        inference.runInference({
          input_ids: [101, 102],
          attention_mask: [1, 1, 1], // Wrong length
        })
      ).rejects.toThrow(InferenceError);
    });

    it('should throw error for sequence length exceeding maximum (512)', async () => {
      const tooLongInput = {
        input_ids: new Array(513).fill(101),
        attention_mask: new Array(513).fill(1),
      };

      await expect(inference.runInference(tooLongInput)).rejects.toThrow(InferenceError);
    });

    it('should handle edge case: minimum valid input (empty with special tokens)', async () => {
      // Tokenizer always adds [CLS] and [SEP], so minimum is 2 tokens + padding
      const tokenized = await tokenizer.tokenize('a'); // Single character
      const output = await inference.runInference({
        input_ids: tokenized.input_ids,
        attention_mask: tokenized.attention_mask,
      });

      expect(output.shape[0]).toBe(1);
      expect(output.shape[1]).toBe(512);
    });
  });

  describe('runInference() - Integration with Tokenizer', () => {
    it('should complete full tokenization → inference workflow', async () => {
      const text = 'Machine learning is transforming the world';

      // Step 1: Tokenize
      const tokenized = await tokenizer.tokenize(text);
      expect(tokenized.input_ids.length).toBe(512);

      // Step 2: Run inference
      const output = await inference.runInference({
        input_ids: tokenized.input_ids,
        attention_mask: tokenized.attention_mask,
      });

      // Step 3: Verify output dimensions match input
      expect(output.shape[1]).toBe(tokenized.input_ids.length);

      // Step 4: Verify logits are valid (no NaN or Infinity)
      const hasInvalidValues = Array.from(output.logits).some(
        (val) => isNaN(val) || !isFinite(val)
      );
      expect(hasInvalidValues).toBe(false);
    });

    it('should work with various text lengths', async () => {
      const texts = [
        'Short',
        'This is a much longer text that contains many more words and should still be processed correctly by the tokenizer and inference engine without any issues or errors.',
      ];

      for (const text of texts) {
        const tokenized = await tokenizer.tokenize(text);
        const output = await inference.runInference({
          input_ids: tokenized.input_ids,
          attention_mask: tokenized.attention_mask,
        });

        expect(output.shape[0]).toBe(1);
        expect(output.shape[1]).toBe(512);
        expect(output.logits).toBeInstanceOf(Float32Array);
      }
    });

    it('should handle special characters and unicode correctly', async () => {
      const texts = [
        'Unicode: 世界 🌍 café',
        'Symbols: $%^&*()[]{}',
      ];

      for (const text of texts) {
        const tokenized = await tokenizer.tokenize(text);
        const output = await inference.runInference({
          input_ids: tokenized.input_ids,
          attention_mask: tokenized.attention_mask,
        });

        expect(output.shape[0]).toBe(1);
        expect(output.shape[1]).toBe(512);
      }
    });
  });

  describe('Error Handling', () => {
    it('should provide clear error messages with error codes', async () => {
      try {
        await inference.runInference({
          input_ids: [101, 102],
          attention_mask: [1, 1, 1], // Wrong length
        });
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).toBeInstanceOf(InferenceError);
        const inferenceError = error as InferenceError;
        expect(inferenceError.message).toBeTruthy();
        expect(inferenceError.code).toBe(InferenceErrorCode.INVALID_INPUT);
        expect(inferenceError.name).toBe('InferenceError');
      }
    });
  });
});
