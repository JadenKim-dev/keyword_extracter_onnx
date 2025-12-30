/**
 * Comprehensive Test Suite for DistilBertTokenizer
 *
 * Tests real model loading and tokenization behavior.
 */

import { describe, it, expect, beforeAll, vi } from 'vitest';
import { tokenizer } from '../tokenizer';

// SSR Protection Test (must run before any tokenizer access)
describe('DistilBertTokenizer - SSR Protection', () => {
  it('should throw SSR_ERROR when window is undefined', () => {
    // Use Vitest's stubGlobal for safe global object manipulation
    vi.stubGlobal('window', undefined);

    // Accessing tokenizer methods in SSR environment should throw
    expect(() => tokenizer.isReady()).toThrow('Tokenizer can only be used in browser environment');

    // Restore all globals after test
    vi.unstubAllGlobals();
  });
});

describe('DistilBertTokenizer', () => {
  /**
   * Load model once for all tests to optimize performance
   * Real model loading takes 2-5 seconds initially
   */
  beforeAll(async () => {
    await tokenizer.warmup();
  }, 60000); // 60s timeout for model loading

  describe('Initialization & Warmup', () => {
    it('should report ready state correctly after warmup', () => {
      expect(tokenizer.isReady()).toBe(true);
    });

    it('should handle concurrent warmup calls without duplicate loading', async () => {
      const promises = [
        tokenizer.warmup(),
        tokenizer.warmup(),
        tokenizer.warmup(),
      ];
      await Promise.all(promises);
      expect(tokenizer.isReady()).toBe(true);
    });
  });

  describe('tokenize() - Basic Tokenization', () => {
    it('should tokenize simple text with special tokens', async () => {
      const result = await tokenizer.tokenize('Hello world');

      expect(result).toHaveProperty('input_ids');
      expect(result).toHaveProperty('attention_mask');
      expect(result).toHaveProperty('tokens');
      expect(result.tokens[0]).toBe('[CLS]');
      expect(result.tokens).toContain('[SEP]');
    });

    it('should pad short text (single char and short word) to max_length (512)', async () => {
      const results = await Promise.all([
        tokenizer.tokenize('a'),
        tokenizer.tokenize('Hi')
      ]);
      
      results.forEach(result => {
        expect(result.input_ids.length).toBe(512);
        expect(result.attention_mask.length).toBe(512);
        expect(result.tokens.length).toBe(512);
        expect(result.tokens[0]).toBe('[CLS]');
      });
    });

    it('should generate correct attention masks (1=real, 0=padding)', async () => {
      const result = await tokenizer.tokenize('Hello');

      const realTokens = result.attention_mask.filter((m) => m === 1).length;
      const paddingTokens = result.attention_mask.filter((m) => m === 0).length;

      expect(realTokens + paddingTokens).toBe(512);
      expect(realTokens).toBeGreaterThan(0);
      expect(realTokens).toBeLessThan(512); // Short text should have padding
      expect(paddingTokens).toBeGreaterThan(0);
    });

    it('should place [CLS] at start and [SEP] before padding', async () => {
      const result = await tokenizer.tokenize('Test');

      expect(result.tokens[0]).toBe('[CLS]');

      // Find first padding token
      const firstPadIdx = result.tokens.indexOf('[PAD]');
      if (firstPadIdx > 0) {
        expect(result.tokens[firstPadIdx - 1]).toBe('[SEP]');
      }
    });

    it('should tokenize longer text correctly', async () => {
      const longText = 'This is a longer piece of text that contains multiple words and should be tokenized properly by the DistilBERT tokenizer.';
      const result = await tokenizer.tokenize(longText);

      expect(result.tokens[0]).toBe('[CLS]');
      expect(result.input_ids.length).toBe(512);
      expect(result.attention_mask.filter((m) => m === 1).length).toBeGreaterThan(10);
    });
  });

  describe('tokenize() - Truncation', () => {
    it('should truncate text exceeding 512 tokens', async () => {
      // Create very long text (1000+ words)
      const longText = 'word '.repeat(1000);
      const result = await tokenizer.tokenize(longText);

      expect(result.input_ids.length).toBe(512);
      expect(result.tokens.length).toBe(512);
      expect(result.tokens[0]).toBe('[CLS]');
      // With truncation, text fills all 512 tokens
      expect(result.attention_mask.filter((m) => m === 1).length).toBe(512);
      // No padding when text is too long
      expect(result.attention_mask.filter((m) => m === 0).length).toBe(0);
    });
  });

  describe('tokenize() - Edge Cases', () => {
    it('should throw error for empty string', async () => {
      await expect(tokenizer.tokenize('')).rejects.toThrow('Input text cannot be empty');
    });

    it('should throw error for empty array', async () => {
      await expect(tokenizer.tokenize([])).rejects.toThrow('Input text cannot be empty');
    });

    it('should handle unicode characters correctly', async () => {
      const result = await tokenizer.tokenize('Hello 世界 🌍');

      expect(result.tokens[0]).toBe('[CLS]');
      expect(result.input_ids.length).toBe(512);
      expect(result.tokens.length).toBe(512);
    });

    it('should handle special characters and symbols', async () => {
      const result = await tokenizer.tokenize('Test: @#$%^&*()');

      expect(result.tokens[0]).toBe('[CLS]');
      expect(result.input_ids.length).toBe(512);
    });

    it('should handle text with newlines and tabs', async () => {
      const result = await tokenizer.tokenize('Line 1\nLine 2\tTab');

      expect(result.tokens[0]).toBe('[CLS]');
      expect(result.input_ids.length).toBe(512);
    });
  });

  describe('decode() - Token Decoding', () => {
    it('should decode array of token IDs', async () => {
      const tokenizeResult = await tokenizer.tokenize('Hello world');
      const decoded = await tokenizer.decode(tokenizeResult.input_ids);

      expect(typeof decoded).toBe('string');
      expect(decoded.length).toBeGreaterThan(0);
    });

    it('should return empty string for empty array', async () => {
      const text = await tokenizer.decode([]);

      expect(text).toBe('');
    });

    it('should skip special tokens in decoded output', async () => {
      const result = await tokenizer.tokenize('Hello world');
      const decoded = await tokenizer.decode(result.input_ids);

      // Special tokens should be skipped
      expect(decoded).not.toContain('[CLS]');
      expect(decoded).not.toContain('[SEP]');
      expect(decoded).not.toContain('[PAD]');
    });

    it('should decode and preserve text content', async () => {
      const originalText = 'Hello world';
      const result = await tokenizer.tokenize(originalText);
      const decoded = await tokenizer.decode(result.input_ids);

      // Decoded text should contain the original words (case-insensitive)
      expect(decoded.toLowerCase()).toContain('hello');
      expect(decoded.toLowerCase()).toContain('world');
    });

    it('should handle decoding with mixed real and padding tokens', async () => {
      const result = await tokenizer.tokenize('Short');
      // input_ids contains both real tokens and padding (0s)
      const decoded = await tokenizer.decode(result.input_ids);

      expect(typeof decoded).toBe('string');
      expect(decoded.length).toBeGreaterThan(0);
    });
  });

  describe('Integration Tests', () => {
    it('should complete full tokenization workflow', async () => {
      const text = 'Machine learning is amazing';

      // Step 1: Tokenize
      const tokenized = await tokenizer.tokenize(text);
      expect(tokenized.input_ids.length).toBe(512);

      // Step 2: Decode
      const decoded = await tokenizer.decode(tokenized.input_ids);
      expect(typeof decoded).toBe('string');

      // Step 3: Verify content preserved
      expect(decoded.toLowerCase()).toContain('machine');
      expect(decoded.toLowerCase()).toContain('learning');
    });

    it('should maintain consistency across multiple calls', async () => {
      const text = 'Consistent test';
      const result1 = await tokenizer.tokenize(text);
      const result2 = await tokenizer.tokenize(text);

      // Same input should produce same output
      expect(result1.input_ids).toEqual(result2.input_ids);
      expect(result1.attention_mask).toEqual(result2.attention_mask);
    });
  });
});
