import { describe, it, expect } from 'vitest';
import { extractKeywords } from '../postprocess';
import type { InferenceOutput } from '../types/inference';
import type { TokenizerOutput } from '../types/tokenizer';

// ============================================================================
// Unit Tests - Logits to Predictions Conversion
// ============================================================================

describe('BIO Token Predictions', () => {
  it('should convert logits to predictions correctly', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        // Token 0: Strong O prediction
        3.0, 0.1, 0.1,
        // Token 1: Strong B-KEY prediction
        0.1, 3.0, 0.1,
        // Token 2: Strong I-KEY prediction
        0.1, 0.1, 3.0,
      ]),
      shape: [1, 3, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103],
      attention_mask: [1, 1, 1],
      tokens: ['[CLS]', 'machine', 'learning'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    // Since [CLS] is filtered, we should get one keyword "machine learning"
    // (B-KEY + I-KEY)
    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('machine learning');
    expect(result.keywords[0].confidence).toBeGreaterThan(0.9);  // High confidence
  });
});

// ============================================================================
// Unit Tests - WordPiece Reconstruction
// ============================================================================

describe('WordPiece Reconstruction', () => {
  it('should handle simple words without subwords', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.0, 0.1,  // B-KEY
        0.1, 0.1, 3.0,  // I-KEY
      ]),
      shape: [1, 2, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102],
      attention_mask: [1, 1],
      tokens: ['machine', 'learning'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords[0].text).toBe('machine learning');
  });

  it('should merge WordPiece subwords correctly', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.0, 0.1,  // B-KEY
        0.1, 0.1, 3.0,  // I-KEY
      ]),
      shape: [1, 2, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102],
      attention_mask: [1, 1],
      tokens: ['play', '##ing'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords[0].text).toBe('playing');
  });

  it('should handle mixed words and subwords', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.0, 0.1,  // B-KEY: "natural"
        0.1, 0.1, 3.0,  // I-KEY: "language"
        0.1, 0.1, 3.0,  // I-KEY: "process"
        0.1, 0.1, 3.0,  // I-KEY: "##ing"
      ]),
      shape: [1, 4, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103, 104],
      attention_mask: [1, 1, 1, 1],
      tokens: ['natural', 'language', 'process', '##ing'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords[0].text).toBe('natural language processing');
  });

  it('should handle single token', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([0.1, 3.0, 0.1]),
      shape: [1, 1, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101],
      attention_mask: [1],
      tokens: ['hello'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords[0].text).toBe('hello');
  });

  it('should handle token starting with ## as first token', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([0.1, 3.0, 0.1]),
      shape: [1, 1, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101],
      attention_mask: [1],
      tokens: ['##ing'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords[0].text).toBe('ing');
  });
});

// ============================================================================
// Unit Tests - BIO Span Extraction
// ============================================================================

describe('BIO Span Extraction', () => {
  it('should extract single keyword span (B-KEY only)', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        3.0, 0.1, 0.1,  // O
        0.1, 3.0, 0.1,  // B-KEY
        3.0, 0.1, 0.1,  // O
      ]),
      shape: [1, 3, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103],
      attention_mask: [1, 1, 1],
      tokens: ['the', 'machine', 'is'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('machine');
    expect(result.keywords[0].tokenCount).toBe(1);
  });

  it('should extract multi-token keyword span (B-KEY + I-KEY)', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.0, 0.1,  // B-KEY
        0.1, 0.1, 3.0,  // I-KEY
        3.0, 0.1, 0.1,  // O
      ]),
      shape: [1, 3, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103],
      attention_mask: [1, 1, 1],
      tokens: ['machine', 'learning', 'is'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('machine learning');
    expect(result.keywords[0].tokenCount).toBe(2);
  });

  it('should extract multiple separate keyword spans', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.0, 0.1,  // B-KEY: machine
        3.0, 0.1, 0.1,  // O: and
        0.1, 3.0, 0.1,  // B-KEY: deep
        0.1, 0.1, 3.0,  // I-KEY: learning
      ]),
      shape: [1, 4, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103, 104],
      attention_mask: [1, 1, 1, 1],
      tokens: ['machine', 'and', 'deep', 'learning'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords).toHaveLength(2);
    expect(result.keywords.map(k => k.text).sort()).toEqual(['deep learning', 'machine']);
  });

  it('should ignore I-KEY without preceding B-KEY (orphaned I-KEY)', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        3.0, 0.1, 0.1,  // O
        0.1, 0.1, 3.0,  // I-KEY (orphaned)
        3.0, 0.1, 0.1,  // O
      ]),
      shape: [1, 3, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103],
      attention_mask: [1, 1, 1],
      tokens: ['the', 'machine', 'is'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords).toHaveLength(0);  // Orphaned I-KEY ignored
  });

  it('should handle consecutive keyword spans (B-KEY after I-KEY)', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.0, 0.1,  // B-KEY: machine
        0.1, 0.1, 3.0,  // I-KEY: learning
        0.1, 3.0, 0.1,  // B-KEY: deep (new span)
        0.1, 0.1, 3.0,  // I-KEY: neural
      ]),
      shape: [1, 4, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103, 104],
      attention_mask: [1, 1, 1, 1],
      tokens: ['machine', 'learning', 'deep', 'neural'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords).toHaveLength(2);
    expect(result.keywords.map(k => k.text).sort()).toEqual(['deep neural', 'machine learning']);
  });
});

// ============================================================================
// Unit Tests - Filtering
// ============================================================================

describe('Filtering', () => {
  it('should filter keywords by minimum length', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.0, 0.1,  // B-KEY: "machine"
        0.1, 0.1, 3.0,  // I-KEY: "learning"
        3.0, 0.1, 0.1,  // O
        0.1, 3.0, 0.1,  // B-KEY: "ai"
      ]),
      shape: [1, 4, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103, 104],
      attention_mask: [1, 1, 1, 1],
      tokens: ['machine', 'learning', 'and', 'ai'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 3,  // Filter out "ai" (2 chars)
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('machine learning');
  });

  it('should filter stopwords', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.0, 0.1,  // B-KEY: "machine"
        0.1, 0.1, 3.0,  // I-KEY: "learning"
        3.0, 0.1, 0.1,  // O
        0.1, 3.0, 0.1,  // B-KEY: "the"
      ]),
      shape: [1, 4, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103, 104],
      attention_mask: [1, 1, 1, 1],
      tokens: ['machine', 'learning', 'and', 'the'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: true,  // Filter out "the"
    });

    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('machine learning');
  });

  it('should filter by confidence threshold', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        // High confidence keyword
        0.1, 5.0, 0.1,  // B-KEY
        0.1, 0.1, 5.0,  // I-KEY
        3.0, 0.1, 0.1,  // O
        // Low confidence keyword
        0.5, 0.6, 0.4,  // B-KEY (lower confidence)
      ]),
      shape: [1, 4, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103, 104],
      attention_mask: [1, 1, 1, 1],
      tokens: ['machine', 'learning', 'and', 'test'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0.5,  // Filter low confidence
      removeStopwords: false,
    });

    // Should keep high confidence keywords
    expect(result.keywords.length).toBeGreaterThan(0);
    expect(result.keywords.every(k => k.confidence >= 0.5)).toBe(true);
  });

  it('should deduplicate case-insensitive keywords and keep highest confidence', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.5, 0.1,  // B-KEY: "Machine" (high confidence)
        0.1, 0.1, 3.5,  // I-KEY: "Learning"
        3.0, 0.1, 0.1,  // O
        0.1, 3.0, 0.1,  // B-KEY: "machine" (lower confidence)
        0.1, 0.1, 3.0,  // I-KEY: "learning"
      ]),
      shape: [1, 5, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103, 104, 105],
      attention_mask: [1, 1, 1, 1, 1],
      tokens: ['Machine', 'Learning', 'and', 'machine', 'learning'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    // Should keep only one "machine learning" (the higher confidence one)
    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('Machine Learning');  // Keeps original casing
    expect(result.keywords[0].confidence).toBeGreaterThan(0.9);  // High confidence preserved
  });

  it('should deduplicate single-word keywords case-insensitively', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.5, 0.1,  // B-KEY: "Machine" (higher confidence)
        3.0, 0.1, 0.1,  // O
        0.1, 3.0, 0.1,  // B-KEY: "machine" (lower confidence)
      ]),
      shape: [1, 3, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103],
      attention_mask: [1, 1, 1],
      tokens: ['Machine', 'and', 'machine'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    // Should keep only one (case-insensitive deduplication)
    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('Machine');  // Keeps higher confidence version
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe('extractKeywords() - Integration', () => {
  it('should extract keyword with default options', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        // [CLS]
        3.0, 0.1, 0.1,
        // "the" - O
        3.0, 0.1, 0.1,
        // "machine" - B-KEY
        0.1, 3.5, 0.1,
        // "learning" - I-KEY
        0.1, 0.1, 3.5,
        // "is" - O
        3.0, 0.1, 0.1,
        // [SEP]
        3.0, 0.1, 0.1,
      ]),
      shape: [1, 6, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 1996, 3698, 4083, 2003, 102],
      attention_mask: [1, 1, 1, 1, 1, 1],
      tokens: ['[CLS]', 'the', 'machine', 'learning', 'is', '[SEP]'],
    };

    const result = extractKeywords(output, tokenizerOutput);

    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('machine learning');
    expect(result.keywords[0].tokenCount).toBe(2);
    expect(result.keywords[0].confidence).toBeGreaterThan(0.5);
  });

  it('should provide comprehensive metadata', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        3.0, 0.1, 0.1,
        0.1, 3.0, 0.1,
        3.0, 0.1, 0.1,
      ]),
      shape: [1, 3, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 0],
      attention_mask: [1, 1, 0],
      tokens: ['[CLS]', 'test', '[PAD]'],
    };

    const result = extractKeywords(output, tokenizerOutput);

    expect(result.metadata.modelOutputShape).toEqual([1, 3, 3]);
    expect(result.metadata.attentionTokens).toBe(2);  // Non-padding tokens
    expect(result.totalTokens).toBe(3);
    expect(result.metadata.options.minLength).toBe(3);
  });

  it('should apply all filters in correct order', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.0, 0.1,  // B-KEY: "machine"
        0.1, 0.1, 3.0,  // I-KEY: "learning"
        3.0, 0.1, 0.1,  // O
        0.1, 3.0, 0.1,  // B-KEY: "ai"
        3.0, 0.1, 0.1,  // O
        0.1, 3.0, 0.1,  // B-KEY: "the"
      ]),
      shape: [1, 6, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103, 104, 105, 106],
      attention_mask: [1, 1, 1, 1, 1, 1],
      tokens: ['machine', 'learning', 'and', 'ai', 'and', 'the'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 3,         // Filters out "ai" (2 chars)
      minConfidence: 0.5,
      removeStopwords: true,  // Filters out "the"
    });

    // Should only have "machine learning"
    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('machine learning');
    expect(result.rawKeywordCount).toBeGreaterThan(result.filteredKeywordCount);
  });

});

// ============================================================================
// Edge Cases
// ============================================================================

describe('extractKeywords() - Edge Cases', () => {
  it('should handle empty predictions (all O labels)', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        3.0, 0.1, 0.1,  // All O
        3.0, 0.1, 0.1,
        3.0, 0.1, 0.1,
      ]),
      shape: [1, 3, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103],
      attention_mask: [1, 1, 1],
      tokens: ['[CLS]', 'test', '[SEP]'],
    };

    const result = extractKeywords(output, tokenizerOutput);

    expect(result.keywords).toHaveLength(0);
    expect(result.rawKeywordCount).toBe(0);
  });

  it('should handle all special tokens', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.0, 0.1,  // B-KEY (but will be filtered)
        0.1, 3.0, 0.1,  // B-KEY (but will be filtered)
      ]),
      shape: [1, 2, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102],
      attention_mask: [1, 1],
      tokens: ['[CLS]', '[SEP]'],
    };

    const result = extractKeywords(output, tokenizerOutput);

    expect(result.keywords).toHaveLength(0);
  });

  it('should handle single-token keywords', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        3.0, 0.1, 0.1,  // [CLS] - O
        0.1, 3.0, 0.1,  // "ai" - B-KEY
        3.0, 0.1, 0.1,  // [SEP] - O
      ]),
      shape: [1, 3, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 9932, 102],
      attention_mask: [1, 1, 1],
      tokens: ['[CLS]', 'ai', '[SEP]'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 2,  // ai is 2 chars, should pass
      minConfidence: 0,
      removeStopwords: false,
    });

    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('ai');
    expect(result.keywords[0].tokenCount).toBe(1);
  });

  it('should skip padding tokens', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        0.1, 3.0, 0.1,  // B-KEY: "machine"
        0.1, 3.0, 0.1,  // B-KEY: "[PAD]" (but attention_mask=0)
        0.1, 3.0, 0.1,  // B-KEY: "[PAD]" (but attention_mask=0)
      ]),
      shape: [1, 3, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 0, 0],
      attention_mask: [1, 0, 0],  // Last two are padding
      tokens: ['machine', '[PAD]', '[PAD]'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    // Should only extract "machine", padding tokens should be ignored
    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('machine');
  });

  it('should handle keyword at sequence end', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        3.0, 0.1, 0.1,  // O: "the"
        0.1, 3.0, 0.1,  // B-KEY: "machine"
        0.1, 0.1, 3.0,  // I-KEY: "learning"
      ]),
      shape: [1, 3, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [101, 102, 103],
      attention_mask: [1, 1, 1],
      tokens: ['the', 'machine', 'learning'],
    };

    const result = extractKeywords(output, tokenizerOutput, {
      minLength: 0,
      minConfidence: 0,
      removeStopwords: false,
    });

    // Should properly close the span at sequence end
    expect(result.keywords).toHaveLength(1);
    expect(result.keywords[0].text).toBe('machine learning');
    expect(result.keywords[0].endTokenIndex).toBe(3);
  });

  it('should handle empty input with only padding', () => {
    const output: InferenceOutput = {
      logits: new Float32Array([
        3.0, 0.1, 0.1,
        3.0, 0.1, 0.1,
      ]),
      shape: [1, 2, 3],
    };

    const tokenizerOutput: TokenizerOutput = {
      input_ids: [0, 0],
      attention_mask: [0, 0],  // All padding
      tokens: ['[PAD]', '[PAD]'],
    };

    const result = extractKeywords(output, tokenizerOutput);

    expect(result.keywords).toHaveLength(0);
    expect(result.metadata.attentionTokens).toBe(0);
  });
});
