/**
 * Post-processing pipeline for keyword extraction from ONNX model outputs
 *
 * Converts raw BIO-tagged logits into structured keyword results with
 * proper text reconstruction and filtering.
 */

import type { InferenceOutput } from './types/inference';
import type { TokenizerOutput } from './types/tokenizer';
import type {
  BIOLabel,
  TokenPrediction,
  KeywordSpan,
  Keyword,
  PostProcessingOptions,
  KeywordExtractionResult,
} from './types/keywords';
import { BIOLabel as BIOLabelEnum } from './types/keywords';
import { ENGLISH_STOPWORDS, isStopword } from './constants/stopwords';

// Special tokens that should be filtered before processing
const SPECIAL_TOKENS = new Set(['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]']);

/**
 * Apply softmax function to an array of logits
 *
 * Uses numerical stability technique by subtracting the maximum value
 * before computing exponentials to prevent overflow.
 *
 * @param logits - Array of logit values
 * @returns New array of probabilities that sum to 1.0
 *
 * @example
 * ```typescript
 * const probs = softmax([2.0, 1.0, 0.1]);
 * // Result: [0.659, 0.242, 0.099]
 * ```
 */
function softmax(logits: number[]): number[] {
  const result = new Array(logits.length);
  
  // Find max for numerical stability
  let maxLogit = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
    }
  }
  
  // Compute exp(logit - max) and sum
  let sumExps = 0;
  for (let i = 0; i < logits.length; i++) {
    const exp = Math.exp(logits[i] - maxLogit);
    result[i] = exp;
    sumExps += exp;
  }
  
  // Normalize by sum
  for (let i = 0; i < logits.length; i++) {
    result[i] /= sumExps;
  }
  
  return result;
}

/**
 * Find the index of the maximum value in an array
 *
 * @param values - Array of numeric values
 * @returns Index of the maximum value
 *
 * @example
 * ```typescript
 * const idx = argmax([0.1, 0.7, 0.2]);  // Returns 1
 * ```
 */
function argmax(values: number[]): number {
  return values.reduce((maxIdx, val, idx, arr) => 
    val > arr[maxIdx] ? idx : maxIdx
  , 0);
}

/**
 * Convert raw logits to token predictions with BIO labels
 *
 * Applies softmax to each token's logits and selects the highest 
 * probability label (argmax).
 *
 * @param logits - Flattened Float32Array of shape [batch, seq_len, num_classes]
 * @param shape - Shape tuple [batch_size, sequence_length, num_classes]
 * @param tokens - Array of token strings
 * @returns Array of TokenPrediction objects
 *
 * @example
 * ```typescript
 * // Softmax example: [2.0, 1.0, 0.1] -> [0.659, 0.242, 0.099]
 * const predictions = convertLogitsToPredictions(
 *   new Float32Array([2.0, 1.0, 0.1]),
 *   [1, 1, 3],
 *   ['token']
 * );
 * ```
 */
function convertLogitsToPredictions(
  logits: Float32Array,
  shape: [number, number, number],
  tokens: string[]
): TokenPrediction[] {
  const [, seqLen, numClasses] = shape;
  const predictions: TokenPrediction[] = new Array(seqLen);

  for (let i = 0; i < seqLen; i++) {
    const tokenLogits: number[] = new Array(numClasses);
    // Calculate starting position of i-th token's logits in flattened 1D array (row-major order)
    // e.g., i=0 → offset=0 (indices 0,1,2), i=1 → offset=3 (indices 3,4,5)
    const offset = i * numClasses;

    // Extract logits for current token
    for (let j = 0; j < numClasses; j++) {
      tokenLogits[j] = logits[offset + j];
    }

    // Apply softmax to get probabilities
    const probabilities = softmax(tokenLogits);

    // Find predicted label (argmax)
    const predictedLabel = argmax(probabilities);

    predictions[i] = {
      tokenIndex: i,
      label: predictedLabel as BIOLabel,
      confidence: probabilities[predictedLabel],
      probabilities: [probabilities[0], probabilities[1], probabilities[2]] as [number, number, number],
      token: tokens[i],
    };
  }

  return predictions;
}

/**
 * Filter out special tokens from predictions
 *
 * Removes [CLS], [SEP], [PAD], [UNK], [MASK] and respects attention mask
 *
 * @param predictions - Array of token predictions
 * @param attentionMask - Attention mask (1 for real tokens, 0 for padding)
 * @returns Filtered predictions array
 */
function filterSpecialTokens(
  predictions: TokenPrediction[],
  attentionMask: number[]
): TokenPrediction[] {
  return predictions.filter((pred, idx) =>
    attentionMask[idx] === 1 && !SPECIAL_TOKENS.has(pred.token)
  );
}

/**
 * Create a KeywordSpan from current span data and confidences
 *
 * Helper function to avoid duplicating span creation logic.
 * Calculates average confidence from confidence array.
 *
 * @param currentSpan - Partial span data without confidence
 * @param confidences - Array of confidence scores for tokens in span
 * @returns Complete KeywordSpan object
 */
function createKeywordSpan(
  currentSpan: Omit<KeywordSpan, 'confidence'>,
  confidences: number[]
): KeywordSpan {
  return {
    ...currentSpan,
    confidence: confidences.reduce((a, b) => a + b, 0) / confidences.length,
  };
}

/**
 * Extract keyword spans from BIO-tagged predictions
 *
 * Groups consecutive B-KEY + I-KEY* sequences into keyword spans.
 * Orphaned I-KEY tokens (without preceding B-KEY) are ignored.
 *
 * @param predictions - Array of token predictions
 * @param attentionMask - Attention mask to skip padding
 * @returns Array of keyword spans
 *
 * @example
 * ```typescript
 * // Input: [O, B-KEY, I-KEY, O, B-KEY]
 * // Output: [
 * //   { rawTokens: ['machine', 'learning'], ... },
 * //   { rawTokens: ['ai'], ... }
 * // ]
 * ```
 */
function extractKeywordSpans(
  predictions: TokenPrediction[],
  attentionMask: number[]
): KeywordSpan[] {
  const spans: KeywordSpan[] = [];
  let currentSpan: Omit<KeywordSpan, 'confidence'> | null = null;
  const confidences: number[] = [];

  for (const [i, pred] of predictions.entries()) {
    if (attentionMask[i] === 0) continue;  // Skip padding

    if (pred.label === BIOLabelEnum.B_KEY) {
      // Start new span - save previous if exists
      if (currentSpan) {
        spans.push(createKeywordSpan(currentSpan, confidences));
      }

      // Initialize new span
      currentSpan = {
        startTokenIndex: i,
        endTokenIndex: i + 1,
        tokenIndices: [i],
        rawTokens: [pred.token],
      };
      confidences.length = 0;
      confidences.push(pred.confidence);
    }
    else if (pred.label === BIOLabelEnum.I_KEY && currentSpan) {
      // Continue current span
      currentSpan.tokenIndices.push(i);
      currentSpan.rawTokens.push(pred.token);
      currentSpan.endTokenIndex = i + 1;
      confidences.push(pred.confidence);
    }
    else if (pred.label === BIOLabelEnum.O) {
      // End current span
      if (currentSpan) {
        spans.push(createKeywordSpan(currentSpan, confidences));
        currentSpan = null;
        confidences.length = 0;
      }
    }
    // Note: I-KEY without preceding B-KEY is ignored (orphaned I-KEY)
  }

  // Handle last span if it exists
  if (currentSpan) {
    spans.push(createKeywordSpan(currentSpan, confidences));
  }

  return spans;
}

/**
 * Reconstruct text from WordPiece tokens
 *
 * Handles subword tokens with ## prefix by merging them correctly.
 *
 * Rules:
 * - Tokens starting with "##" are subword continuations
 * - Remove "##" prefix and concatenate directly to previous token
 * - Tokens without "##" are word boundaries - add space before them
 *
 * @param tokens - Array of WordPiece tokens
 * @returns Reconstructed text string
 *
 * @example
 * ```typescript
 * reconstructText(['machine', 'learning']);  // "machine learning"
 * reconstructText(['play', '##ing']);  // "playing"
 * reconstructText(['natural', 'language', 'process', '##ing']);  // "natural language processing"
 * ```
 */
function reconstructText(tokens: string[]): string {
  if (tokens.length === 0) return '';

  // Handle first token (may or may not start with ##)
  let result = tokens[0].startsWith('##') ? tokens[0].substring(2) : tokens[0];

  for (let i = 1; i < tokens.length; i++) {
    const token = tokens[i];

    if (token.startsWith('##')) {
      // Subword continuation - remove ## and append directly
      result += token.substring(2);
    } else {
      // New word - add space before
      result += ' ' + token;
    }
  }

  return result.trim();
}

/**
 * Filter keywords by minimum text length
 *
 * @param keywords - Array of keywords
 * @param minLength - Minimum character length
 * @returns Filtered keywords array
 */
function filterByLength(keywords: Keyword[], minLength: number): Keyword[] {
  return keywords.filter(kw => kw.text.length >= minLength);
}

/**
 * Filter out stopwords from keywords
 *
 * @param keywords - Array of keywords
 * @param customStopwords - Optional custom stopwords set
 * @returns Filtered keywords array
 */
function filterStopwords(keywords: Keyword[], customStopwords?: Set<string>): Keyword[] {
  return keywords.filter(kw => !isStopword(kw.text, customStopwords));
}

/**
 * Filter keywords by minimum confidence threshold
 *
 * @param keywords - Array of keywords
 * @param minConfidence - Minimum confidence score [0-1]
 * @returns Filtered keywords array
 */
function filterByConfidence(keywords: Keyword[], minConfidence: number): Keyword[] {
  return keywords.filter(kw => kw.confidence >= minConfidence);
}

/**
 * Remove duplicate keywords, keeping highest confidence instance
 *
 * @param keywords - Array of keywords
 * @returns Deduplicated keywords array
 */
function deduplicate(keywords: Keyword[]): Keyword[] {
  const seen = new Map<string, Keyword>();

  for (const kw of keywords) {
    const key = kw.text.toLowerCase();

    const existing = seen.get(key);
    if (!existing || kw.confidence > existing.confidence) {
      seen.set(key, kw);  // Keep highest confidence
    }
  }

  return Array.from(seen.values());
}

/**
 * Extract keywords from ONNX model output
 *
 * Main function that orchestrates the entire post-processing pipeline:
 * 1. Convert logits to BIO predictions
 * 2. Filter special tokens
 * 3. Extract keyword spans using BIO tagging
 * 4. Reconstruct text from WordPiece tokens
 * 5. Apply post-processing filters
 * 6. Sort by confidence
 *
 * @param output - Raw ONNX inference output with logits
 * @param tokenizerOutput - Tokenizer output with tokens and attention mask
 * @param options - Optional post-processing configuration
 * @returns Structured keyword extraction result
 *
 * @example
 * ```typescript
 * const tokenized = await tokenizer.tokenize("Machine learning is transforming AI");
 * const inference = await inference.runInference({
 *   input_ids: tokenized.input_ids,
 *   attention_mask: tokenized.attention_mask,
 * });
 *
 * const result = extractKeywords(inference, tokenized, {
 *   minLength: 3,
 *   minConfidence: 0.5,
 *   removeStopwords: true,
 * });
 *
 * console.log(result.keywords);
 * // [
 * //   { text: "Machine learning", confidence: 0.89, ... },
 * //   { text: "AI", confidence: 0.76, ... }
 * // ]
 * ```
 */
export function extractKeywords(
  output: InferenceOutput,
  tokenizerOutput: TokenizerOutput,
  options?: PostProcessingOptions
): KeywordExtractionResult {
  // Step 1: Merge options with defaults
  const opts: Required<PostProcessingOptions> = {
    minLength: options?.minLength ?? 3,
    minConfidence: options?.minConfidence ?? 0.5,
    removeStopwords: options?.removeStopwords ?? true,
    customStopwords: options?.customStopwords ?? ENGLISH_STOPWORDS,
  };

  // Step 2: Convert logits to predictions
  const predictions = convertLogitsToPredictions(
    output.logits,
    output.shape,
    tokenizerOutput.tokens
  );

  // Step 3: Filter special tokens
  const filteredPredictions = filterSpecialTokens(
    predictions,
    tokenizerOutput.attention_mask
  );

  // Step 4: Extract keyword spans using BIO tagging
  const rawSpans = extractKeywordSpans(
    filteredPredictions,
    tokenizerOutput.attention_mask
  );

  // Step 5: Reconstruct text from WordPiece tokens
  let keywords: Keyword[] = rawSpans.map(span => ({
    text: reconstructText(span.rawTokens),
    confidence: span.confidence,
    startTokenIndex: span.startTokenIndex,
    endTokenIndex: span.endTokenIndex,
    tokenCount: span.tokenIndices.length,
  }));

  const rawKeywordCount = keywords.length;

  // Step 6: Apply filters in sequence
  if (opts.minLength > 0) {
    keywords = filterByLength(keywords, opts.minLength);
  }

  if (opts.removeStopwords) {
    keywords = filterStopwords(keywords, opts.customStopwords);
  }

  if (opts.minConfidence > 0) {
    keywords = filterByConfidence(keywords, opts.minConfidence);
  }

  keywords = deduplicate(keywords);

  // Step 7: Sort by confidence (descending)
  keywords.sort((a, b) => b.confidence - a.confidence);

  // Step 8: Build result
  return {
    keywords,
    totalTokens: tokenizerOutput.tokens.length,
    rawKeywordCount,
    filteredKeywordCount: keywords.length,
    metadata: {
      modelOutputShape: output.shape,
      attentionTokens: tokenizerOutput.attention_mask.filter(m => m === 1).length,
      options: opts,
    },
  };
}

// Export types
export type {
  BIOLabel,
  TokenPrediction,
  KeywordSpan,
  Keyword,
  PostProcessingOptions,
  KeywordExtractionResult,
};
