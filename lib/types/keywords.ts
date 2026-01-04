/**
 * Type definitions for keyword extraction and post-processing
 */

/**
 * BIO tagging labels for token classification
 *
 * Based on ml6team/keyphrase-extraction-distilbert-inspec config.json:
 * - 0: "B-KEY" (Begin-keyword)
 * - 1: "I-KEY" (Inside-keyword)
 * - 2: "O" (Outside - non-keyword)
 *
 * NOTE: Label order changed from previous adapted model which used:
 * {0: "O", 1: "B-KEY", 2: "I-KEY"}
 */
export enum BIOLabel {
  B_KEY = 0,  // Begin-keyword (was 1 in old model)
  I_KEY = 1,  // Inside-keyword (was 2 in old model)
  O = 2,      // Outside (was 0 in old model)
}

/**
 * Predicted label with confidence for a single token
 */
export interface TokenPrediction {
  /** Token index in the original sequence */
  tokenIndex: number;
  /** Predicted BIO label */
  label: BIOLabel;
  /** Confidence score for this prediction [0-1] */
  confidence: number;
  /** Probabilities for all classes [O, B-KEY, I-KEY] */
  probabilities: [number, number, number];
  /** The actual token string */
  token: string;
}

/**
 * Raw keyword span (before text reconstruction)
 */
export interface KeywordSpan {
  /** Start token index (inclusive) */
  startTokenIndex: number;
  /** End token index (exclusive) */
  endTokenIndex: number;
  /** Token indices included in this span */
  tokenIndices: number[];
  /** Raw tokens (may include ## prefixes) */
  rawTokens: string[];
  /** Average confidence across all tokens in span */
  confidence: number;
}

/**
 * Final extracted keyword with metadata
 */
export interface Keyword {
  /** Reconstructed keyword text (## prefixes removed, properly joined) */
  text: string;
  /** Overall confidence score [0-1] */
  confidence: number;
  /** Start token index in original sequence */
  startTokenIndex: number;
  /** End token index in original sequence */
  endTokenIndex: number;
  /** Number of tokens in this keyword */
  tokenCount: number;
  /** Character-level position info (optional, for future use) */
  charPosition?: {
    start: number;
    end: number;
  };
}

/**
 * Configuration options for post-processing
 */
export interface PostProcessingOptions {
  /** Minimum character length for keywords (default: 3) */
  minLength?: number;
  /** Minimum confidence threshold [0-1] (default: 0.5) */
  minConfidence?: number;
  /** Remove stopwords (default: true) */
  removeStopwords?: boolean;
  /** Custom stopwords set (default: English stopwords) */
  customStopwords?: Set<string>;
}

/**
 * Complete keyword extraction result
 */
export interface KeywordExtractionResult {
  /** List of extracted keywords */
  keywords: Keyword[];
  /** Total number of tokens processed */
  totalTokens: number;
  /** Number of keywords before filtering */
  rawKeywordCount: number;
  /** Number of keywords after filtering */
  filteredKeywordCount: number;
  /** Processing metadata */
  metadata: {
    /** Model output shape */
    modelOutputShape: [number, number, number];
    /** Number of attention tokens (non-padding) */
    attentionTokens: number;
    /** Options used for post-processing */
    options: Required<PostProcessingOptions>;
  };
}
