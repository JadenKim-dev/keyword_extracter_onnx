/**
 * Output format from the tokenizer
 */
export interface TokenizerOutput {
  /** Array of token IDs (integers) */
  input_ids: number[];
  /** Array indicating which tokens should be attended to (1) vs padding (0) */
  attention_mask: number[];
  /** Array of string tokens (decoded from IDs) */
  tokens: string[];
}
