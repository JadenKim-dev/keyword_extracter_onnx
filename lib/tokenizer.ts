import { AutoTokenizer, env, PreTrainedTokenizer, Tensor } from '@huggingface/transformers';
import { isFunction } from 'es-toolkit';
import type { TokenizerOutput } from './types/tokenizer';

// Enable local model loading in browser
if (typeof window !== 'undefined') {
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
}

// Constants
const MODEL_NAME = 'keyword_model';
const MAX_LENGTH = 512;
const TOKENIZER_NOT_READY_ERROR = 'Tokenizer not initialized. Call warmup() first.';
const EMPTY_INPUT_ERROR = 'Input text cannot be empty';
const SSR_ERROR = 'Tokenizer can only be used in browser environment';

/**
 * Type guard: Check if value is a 2D array
 */
function is2DArray<T>(value: unknown): value is T[][] {
  return Array.isArray(value) && Array.isArray(value[0]);
}

/**
 * Helper: Convert Tensor or 2D array to 1D array
 * 
 * Note: Although typed as Tensor | T[] | T[][], we use return_tensor=false
 * in tokenizer config, so this always receives JavaScript arrays in practice.
 * The Tensor check is kept for type compatibility.
 */
function toFlatArray<T>(data: Tensor | T[] | T[][]): T[] {
  // Step 1: Convert Tensor to array
  const array: T[] | T[][] = data instanceof Tensor ? data.tolist() : data;
  
  // Step 2: Handle 2D array - extract first batch
  if (is2DArray(array)) {
    return array[0];
  }
  
  // Step 3: Return 1D array as-is
  return array;
}

/**
 * DistilBERT Tokenizer
 *
 * Manages a DistilBERT tokenizer instance to prevent
 * memory issues and redundant loading.
 */
class DistilBertTokenizer {
  private tokenizer: PreTrainedTokenizer | null = null;
  private loadingPromise: Promise<void> | null = null;

  /**
   * Initialize the tokenizer (lazy loading)
   * Uses promise caching to prevent race conditions
   */
  private async initializeTokenizer(): Promise<void> {
    if (this.tokenizer) return;

    // Currently loading - wait for existing promise
    if (this.loadingPromise) {
      await this.loadingPromise;
      return;
    }

    this.loadingPromise = AutoTokenizer.from_pretrained(MODEL_NAME)
      .then(tokenizer => {
        this.tokenizer = tokenizer;
      })
      .finally(() => {
        this.loadingPromise = null;
      });

    await this.loadingPromise;
  }

  /**
   * Ensure tokenizer is ready
   */
  private ensureReady(): void {
    if (!this.tokenizer) {
      throw new Error(TOKENIZER_NOT_READY_ERROR);
    }
  }

  /**
   * Tokenize text into IDs, attention mask, and tokens
   *
   * @param text - Single text string or array of texts
   * @returns Promise with input_ids, attention_mask, and tokens
   *
   * @example
   * ```typescript
   * const output = await tokenize("Hello world");
   * console.log(output.tokens); // ['[CLS]', 'hello', 'world', '[SEP]', ...]
   * console.log(output.input_ids); // [101, 7592, 2088, 102, 0, 0, ...]
   * ```
   */
  public async tokenize(text: string | string[]): Promise<TokenizerOutput> {
    this.ensureReady();

    // Validate input
    if (text.length === 0) {
      throw new Error(EMPTY_INPUT_ERROR);
    }

    // Call the tokenizer
    const encoded = this.tokenizer!._call(text, {
      padding: 'max_length',
      truncation: true,
      max_length: MAX_LENGTH,
      return_tensor: false,
    });

    // Extract and flatten input_ids
    const input_ids = toFlatArray<number>(encoded.input_ids);

    // Decode each ID to get tokens
    const tokens = input_ids.map(id =>
      this.tokenizer!.decode([id], { skip_special_tokens: false })
    );

    // Extract and flatten attention_mask
    const attention_mask = toFlatArray<number>(encoded.attention_mask);

    return { input_ids, attention_mask, tokens };
  }

  /**
   * Decode token IDs back to text
   *
   * @param token_ids - Single token ID or array of token IDs
   * @returns Promise with decoded text
   *
   * @example
   * ```typescript
   * const text = await decode([101, 7592, 2088, 102]);
   * console.log(text); // "hello world"
   * ```
   */
  public async decode(token_ids: number | number[]): Promise<string> {
    this.ensureReady();

    const ids = Array.isArray(token_ids) ? token_ids : [token_ids];
    if (ids.length === 0) return '';

    return this.tokenizer!.decode(ids, {
      skip_special_tokens: true,
      clean_up_tokenization_spaces: true,
    });
  }

  /**
   * Check if tokenizer is ready to use
   */
  public isReady(): boolean {
    return this.tokenizer !== null;
  }

  /**
   * Pre-warm the tokenizer by loading it in advance
   */
  public async warmup(): Promise<void> {
    await this.initializeTokenizer();
  }
}

// Singleton instance with SSR safety
let instance: DistilBertTokenizer | null = null;

/**
 * Get the singleton tokenizer instance
 * 
 * @throws Error if called in non-browser environment (SSR)
 */
function getTokenizer(): DistilBertTokenizer {
  if (typeof window === 'undefined') {
    throw new Error(SSR_ERROR);
  }
  return instance ??= new DistilBertTokenizer();
}

export const tokenizer = new Proxy({} as DistilBertTokenizer, {
  get(_, prop: string) {
    const instance = getTokenizer();
    const value = instance[prop as keyof DistilBertTokenizer];
    return isFunction(value) ? value.bind(instance) : value;
  }
});

// Export types
export type { TokenizerOutput };
