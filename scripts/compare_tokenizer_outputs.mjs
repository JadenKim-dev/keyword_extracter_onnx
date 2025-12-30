#!/usr/bin/env node
/**
 * Compare tokenizer outputs between browser and Python implementations
 *
 * This script loads the browser tokenizer and tokenizes the same test inputs
 * as the Python validation script, then displays the results for comparison.
 */

import { AutoTokenizer } from '@huggingface/transformers';

// Test case to compare (matching Python validation)
const testInput = "Hello world";

console.log('🔍 Browser Tokenizer Output Comparison\n');
console.log('Loading tokenizer from: ./public/models/keyword_model');

// Load tokenizer
const tokenizer = await AutoTokenizer.from_pretrained('./public/models/keyword_model', {
  local_files_only: true
});

console.log('✓ Tokenizer loaded\n');

// Tokenize with same config as Python
const encoded = await tokenizer(testInput, {
  padding: 'max_length',
  truncation: true,
  max_length: 512,
  return_tensor: false
});

// Extract arrays
const input_ids = Array.isArray(encoded.input_ids[0])
  ? encoded.input_ids[0]
  : encoded.input_ids;

const attention_mask = Array.isArray(encoded.attention_mask[0])
  ? encoded.attention_mask[0]
  : encoded.attention_mask;

// Decode tokens
const tokens = input_ids.map(id => tokenizer.decode([id], { skip_special_tokens: false }));

console.log(`Test input: "${testInput}"`);
console.log('\nResults:');
console.log('- First 10 input_ids:', input_ids.slice(0, 10));
console.log('- First 10 attention_mask:', attention_mask.slice(0, 10));
console.log('- First 10 tokens:', tokens.slice(0, 10));

console.log('\nArray lengths:');
console.log('- input_ids:', input_ids.length);
console.log('- attention_mask:', attention_mask.length);
console.log('- tokens:', tokens.length);

console.log('\nValidation:');
console.log('- Real token count:', attention_mask.reduce((sum, val) => sum + val, 0));
console.log('- Padding count:', attention_mask.filter(val => val === 0).length);
console.log('- First token is [CLS]:', tokens[0] === '[CLS]');
console.log('- Has [SEP] token:', tokens.includes('[SEP]'));

// Expected Python output for comparison
console.log('\n' + '='.repeat(60));
console.log('Expected Python output (for comparison):');
console.log('- First 10 input_ids: [101, 8667, 1362, 102, 0, 0, 0, 0, 0, 0]');
console.log('- First 10 attention_mask: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]');
console.log("- First 10 tokens: ['[CLS]', 'Hello', 'world', '[SEP]', '[PAD]', ...]");

console.log('\n✓ Comparison complete');
