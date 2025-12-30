/**
 * Vitest Test Setup
 *
 * Configures the test environment for browser-based tokenizer tests.
 * Since we're using real model loading, minimal setup is required.
 */

import { env } from '@huggingface/transformers';
import path from 'path';

// Configure transformers.js environment for test
env.localModelPath = path.resolve(__dirname, '../../public/models');
