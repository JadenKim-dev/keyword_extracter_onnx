import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    // Browser environment for DOM APIs and window object
    environment: 'happy-dom',

    // Enable global test APIs (describe, it, expect) without imports
    globals: true,

    // Environment variables for tests
    env: {
      NODE_ENV: 'test',
    },

    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      include: ['lib/**/*.ts'],
      exclude: [
        'lib/**/*.test.ts',
        'lib/**/*.spec.ts',
        'lib/types/**/*',
      ],
      // Enforce coverage thresholds for core modules
      // Note: branches set to 80% for tokenizer due to singleton pattern edge cases
      // Note: inference.ts set to 75% due to multi-backend fallback and error paths that are hard to test in unit tests
      thresholds: {
        'lib/tokenizer.ts': {
          lines: 90,
          functions: 90,
          branches: 80,
          statements: 90,
        },
        'lib/inference.ts': {
          lines: 75,
          functions: 100,
          branches: 70,
          statements: 75,
        },
      },
      all: true,
    },

    // Test setup file
    setupFiles: ['./lib/__tests__/setup.ts'],

    // Extended timeouts for real model loading (2-5 seconds)
    testTimeout: 60000,  // 60 seconds
    hookTimeout: 60000,

    // Exclude patterns for test discovery
    exclude: [
      '**/node_modules/**',
      '**/public/**',
      '**/.next/**',
      '**/coverage/**',
      '**/.{git,cache}/**',
    ],
  },

  // Path alias to match tsconfig.json
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    },
  },
});
