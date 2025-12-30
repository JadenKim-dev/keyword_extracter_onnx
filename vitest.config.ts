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
      // Enforce 90% coverage threshold for tokenizer.ts
      // Note: branches set to 85% due to singleton pattern edge cases that are difficult to test
      thresholds: {
        'lib/tokenizer.ts': {
          lines: 90,
          functions: 90,
          branches: 85,
          statements: 90,
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
