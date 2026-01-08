import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom/vitest'
import KeywordExtractorPage from '../page'

// Mock dependencies
vi.mock('@/lib/tokenizer', () => ({
  tokenizer: {
    warmup: vi.fn().mockResolvedValue(undefined),
    tokenize: vi.fn().mockResolvedValue({
      input_ids: [101, 7308, 4083, 102],
      attention_mask: [1, 1, 1, 1],
      tokens: ['[CLS]', 'machine', 'learning', '[SEP]']
    })
  }
}))

vi.mock('@/lib/inference', () => ({
  inference: {
    warmup: vi.fn().mockResolvedValue(undefined),
    runInference: vi.fn().mockResolvedValue({
      logits: new Float32Array([
        0.1, 0.9, 0.5,  // Token 0
        0.8, 0.1, 0.6,  // Token 1
        0.7, 0.2, 0.5,  // Token 2
        0.1, 0.9, 0.4   // Token 3
      ]),
      shape: [1, 4, 3]
    })
  }
}))

vi.mock('@/lib/postprocess', () => ({
  extractKeywords: vi.fn().mockReturnValue({
    keywords: [
      { text: 'machine', confidence: 0.92 },
      { text: 'learning', confidence: 0.87 }
    ]
  })
}))

describe('KeywordExtractorPage - Debouncing', () => {
  afterEach(() => {
    vi.clearAllMocks()
  })

  it('should trigger extraction after 500ms of no typing', async () => {
    const { inference } = await import('@/lib/inference')

    render(<KeywordExtractorPage />)

    // Wait for models to load
    await waitFor(() => {
      expect(screen.queryByText('Loading models...')).not.toBeInTheDocument()
    }, { timeout: 5000 })

    const textarea = screen.getByPlaceholderText(/Enter your text paragraph/i)
    const user = userEvent.setup()

    // Type text
    await user.type(textarea, 'machine learning')

    // Wait for debounce (500ms) + some buffer
    await waitFor(() => {
      expect(inference.runInference).toHaveBeenCalled()
    }, { timeout: 1500 })
  })

  it('should clear results when input becomes empty', async () => {
    render(<KeywordExtractorPage />)

    await waitFor(() => {
      expect(screen.queryByText('Loading models...')).not.toBeInTheDocument()
    }, { timeout: 5000 })

    const textarea = screen.getByPlaceholderText(/Enter your text paragraph/i)
    const user = userEvent.setup()

    // Type and extract
    await user.type(textarea, 'machine learning')

    await waitFor(() => {
      expect(screen.getByText('machine')).toBeInTheDocument()
    }, { timeout: 1500 })

    // Clear input
    await user.clear(textarea)

    // Results should disappear
    await waitFor(() => {
      expect(screen.queryByText('machine')).not.toBeInTheDocument()
    })
  })

  it('should allow manual button click to trigger extraction immediately', async () => {
    const { inference } = await import('@/lib/inference')

    render(<KeywordExtractorPage />)

    await waitFor(() => {
      expect(screen.queryByText('Loading models...')).not.toBeInTheDocument()
    }, { timeout: 5000 })

    const textarea = screen.getByPlaceholderText(/Enter your text paragraph/i)
    const user = userEvent.setup()

    // Type text
    await user.type(textarea, 'machine learning')

    // Find and click button (should be enabled after typing)
    const button = screen.getByRole('button', { name: /Extract Keywords/i })
    await user.click(button)

    // Should trigger inference immediately
    await waitFor(() => {
      expect(inference.runInference).toHaveBeenCalled()
    }, { timeout: 500 })
  })

  it('should display loading indicator during extraction', async () => {
    const { inference } = await import('@/lib/inference')

    // Make inference take some time
    vi.mocked(inference.runInference).mockImplementation(
      () => new Promise(resolve => setTimeout(() => resolve({
        logits: new Float32Array([0.1, 0.9, 0.5]),
        shape: [1, 1, 3]
      }), 300))
    )

    render(<KeywordExtractorPage />)

    await waitFor(() => {
      expect(screen.queryByText('Loading models...')).not.toBeInTheDocument()
    }, { timeout: 5000 })

    const textarea = screen.getByPlaceholderText(/Enter your text paragraph/i)
    const user = userEvent.setup()

    // Type text
    await user.type(textarea, 'machine learning')

    // Wait for auto-extraction to trigger
    await waitFor(() => {
      expect(screen.getByText('Extracting...')).toBeInTheDocument()
    }, { timeout: 1500 })
  })

  it('should handle button disabled state correctly', async () => {
    render(<KeywordExtractorPage />)

    await waitFor(() => {
      expect(screen.queryByText('Loading models...')).not.toBeInTheDocument()
    }, { timeout: 5000 })

    const button = screen.getByRole('button', { name: /Extract Keywords/i })

    // Button should be disabled when input is empty
    expect(button).toBeDisabled()

    // Type something
    const textarea = screen.getByPlaceholderText(/Enter your text paragraph/i)
    const user = userEvent.setup()
    await user.type(textarea, 'test')

    // Button should now be enabled
    await waitFor(() => {
      expect(button).not.toBeDisabled()
    })
  })
})
