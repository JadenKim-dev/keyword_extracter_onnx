'use client'

import { useState, useEffect, useCallback } from 'react'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { tokenizer } from '@/lib/tokenizer'
import { inference } from '@/lib/inference'
import { extractKeywords } from '@/lib/postprocess'
import { Loader2, Copy, AlertCircle } from 'lucide-react'

export default function KeywordExtractorPage() {
  // State management
  const [inputText, setInputText] = useState<string>('')
  const [keywords, setKeywords] = useState<Array<{ text: string; confidence: number }>>([])
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [isModelReady, setIsModelReady] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState<boolean>(false)

  // Initialize models on mount
  useEffect(() => {
    const initializeModels = async () => {
      try {
        await tokenizer.warmup()
        await inference.warmup({ modelVariant: 'fp32' })  // TEMP: Test FP32 model
        setIsModelReady(true)
      } catch (err) {
        setError('Failed to initialize models. Please refresh the page.')
        console.error('Model initialization error:', err)
      }
    }

    initializeModels()
  }, [])

  // Extract keywords handler
  const handleExtract = useCallback(async () => {
    // Validate input
    if (!inputText.trim()) {
      setError('Please enter text to extract keywords.')
      return
    }

    if (!isModelReady) {
      setError('Models are still loading. Please wait.')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      // Tokenize input text
      const tokenized = await tokenizer.tokenize(inputText)

      // Run ONNX inference
      const inferenceOutput = await inference.runInference({
        input_ids: tokenized.input_ids,
        attention_mask: tokenized.attention_mask,
      })

      // Extract keywords from model output
      const result = extractKeywords(inferenceOutput, tokenized, {
        minLength: 3,
        minConfidence: 0.5,
        removeStopwords: true,
      })

      setKeywords(result.keywords)
    } catch (err) {
      setError('Keyword extraction failed. Please try again.')
      console.error('Extraction error:', err)
    } finally {
      setIsLoading(false)
    }
  }, [inputText, isModelReady])

  // Copy keywords to clipboard
  const copyKeywords = useCallback(() => {
    const keywordText = keywords.map((k) => k.text).join(', ')
    navigator.clipboard.writeText(keywordText)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [keywords])

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto max-w-4xl px-4 py-8 space-y-6">
        {/* Header Section */}
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">
            Browser-Based Keyword Extraction
          </h1>
          <p className="text-muted-foreground">
            Extract keywords from text using ONNX models running directly in your browser.
            All processing happens locally - no data is sent to any server.
          </p>
        </div>

        {/* Model Loading Indicator */}
        {!isModelReady && (
          <div className="space-y-3 p-4 border rounded-md bg-muted/50">
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm font-medium">Loading models...</span>
            </div>
            <Skeleton className="h-2 w-full" />
            <Skeleton className="h-2 w-3/4" />
            <p className="text-xs text-muted-foreground">
              This may take a few moments on first load. Models are cached for future visits.
            </p>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="flex items-start gap-3 p-4 border border-destructive bg-destructive/10 rounded-md">
            <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-medium text-destructive">{error}</p>
            </div>
          </div>
        )}

        {/* Input Section */}
        <div className="space-y-2">
          <label htmlFor="text-input" className="text-sm font-medium">
            Enter Text
          </label>
          <Textarea
            id="text-input"
            placeholder="Enter your text paragraph here... For example: 'Machine learning and artificial intelligence are transforming how we process natural language.'"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={isLoading || !isModelReady}
            rows={8}
            className="resize-none"
          />
          <p className="text-xs text-muted-foreground">
            {inputText.length} characters
          </p>
        </div>

        {/* Extract Button */}
        <Button
          onClick={handleExtract}
          disabled={!isModelReady || isLoading || !inputText.trim()}
          size="lg"
          className="w-full sm:w-auto"
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Extracting Keywords...
            </>
          ) : (
            'Extract Keywords'
          )}
        </Button>

        {/* Keywords Output Section */}
        {keywords.length > 0 && (
          <div className="space-y-3 p-4 border rounded-md bg-card">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">
                Extracted Keywords ({keywords.length})
              </h2>
              <Button
                variant="outline"
                size="sm"
                onClick={copyKeywords}
                className="gap-2"
              >
                <Copy className="h-3 w-3" />
                {copied ? 'Copied!' : 'Copy'}
              </Button>
            </div>
            <div className="flex flex-wrap gap-2">
              {keywords.map((kw, idx) => (
                <Badge key={idx} variant="default" className="gap-1.5 px-3 py-1">
                  <span>{kw.text}</span>
                  <span className="text-xs opacity-70 font-normal">
                    {(kw.confidence * 100).toFixed(0)}%
                  </span>
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {keywords.length === 0 && isModelReady && !isLoading && inputText.trim() === '' && (
          <div className="text-center text-muted-foreground py-12 border border-dashed rounded-md">
            <p className="text-sm">
              Enter text above and click &quot;Extract Keywords&quot; to begin
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
