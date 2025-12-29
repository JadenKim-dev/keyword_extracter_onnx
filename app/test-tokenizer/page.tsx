'use client';

import { useState, useEffect } from 'react';
import { tokenizer } from '@/lib/tokenizer';
import type { TokenizerOutput } from '@/lib/tokenizer';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export default function TestTokenizerPage() {
  const [text, setText] = useState('Hello world! This is a test of the DistilBERT tokenizer.');
  const [result, setResult] = useState<TokenizerOutput | null>(null);
  const [decodedText, setDecodedText] = useState<string>('');
  const [isReady, setIsReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Pre-warm tokenizer on mount
  useEffect(() => {
    (async () => {
      console.log('Warming up tokenizer...');
      
      try {
        await tokenizer.warmup();
      } catch (err) {
        console.error('Failed to warm up tokenizer:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        return;
      }
      
      setIsReady(tokenizer.isReady());
      console.log('✅ Tokenizer ready!');
    })();
  }, []);

  const handleTokenize = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    setIsLoading(true);
    setError(null);

    const startTime = performance.now();
    
    try {
      const output = await tokenizer.tokenize(text);
      setResult(output);

      // Also decode the tokens back to text
      const decoded = await tokenizer.decode(output.input_ids);
      setDecodedText(decoded);
    } catch (err) {
      console.error('Tokenization failed:', err);
      setError(err instanceof Error ? err.message : 'Tokenization failed');
    } finally {
      setIsLoading(false);
    }

    const endTime = performance.now();
    console.log('Tokenization completed in', (endTime - startTime).toFixed(2), 'ms');
  };

  return (
    <div className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">DistilBERT Tokenizer Test</h1>
        <p className="text-gray-600 mb-6">
          Testing @huggingface/transformers browser tokenizer
        </p>

        {/* Status indicator */}
        <Card className="mb-6">
          <CardContent className="flex items-center gap-2 py-4">
            <div
              className={`w-3 h-3 rounded-full ${
                isReady ? 'bg-green-500' : 'bg-yellow-500'
              }`}
            />
            <span className="font-medium">
              Tokenizer Status: {isReady ? 'Ready' : 'Loading...'}
            </span>
          </CardContent>
        </Card>

        {/* Error display */}
        {error && (
          <Card className="mb-6 border-red-200 bg-red-50">
            <CardContent className="py-4">
              <p className="text-red-800 font-medium">Error:</p>
              <p className="text-red-600">{error}</p>
            </CardContent>
          </Card>
        )}

        {/* Input section */}
        <Card className="mb-6">
          <CardContent className="py-6">
            <label className="block mb-2 font-medium">Input Text:</label>
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={4}
              placeholder="Enter text to tokenize..."
            />
            <Button
              onClick={handleTokenize}
              disabled={!isReady || isLoading}
              className="mt-4"
            >
              {isLoading ? 'Tokenizing...' : 'Tokenize'}
            </Button>
          </CardContent>
        </Card>

        {/* Results section */}
        {result && (
          <div className="space-y-6">
            {/* Tokens */}
            <Card>
              <CardHeader>
                <CardTitle>Tokens</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {result.tokens.map((token, idx) => (
                    <Badge
                      key={idx}
                      variant={token.startsWith('[') && token.endsWith(']') ? 'secondary' : 'outline'}
                      className={`font-mono ${
                        token.startsWith('[') && token.endsWith(']')
                          ? 'bg-purple-100 text-purple-800'
                          : 'bg-blue-100 text-blue-800'
                      }`}
                    >
                      {token}
                    </Badge>
                  ))}
                </div>
                <p className="mt-3 text-sm text-gray-600">
                  Total tokens: {result.tokens.length}
                </p>
              </CardContent>
            </Card>

            {/* Input IDs */}
            <Card>
              <CardHeader>
                <CardTitle>Input IDs</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="p-3 bg-gray-50 rounded font-mono text-sm overflow-x-auto">
                  [{result.input_ids.join(', ')}]
                </div>
                <p className="mt-3 text-sm text-gray-600">
                  Length: {result.input_ids.length}
                </p>
              </CardContent>
            </Card>

            {/* Attention Mask */}
            <Card>
              <CardHeader>
                <CardTitle>Attention Mask</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="p-3 bg-gray-50 rounded font-mono text-sm overflow-x-auto">
                  [{result.attention_mask.join(', ')}]
                </div>
                <p className="mt-3 text-sm text-gray-600">
                  Non-padding tokens: {result.attention_mask.filter(m => m === 1).length}
                </p>
              </CardContent>
            </Card>

            {/* Decoded Text */}
            <Card>
              <CardHeader>
                <CardTitle>Decoded Text</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="p-3 bg-gray-50 rounded">
                  <p className="text-gray-800">{decodedText}</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Instructions */}
        <Card className="mt-8 bg-blue-50 border-blue-200">
          <CardHeader>
            <CardTitle className="text-blue-900">Testing Instructions</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc list-inside space-y-1 text-blue-800">
              <li>Enter text in the input field above</li>
              <li>Click &quot;Tokenize&quot; to see the results</li>
              <li>Purple tokens are special tokens ([CLS], [SEP], [PAD])</li>
              <li>Blue tokens are regular word pieces</li>
              <li>Open browser DevTools Console for detailed logs</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
