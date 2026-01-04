# INT8 Quantization Browser Limitation

## Summary

The INT8 quantized model **does not work in web browsers** due to ONNX Runtime Web limitations.

## Root Cause

Our INT8 model uses **dynamic quantization** which creates these operators:
- `DynamicQuantizeLinear` (49 nodes)
- `MatMulInteger` (49 nodes)
- `DequantizeLinear` (2 nodes)

These operators have **incomplete/broken support** in ONNX Runtime Web, even with WebGPU backend.

## Evidence

### Python (onnxruntime) - Works ✅
```
Token 1 (machine):  [ 7.63, -3.01, -2.67] → B-KEY ✓
Token 2 (learning): [-4.74,  6.85, -2.93] → I-KEY ✓
```

### Browser (onnxruntime-web) - Broken ❌
```
Token 1 (machine):  [-5.51, -4.27,  7.98] → O ✗
Token 2 (learning): [-5.77, -4.41,  8.19] → O ✗
```

Same model file, same inputs, completely different outputs.

## Quantization Types

| Type | Method | Browser Support | Notes |
|------|--------|-----------------|-------|
| **Dynamic INT8** | `is_static=False` | ❌ Broken | Uses DynamicQuantizeLinear |
| **Static INT8** | `is_static=True` | ⚠️ Limited | Requires calibration data |
| **FP32** | No quantization | ✅ Perfect | Recommended for web |

## Solution

**Use FP32 model for browser deployment:**

```typescript
// In app/page.tsx
await inference.warmup({ modelVariant: 'fp32' })
```

### Trade-offs

| Model | Size | Browser Support | Accuracy | Setup |
|-------|------|-----------------|----------|-------|
| FP32 | 253 MB | ✅ Perfect | Best | Simple |
| Dynamic INT8 | 64 MB | ❌ Broken | Good | Simple |
| Static INT8 | ~70 MB | ⚠️ Complex | Good | Very complex |

## Why FP32 is Acceptable

1. **Modern broadband**: 253 MB loads in 5-10 seconds
2. **Browser caching**: Instant load after first visit
3. **Best compatibility**: Works everywhere
4. **Best accuracy**: No quantization loss
5. **Simpler deployment**: No compatibility workarounds

## Future Options

If 253 MB becomes problematic:

1. **Try Static INT8 Quantization** (requires calibration dataset)
2. **Wait for onnxruntime-web improvements**
3. **Use model compression** (knowledge distillation, pruning)
4. **Lazy loading** (load on demand, not on page load)

## References

- ONNX Runtime Web operator support: https://onnxruntime.ai/docs/reference/operators/
- Dynamic quantization: Uses `DynamicQuantizeLinear` operator
- Static quantization: Uses `QLinearMatMul` operator (better web support)

## Testing Performed

```bash
# Python INT8 - Works
uv run pytest tests/test_onnx_model.py::TestBIOLabelPredictions -v
# Result: 6 passed ✓

# Browser FP32 - Works
# Result: Keywords extracted correctly ✓

# Browser INT8 - Broken
# Result: All tokens predicted as O (Outside) ✗
```

## Conclusion

**For browser deployment, use FP32 model.** The 253 MB size is acceptable for modern web applications, and browser caching makes subsequent loads instant.
