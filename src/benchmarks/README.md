# Performance Benchmarks

## test.py - Pandas vs Polars Comparison

Benchmarks comparing Pandas and Polars on 10 million rows of data.

### Tests

1. **Filtering** (age > 30)
   - Bad Pandas (apply): ~8.5s ❌
   - Good Pandas (vectorized): ~0.15s ✅
   - Polars: ~0.08s ⚡

2. **Complex Operations** (conditional bonus calculation + groupby)
   - Pandas + NumPy: ~0.45s
   - Polars: ~0.18s (2-3x faster)

3. **Multiple Aggregations** (mean, std, min, max)
   - Pandas: ~0.65s
   - Polars: ~0.22s (3x faster)

### Key Takeaways

✅ **Always use vectorized operations in Pandas** (avoid `.apply()`)
✅ **Polars is 2-3x faster** for complex operations
✅ **Proper Pandas + NumPy** is already quite fast
⚠️ **Migration isn't free** - evaluate if 2-3x speedup justifies the effort

### Run

```bash
# From project root
source venv/bin/activate
python src/benchmarks/test.py

# Or after pip install -e .
python -m benchmarks.test
```
