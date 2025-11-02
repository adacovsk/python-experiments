import pandas as pd
import polars as pl
import numpy as np
import time

# Create test data (10 million rows)
n = 10_000_000
data = {
    'age': np.random.randint(18, 80, n),
    'salary': np.random.randint(30000, 150000, n),
    'department': np.random.choice(['Sales', 'Engineering', 'Marketing'], n)
}

# ============ TEST 1: FILTERING ============

# Bad Pandas (slow)
df_pd = pd.DataFrame(data)
start = time.time()
result = df_pd[df_pd['age'].apply(lambda x: x > 30)]  # 😱 Don't do this
print(f"Pandas apply: {time.time() - start:.2f}s")

# Good Pandas + NumPy (fast)
start = time.time()
result = df_pd[df_pd['age'] > 30]  # Uses NumPy under the hood
print(f"Pandas vectorized: {time.time() - start:.2f}s")

# Polars
df_pl = pl.DataFrame(data)
start = time.time()
result = df_pl.filter(pl.col('age') > 30)
print(f"Polars: {time.time() - start:.2f}s")

# Results:
# Pandas apply: 8.5s (awful)
# Pandas vectorized: 0.15s (good!)
# Polars: 0.08s (faster, but not 10x)


# ============ TEST 2: COMPLEX OPERATIONS ============

# Pandas + NumPy
start = time.time()
df_pd['bonus'] = np.where(
    df_pd['age'] > 30,
    df_pd['salary'] * 0.1,
    df_pd['salary'] * 0.05
)
result = df_pd.groupby('department')['bonus'].mean()
print(f"Pandas+NumPy: {time.time() - start:.2f}s")

# Polars
start = time.time()
result = (df_pl
    .with_columns(
        pl.when(pl.col('age') > 30)
          .then(pl.col('salary') * 0.1)
          .otherwise(pl.col('salary') * 0.05)
          .alias('bonus')
    )
    .group_by('department')
    .agg(pl.col('bonus').mean())
)
print(f"Polars: {time.time() - start:.2f}s")

# Results:
# Pandas+NumPy: 0.45s
# Polars: 0.18s (2-3x faster)


# ============ TEST 3: MULTIPLE AGGREGATIONS ============

# Pandas + NumPy
start = time.time()
result = df_pd.groupby('department').agg({
    'age': ['mean', 'std', 'min', 'max'],
    'salary': ['mean', 'std', 'min', 'max']
})
print(f"Pandas: {time.time() - start:.2f}s")

# Polars
start = time.time()
result = df_pl.group_by('department').agg([
    pl.col('age').mean(),
    pl.col('age').std(),
    pl.col('age').min(),
    pl.col('age').max(),
    pl.col('salary').mean(),
    pl.col('salary').std(),
    pl.col('salary').min(),
    pl.col('salary').max()
])
print(f"Polars: {time.time() - start:.2f}s")

# Results:
# Pandas: 0.65s
# Polars: 0.22s (3x faster)