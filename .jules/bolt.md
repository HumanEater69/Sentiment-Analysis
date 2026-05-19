## 2024-05-19 - Pandas Vectorization vs Apply
**Learning:** `pandas.DataFrame.apply` across rows is effectively a Python for-loop and extremely slow for mathematical operations. In this codebase, calculating a ratio over 500,000 rows took ~26 seconds with `apply` but only ~0.02 seconds with `numpy.where`.
**Action:** Always look for opportunities to replace `df.apply` with vectorized `numpy` or `pandas` operations when performing mathematical or conditional logic across entire columns.
