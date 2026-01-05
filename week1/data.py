import pandas as pd
import numpy as np
from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value

# Customer-level summary: frequency, recency, T, monetary_value (AOV)
summary = load_cdnow_summary_data_with_monetary_value()
print(summary.head(3))
print("Shape:", summary.shape)
print("Columns:", list(summary.columns))
