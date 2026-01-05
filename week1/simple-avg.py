import pandas as pd
import numpy as np
from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value

# Customer-level summary: frequency, recency, T, monetary_value (AOV)
summary = load_cdnow_summary_data_with_monetary_value()

# Simple Baseline CLV Calculation
# Cohort Average Order Value (exclude NaNs)
aov = summary['monetary_value'].dropna().mean()

# Average weekly transaction rate across customers
avg_weekly_rate = (summary['frequency'] / summary['T']).mean()

# Baseline projection for 26 weeks
horizon_weeks = 26
clv_simple = avg_weekly_rate * aov * horizon_weeks

print(f"Average Order Value: ${aov:.2f}")
print(f"Avg weekly transactions: {avg_weekly_rate:.4f}")
print(f"Simple-average CLV (26w): ${clv_simple:.2f}")