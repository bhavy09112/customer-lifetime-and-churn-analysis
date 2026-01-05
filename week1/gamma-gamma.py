import pandas as pd
import numpy as np
from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value
from lifetimes import GammaGammaFitter
from lifetimes import BetaGeoFitter

# Customer-level summary: frequency, recency, T, monetary_value (AOV)
summary = load_cdnow_summary_data_with_monetary_value()

# Fit BG/NBD on frequency/recency/T (all in weeks)
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])

# Predict expected purchases for the next 26 weeks
future_weeks = 26
summary['pred_txns_26w'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    future_weeks, summary['frequency'], summary['recency'], summary['T']
)

# Fit Gamma-Gamma on customers with repeat history and monetary values
mask = (summary['frequency'] > 0) & summary['monetary_value'].notna()
ggf = GammaGammaFitter(penalizer_coef=0.001).fit(
    summary.loc[mask, 'frequency'], summary.loc[mask, 'monetary_value']
)

# Calculate CLV for 26 weeks with reasonable discount rate
# Using 0.5% weekly discount (≈ 30% annual)
clv_26w = ggf.customer_lifetime_value(
    bgf,
    summary['frequency'], summary['recency'], summary['T'],
    summary['monetary_value'],
    time=26, freq='W', discount_rate=0.005  # 0.5% weekly discount
)
summary['clv_26w'] = clv_26w

print(f"\nCLV Summary (26 weeks, 0.5% weekly discount):")
print()
print(summary['clv_26w'].describe()[['mean','50%','75%','max']].round(2))
print()
print("\nSample predictions:")
print(summary[['pred_txns_26w','clv_26w']].head(5).round(2))