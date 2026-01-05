import pandas as pd
import numpy as np
from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value
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

print("\nPredicted transactions (26 weeks):")
print()
print(summary['pred_txns_26w'].describe()[['count','mean','50%','max']].round(3))
print()
print(summary[['pred_txns_26w']].head(5).round(3))