# Week 1

## Overview

This folder contains python models for calculating **Customer Lifetime Value (CLV)** using both simple averaging methods and advanced probabilistic modeling techniques.

We are moving from simple to advance models for calculating CLV.

---

## Dataset Structure

The data we are going to use should contain these keys:

| Column | Description |
|--------|-------------|
| **frequency** | Number of repeat purchases made by each customer (excludes the initial purchase). Customers with frequency = 0 made only one purchase |
| **recency** | Time elapsed since the customer's most recent purchase, measured in weeks |
| **T** | Total observation period for each customer, representing how long we've been tracking them in weeks |
| **monetary_value** | Average order value (AOV) calculated from all customer purchases |

### Data Quality Notes

- Some customers have `frequency = 0, recency = 0, monetary_value = 0`. These are one-time purchasers or may have data quality issues
- The Gamma-Gamma model requires `frequency > 0`, so these customers are filtered during model fitting
- However, CLV calculations still produce estimates for zero-frequency customers using population average fallbacks
- we are using week as a unit

---

## Files in This Folder

### 1. **data.py**
**Purpose:** Load and explore the customer dataset

This script demonstrates how to import the CDNOW customer data:
- Loads the customer-level summary data with monetary values
- Displaying some rows
- Shows dataset shape (number of customers and features)
- Printing column names and data types

**Run this first** to understand the type of data.

---

### 2. **simple-avg.py**
**Purpose:** Calculate CLV by taking simple baseline averages

What it is doing:
- Calculates cohort-level Average Order Value (AOV)
- Computes average weekly transaction rate across all customers
- We are calculating over a 26-week period

**Key Limitation:** This method treats all customers identically and doesn't capture individual differences. A frequent buyer is predicted the same as a one-time purchaser.

**Output Example:**
```
Average Order Value: $14.08
Avg weekly transactions: 0.0319
Simple-average CLV (26w): $11.67
```

---

### 3. **bg-nbd.py**
**Purpose:** Predict individual transaction frequency using the BG/NBD model

Implements the **Beta-Geometric/Negative Binomial Distribution (BG/NBD)** model:
- Fits a probabilistic model to customer frequency, recency, and tenure (T)
- Predicts expected number of transactions for each customer over the next 26 weeks
- Accounts for customer lifecycle and inactivity patterns

**Key Features:**
- Recognizes that some customers are inactive while others remain active
- Produces individualized predictions ranging from <0.1 to 16+ transactions
- Treats zero-frequency customers as potentially still active (rather than writing them off)
- Uses a `penalizer_coef=0.001` to prevent overfitting

**Output Includes:**
- Summary statistics of predicted transactions
- Individual predictions for sample customers
- Wide variation reflecting different customer behaviors

---

### 4. **gamma-gamma.py**
**Purpose:** Complete CLV calculation combining transaction frequency and monetary value

Implements the full CLV pipeline using both BG/NBD and Gamma-Gamma models:

**Steps:**
1. Fits BG/NBD model to predict transaction frequency (same as bg-nbd.py)
2. Fits **Gamma-Gamma model** on customers with `frequency > 0` to estimate individual monetary values
3. Combines predictions to calculate complete 26-week CLV with discounting

**Key Parameters:**
- `time=26`: Projects CLV over 26 weeks
- `discount_rate=0.005`: Applies 0.5% weekly discount (≈30% annual), reflecting the time value of money
- Includes fallback mechanism for customers with zero frequency (uses population average monetary value)

**Output Example:**
```
CLV Summary (26 weeks, 0.5% weekly discount):
mean:  $63.02
50%:   $21.32
75%:   $34.58
max:   $2,484.31
```

This reveals significant value concentration among high-value customers compared to simple averaging.

---

## Installation

Before running these scripts, install required dependencies:
```bash
pip install lifetimes pandas numpy
```

**Required Libraries:**
- `lifetimes`: Implements probabilistic CLV models (BG/NBD, Gamma-Gamma)
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical operations

---

## Running the Scripts

Execute scripts in recommended order:
```bash
# 1. Explore the data
python data.py

# 2. Calculate simple baseline
python simple-avg.py

# 3. Predict transaction frequency
python bg-nbd.py

# 4. Calculate complete CLV
python gamma-gamma.py
```
---
