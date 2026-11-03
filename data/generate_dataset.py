"""
Generate a reproducible synthetic retail-customer dataset.

The dataset mimics the well-known "Mall Customers" structure
(CustomerID, Gender, Age, Annual Income, Spending Score) so the project is
fully self-contained and reproducible. It is built from five latent customer
profiles; K-Means is NOT told about them, it has to rediscover them.

Usage:
    python data/generate_dataset.py
    python data/generate_dataset.py --n 500 --seed 42 --out data/customers.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# (share of customers, age mean/std, income mean/std in k$, spending mean/std)
PROFILES = [
    (0.20, (25, 4), (30, 8), (75, 9)),    # young, low income, high spenders
    (0.20, (45, 8), (85, 10), (20, 8)),   # established, high income, low spenders
    (0.22, (42, 10), (55, 8), (50, 8)),   # average customers
    (0.18, (32, 6), (88, 10), (80, 8)),   # affluent, high spenders
    (0.20, (52, 8), (28, 8), (22, 8)),    # older, low income, low spenders
]


def generate(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for share, (a_m, a_s), (i_m, i_s), (s_m, s_s) in PROFILES:
        size = int(round(n * share))
        age = rng.normal(a_m, a_s, size).clip(18, 70)
        income = rng.normal(i_m, i_s, size).clip(15, 140)
        spend = rng.normal(s_m, s_s, size).clip(1, 100)
        rows.append(pd.DataFrame({"Age": age, "Annual_Income_k": income, "Spending_Score": spend}))

    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.iloc[:n].copy()
    df["Age"] = df["Age"].round().astype(int)
    df["Annual_Income_k"] = df["Annual_Income_k"].round().astype(int)
    df["Spending_Score"] = df["Spending_Score"].round().astype(int)
    df.insert(0, "CustomerID", np.arange(1, len(df) + 1))
    df.insert(1, "Gender", rng.choice(["Male", "Female"], size=len(df), p=[0.44, 0.56]))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--n", type=int, default=500, help="number of customers")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "customers.csv")
    args = parser.parse_args()

    data = generate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    print(f"Saved {len(data)} customers to {args.out}")
