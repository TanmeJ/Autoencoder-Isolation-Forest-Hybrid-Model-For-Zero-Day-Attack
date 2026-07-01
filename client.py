"""
Small helper script for viewing the latest saved results.

Run:
    python client.py
"""

import os

import pandas as pd

import config


def main():
    summary_path = os.path.join(config.RESULTS_DIR, "results_summary.csv")

    if not os.path.exists(summary_path):
        print("No results found yet.")
        print("Run: python main.py train")
        return

    results = pd.read_csv(summary_path)
    print("\nLatest Results")
    print("-" * 60)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
