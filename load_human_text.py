"""
load_human_text.py
Loads human-written news text (EN/DE/ZH) to serve as the "human" class.
Same public benchmark datasets as before — see README for citations.
"""

import pandas as pd
from datasets import load_dataset  # pyright: ignore[reportMissingImports]

N_PER_LANG = 120  

def load_human_samples():
    en = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=42).select(range(N_PER_LANG))
    de = load_dataset("community-datasets/gnad10", split="train").shuffle(seed=42).select(range(N_PER_LANG))

    df = pd.concat([
        pd.DataFrame({"text": en["text"], "language": "en"}),
        pd.DataFrame({"text": de["text"], "language": "de"}),
    ], ignore_index=True)
    df["origin"] = "human"
    df["sample_id"] = df["language"] + "_" + df.groupby("language").cumcount().astype(str)
    df.to_csv("data/human_samples.csv", index=False, encoding="utf-8")
    print(f"Saved {len(df)} human samples to data/human_samples.csv")
    return df

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    load_human_samples()
