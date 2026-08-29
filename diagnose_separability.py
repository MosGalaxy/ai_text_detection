"""
diagnose_separability.py
Standalone diagnostic: trains the same classifier as train_classifier.py,
but reports BOTH training-set and test-set AUC per class.

Why this matters: if training AUC is also low (~0.5), the features/
embedding model genuinely lack the signal to separate the classes --
more data won't fix that. If training AUC is high but test AUC is low,
that's overfitting/noise from a small test set -- more data likely will
help.

Usage: point FEATURES_CSV at any saved run (current data/, or an
archived r1/r2/r3_features.csv) to check that run's separability
without needing to regenerate anything.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

# Point this at whichever run's features file you want to diagnose
FEATURES_CSV = "data/full_with_stylometric_features.csv"

STYLE_FEATURES = [
    "avg_sentence_length", "std_sentence_length", "punctuation_density",
    "avg_word_length", "type_token_ratio", "function_word_ratio",
]

def main():
    df = pd.read_csv(FEATURES_CSV, encoding="utf-8")
    df = df.dropna(subset=STYLE_FEATURES)

    group_sizes = df.groupby(["origin", "language"]).size()
    min_group_size = group_sizes.min()
    print(f"Balancing to {min_group_size} samples per (origin, language):\n{group_sizes}\n")
    df = df.groupby(["origin", "language"], group_keys=False).sample(
        n=min_group_size, random_state=42
    ).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df, df["origin"], test_size=0.25, random_state=42, stratify=df["origin"]
    )

    scaler = StandardScaler()
    X_train_style = scaler.fit_transform(X_train[STYLE_FEATURES])
    X_test_style = scaler.transform(X_test[STYLE_FEATURES])

    print("Encoding embeddings...")
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    X_train_emb = embed_model.encode(X_train["text"].tolist(), show_progress_bar=True)
    X_test_emb = embed_model.encode(X_test["text"].tolist(), show_progress_bar=True)

    X_train_combined = np.hstack([X_train_style, X_train_emb])
    X_test_combined = np.hstack([X_test_style, X_test_emb])

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_combined, y_train)

    train_probs = clf.predict_proba(X_train_combined)
    test_probs = clf.predict_proba(X_test_combined)
    y_train_bin = label_binarize(y_train, classes=clf.classes_)
    y_test_bin = label_binarize(y_test, classes=clf.classes_)

    print(f"\n{'Class':<15} {'Train AUC':>10} {'Test AUC':>10} {'Gap':>8}")
    print("-" * 45)
    for i, cls in enumerate(clf.classes_):
        train_auc = roc_auc_score(y_train_bin[:, i], train_probs[:, i])
        test_auc = roc_auc_score(y_test_bin[:, i], test_probs[:, i])
        gap = train_auc - test_auc
        print(f"{cls:<15} {train_auc:>10.3f} {test_auc:>10.3f} {gap:>8.3f}")

    print("\nInterpretation:")
    print("- Train AUC near 0.5 for a class -> features/embedding model")
    print("  lack signal for that class. More data won't help much.")
    print("- Train AUC high, test AUC low, big gap -> overfitting/noise.")
    print("  More data likely will help.")

if __name__ == "__main__":
    main()
