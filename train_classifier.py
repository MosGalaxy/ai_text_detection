"""
train_classifier.py
3-class LLM attribution: Human vs. Gemini vs. GPT-OSS (via Groq).

Trains on stylometric features + multilingual embeddings concatenated
(per the restructured plan — skip separate ensembles for v1, this is
already a meaningfully harder task than binary classification).

Evaluated with one-vs-rest ROC-AUC per class and a normalized confusion
matrix, since raw counts are less readable once you have 3+ classes.

Honest framing (keep this in your README): expect embeddings to carry
most of the separating signal for Gemini-vs-GPT-OSS specifically, since both
are RLHF'd English/German prose from broadly similar web training data.
Stylometric features should still help most for Human-vs-either-AI.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

STYLE_FEATURES = [
    "avg_sentence_length", "std_sentence_length", "punctuation_density",
    "avg_word_length", "type_token_ratio", "function_word_ratio",
]
CLASSES = ["human", "gemini", "gpt_oss_groq"]

def main():
    df = pd.read_csv("data/full_with_stylometric_features.csv", encoding="utf-8")
    df = df.dropna(subset=STYLE_FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        df, df["origin"], test_size=0.25, random_state=42, stratify=df["origin"]
    )

    # Stylometric features, scaled
    scaler = StandardScaler()
    X_train_style = scaler.fit_transform(X_train[STYLE_FEATURES])
    X_test_style = scaler.transform(X_test[STYLE_FEATURES])

    # Embeddings
    print("Encoding embeddings...")
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    X_train_emb = embed_model.encode(X_train["text"].tolist(), show_progress_bar=True)
    X_test_emb = embed_model.encode(X_test["text"].tolist(), show_progress_bar=True)

    # Concatenate stylometric + embedding features
    X_train_combined = np.hstack([X_train_style, X_train_emb])
    X_test_combined = np.hstack([X_test_style, X_test_emb])

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_combined, y_train)
    probs = clf.predict_proba(X_test_combined)
    preds = clf.predict(X_test_combined)

    print("\n=== Classification report ===")
    print(classification_report(y_test, preds))

    # One-vs-rest ROC-AUC per class
    y_test_binarized = label_binarize(y_test, classes=clf.classes_)
    print("\n=== One-vs-rest ROC-AUC per class ===")
    plt.figure(figsize=(6, 5))
    for i, cls in enumerate(clf.classes_):
        auc = roc_auc_score(y_test_binarized[:, i], probs[:, i])
        print(f"{cls}: {auc:.3f}")
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC — Human vs. Gemini vs. GPT-OSS")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_multiclass.png", dpi=150)
    plt.close()

    # Normalized confusion matrix — the actual interesting result here
    cm = confusion_matrix(y_test, preds, labels=clf.classes_, normalize="true")
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues")
    plt.title("Normalized Confusion Matrix\n(row = true class, values = fraction predicted)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("outputs/confusion_multiclass.png", dpi=150)
    plt.close()
    print("\nSaved outputs/roc_multiclass.png and outputs/confusion_multiclass.png")
    print("Look specifically at the gemini <-> gpt_oss_groq confusion cells — "
          "that's the interesting result to discuss in your README/interview.")

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()
