"""
stylometric_features.py
Classical, interpretable features for human-vs-AI text detection:
sentence length stats, punctuation density, function-word frequency,
type-token ratio (lexical diversity), average word length.

These are language-agnostic-ish but function-word lists are per-language —
only EN/DE lists are filled in below; add ZH function words (or skip that
one feature for ZH, since Chinese doesn't tokenize on whitespace the way
EN/DE do — segment with jieba first).
"""

import re
import numpy as np
import pandas as pd

FUNCTION_WORDS = {
    "en": {"the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "with", "for", "is", "was", "are"},
    "de": {"der", "die", "das", "und", "oder", "aber", "von", "zu", "in", "auf", "mit", "für", "ist", "war", "sind"},
}

def sentence_lengths(text):
    sentences = re.split(r'[.!?。！？]+', text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    return lengths if lengths else [0]

def extract_features(text, language):
    words = text.split()
    n_words = max(len(words), 1)

    sent_lens = sentence_lengths(text)
    punctuation_count = len(re.findall(r'[.,;:!?]', text))
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    type_token_ratio = len(set(words)) / n_words

    fw_set = FUNCTION_WORDS.get(language, set())
    fw_ratio = sum(1 for w in words if w.lower() in fw_set) / n_words if fw_set else np.nan

    return {
        "avg_sentence_length": np.mean(sent_lens),
        "std_sentence_length": np.std(sent_lens),
        "punctuation_density": punctuation_count / n_words,
        "avg_word_length": avg_word_len,
        "type_token_ratio": type_token_ratio,
        "function_word_ratio": fw_ratio,
    }

def build_feature_matrix(df):
    feats = df.apply(lambda row: extract_features(row["text"], row["language"]), axis=1)
    feat_df = pd.DataFrame(list(feats))
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)

if __name__ == "__main__":
    df = pd.concat([
        pd.read_csv("data/human_samples.csv", encoding="utf-8"),
        pd.read_csv("data/ai_samples.csv", encoding="utf-8"),
    ], ignore_index=True)

    n_before = len(df)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    n_after = len(df)
    if n_after < n_before:
        print(f"Dropped {n_before - n_after} rows with missing/empty text.")

    full = build_feature_matrix(df)
    full.to_csv("data/full_with_stylometric_features.csv", index=False, encoding="utf-8")