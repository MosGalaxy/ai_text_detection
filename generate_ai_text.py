"""
generate_ai_text.py
Generates AI-authored counterparts to the human news samples from TWO
providers, producing a 3-class dataset: human / Gemini / GPT-OSS-via-Groq.

Uses gemini-3.5-flash-lite (RPD=500 free tier, confirmed via AI Studio's
own rate limits page) instead of gemini-3.6-flash (RPD=20 — too low for
this project). Also backs up any existing ai_samples.csv before
overwriting, so a partial/failed run never silently destroys good data.

pip install google-genai groq
"""

import os
import time
import shutil
import pandas as pd
from google import genai
from groq import Groq # pyright: ignore[reportMissingImports]

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-3.5-flash-lite"  # RPD=500 on free tier, confirmed via AI Studio

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was deprecated by Groq (June 2026)

PROMPT_TEMPLATE = {
    "en": "Rewrite the following news snippet in your own words, same topic and length, natural news style:\n\n{text}",
    "de": "Schreibe den folgenden Nachrichtenausschnitt in eigenen Worten um, gleiches Thema, ähnliche Länge, natürlicher Nachrichtenstil:\n\n{text}",
}

def generate_gemini(text, language):
    prompt = PROMPT_TEMPLATE[language].format(text=text)
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text

def generate_groq(text, language):
    prompt = PROMPT_TEMPLATE[language].format(text=text)
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.choices[0].message.content

def generate_batch(human_df, generate_fn, source_label, sample_n_per_lang, sleep_seconds):
    rows = []
    for lang in ["en", "de"]:
        subset = human_df[human_df.language == lang].head(sample_n_per_lang)
        for _, row in subset.iterrows():
            try:
                ai_text = generate_fn(row["text"], lang)
                rows.append({"text": ai_text, "language": lang, "origin": source_label, "source_id": row["sample_id"],
                             })
            except Exception as e:
                print(f"[{source_label}/{lang}] skipped one sample: {e}")
            time.sleep(sleep_seconds)
    return rows

def backup_if_exists(path):
    """Copy an existing output file to a timestamped backup before overwriting it."""
    if os.path.exists(path):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = path.replace(".csv", f"_backup_{timestamp}.csv")
        shutil.copy(path, backup_path)
        print(f"Backed up existing {path} -> {backup_path}")

def main(sample_n_per_lang=30):
    human_df = pd.read_csv("data/human_samples.csv", encoding="utf-8")

    print("Generating Gemini samples (RPD=500 on flash-lite -> 4.5s sleep is safe)...")
    gemini_rows = generate_batch(human_df, generate_gemini, "gemini", sample_n_per_lang, sleep_seconds=4.5)

    print("Generating Groq/GPT-OSS samples (fast, generous free tier -> 2.1s sleep is plenty)...")
    groq_rows = generate_batch(human_df, generate_groq, "gpt_oss_groq", sample_n_per_lang, sleep_seconds=2.1)

    ai_df = pd.DataFrame(gemini_rows + groq_rows)

    output_path = "data/ai_samples.csv"
    backup_if_exists(output_path)
    ai_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(ai_df)} AI-generated samples "
          f"({len(gemini_rows)} Gemini, {len(groq_rows)} GPT-OSS/Groq) to {output_path}")

    counts = ai_df.groupby(["origin", "language"]).size()
    print("\nClass balance check:\n", counts)
    if counts.min() < 0.7 * counts.max():
        print("\nWARNING: classes are imbalanced (likely due to API errors above). "
              "Consider re-running the smaller batch before training.")

if __name__ == "__main__":
    main(sample_n_per_lang=120) 