"""
watermark_demo.py
A small, self-contained demonstration of statistical text watermarking —
the same family of technique (green-list/red-list token biasing) that
underlies SynthID-Text and Anthropic's production Claude watermark
(announced Aug 2026, ahead of the EU AI Act transparency requirements).

This is intentionally a TOY implementation on a small local model
(GPT-2), not a reproduction of any production system — the point is to
demonstrate you understand the underlying statistical mechanism, which is
exactly the "Text Watermarking" task named in PAN 2026 and adjacent to
"detection of AI-generated text" in the job posting.

Mechanism (Kirchenbauer et al., 2023, simplified):
1. At each generation step, use the previous token to seed a pseudo-random
   split of the vocabulary into a "green list" (~50%) and "red list" (~50%).
2. Bias the model's logits to favor green-list tokens during sampling.
3. To detect: count what fraction of tokens in a given text fall in their
   respective green lists. Human/unwatermarked text should land close to
   50% (chance). Watermarked text will show a statistically significant
   excess — measurable via a z-test.

Known limitations (be ready to say this in an interview):
- Real production watermarks (SynthID-Text, Anthropic's implementation)
  are more sophisticated — they preserve output quality better and are
  robust to more editing than this toy version.
- This only works well on longer texts; short snippets (a few tokens)
  don't carry enough signal to detect reliably — the same caveat Anthropic
  states publicly about its own watermark.
- Heavy paraphrasing/rewriting breaks this method, same as the real thing.
"""

import hashlib
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

GAMMA = 0.5   # fraction of vocab in the green list
DELTA = 2.0   # logit bias added to green-list tokens

def get_green_list(prev_token_id, vocab_size, gamma=GAMMA, seed_offset=0):
    seed = int(hashlib.sha256(str(prev_token_id + seed_offset).encode()).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(seed)
    n_green = int(vocab_size * gamma)
    green_ids = rng.choice(vocab_size, size=n_green, replace=False)
    return set(green_ids.tolist())

def generate_watermarked(prompt, model, tokenizer, max_new_tokens=60):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    vocab_size = model.config.vocab_size

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :]
        prev_token_id = input_ids[0, -1].item()
        green_list = get_green_list(prev_token_id, vocab_size)
        bias = torch.zeros_like(logits)
        for tid in green_list:
            bias[tid] = DELTA
        biased_logits = logits + bias
        next_token = torch.argmax(biased_logits).unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def detect_watermark(text, tokenizer, vocab_size):
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 2:
        return None

    green_hits = 0
    total = 0
    for i in range(1, len(token_ids)):
        green_list = get_green_list(token_ids[i - 1], vocab_size)
        if token_ids[i] in green_list:
            green_hits += 1
        total += 1

    observed_ratio = green_hits / total
    expected_ratio = GAMMA
    z_score = (observed_ratio - expected_ratio) / np.sqrt(expected_ratio * (1 - expected_ratio) / total)

    return {
        "green_ratio": observed_ratio,
        "z_score": z_score,
        "likely_watermarked": z_score > 4,  # common threshold in the literature
        "n_tokens": total,
    }

if __name__ == "__main__":
    print("Loading GPT-2 (small, local, English-only toy demo)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    prompt = "The latest research on multilingual news analysis shows that"
    watermarked_text = generate_watermarked(prompt, model, tokenizer)
    print("\nGenerated (watermarked):\n", watermarked_text)

    result = detect_watermark(watermarked_text, tokenizer, model.config.vocab_size)
    print("\nDetection result on watermarked text:", result)

    human_text = "The weather today is cloudy with a chance of rain in the afternoon."
    result_human = detect_watermark(human_text, tokenizer, model.config.vocab_size)
    print("\nDetection result on plain human-written text:", result_human)
