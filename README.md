# LLM Attribution: Human vs. Gemini vs. GPT-OSS (English/German)

A portfolio project attributing news-style text to its source — human,
Google Gemini, or OpenAI GPT-OSS 120B (via Groq) — combining
classical stylometric features with multilingual embeddings, evaluated
with one-vs-rest ROC-AUC and a normalized confusion matrix. Includes a
small case study implementing the statistical watermarking technique
underlying real production systems (SynthID-Text, Anthropic's Claude
watermark, live since August 2026).

Scoped to EN/DE and 3 classes deliberately — see "Design decisions" below
for why.

## Why this project
The posting asks for "detection of AI-generated text content as well as
attribution of which LLM generated which text." A binary human-vs-AI
classifier only covers the first half. This project targets both: it's a
genuine multi-class authorship attribution problem, with LLM families
standing in for individual authors — directly relevant to PAN@CLEF's
Generative AI Detection tasks and to the posting's "optimization of
existing authorship analysis methods."

## Data
- **Human text**: [ag_news](https://huggingface.co/datasets/ag_news) (EN),
  [gnad10/10kGNAD](https://huggingface.co/datasets/gnad10) (DE)
- **AI text**: same-topic rewrites from Gemini and OpenAI GPT-OSS 120B
  (via Groq) — see `generate_ai_text.py`

No raw dataset files are committed — only code to reproduce the sample and
derived outputs.

## Design decisions
- **3 classes, not 4+**: a fourth class (e.g. Qwen/DeepSeek via
  SiliconFlow) would strengthen the "different training lineage" story,
  but adds phone-verification and a third API format for a project this
  size. Two AI lineages already demonstrates the attribution mechanism;
  a third is a natural next step, not a requirement to ship.
- **Gemini + Groq specifically**: both are free with no credit card, and
  Groq needs no phone verification — this removes the single riskiest
  dependency from the build.

## Method
1. Load human news samples (`load_human_text.py`).
2. Generate same-topic rewrites from two LLM providers (`generate_ai_text.py`).
3. Extract stylometric features (`stylometric_features.py`).
4. Concatenate stylometric + embedding features, train a multinomial
   Logistic Regression, evaluate with one-vs-rest ROC-AUC and a normalized
   confusion matrix (`train_classifier.py`).
5. Case study: simplified statistical watermarking (`watermark_demo.py`) —
   built only after the classifier is checkpointed and working.
6. Streamlit demo (`app.py`).

## Honest note on separability
Human-vs-AI is a comparatively easy separation — humans show more variance
in planning, revision, and lexical choice. Gemini-vs-GPT-OSS is a narrower
gap: both are RLHF-tuned on broadly similar web-scale English/German
prose. Expect embeddings to carry most of the separating signal for that
specific pair, with stylometric features contributing more to the
human-vs-either-AI distinction. This is a real, defensible finding, not a
project shortcoming — it says something true about how similar modern
RLHF'd models' surface style has become, which is itself a relevant
observation for authorship-attribution work.

## On watermarking, honestly
The watermark component is a **toy demonstration**, not a reproduction of
any production system. Real implementations are more sophisticated and
more robust to editing. The point is hands-on understanding of the
mechanism and its documented limitations — short passages carry too
little signal, heavy editing can break the watermark, and its absence
never proves human authorship.

## Known limitations
- Small sample size (~20-60/class/language) — a showcase, not production.
- Only 2 AI lineages; a 3rd (e.g. Qwen/DeepSeek) would strengthen the
  cross-lineage story further.
- Watermark case study is English-only, small local model (GPT-2/DistilGPT-2).
- Scoped to EN/DE only.

## Setup
```bash
pip install -r requirements.txt
python load_human_text.py
export GEMINI_API_KEY=your_key_here   # aistudio.google.com, free, no card
export GROQ_API_KEY=your_key_here     # console.groq.com, free, no card
python generate_ai_text.py
python stylometric_features.py
python train_classifier.py
python watermark_demo.py    # optional — only after core classifier works
streamlit run app.py
```
