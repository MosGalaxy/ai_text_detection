# LLM Attribution: Human vs. Gemini vs. GPT-OSS (English/German)

A portfolio project attributing news-style text to its source — human,
Google Gemini, or OpenAI GPT-OSS 120B (via Groq) — combining classical
stylometric features with multilingual embeddings, evaluated with
one-vs-rest ROC-AUC and a normalized confusion matrix.

Scoped to EN/DE and 3 classes deliberately — see "Design decisions" below
for why.

## Why this project
The posting asks for "detection of AI-generated text content as well as
attribution of which LLM generated which text." A binary human-vs-AI
classifier only covers the first half. This project targets both: it's a
genuine multi-class authorship attribution problem, with LLM families
standing in for individual authors.

## Data
- **Human text**: [ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) (EN),
  [gnad10/10kGNAD](https://huggingface.co/datasets/community-datasets/gnad10) (DE)
- **AI text**: same-topic rewrites from `gemini-3.5-flash-lite` and
  `openai/gpt-oss-120b` (via Groq)

No raw dataset files are committed — only code to reproduce the sample.

## Results
On a class-balanced test set (13 samples per class):

| Class | ROC-AUC (one-vs-rest) |
|---|---|
| Human | 0.731 |
| GPT-OSS | 0.536 |
| Gemini | 0.497 |

**Human-vs-AI separation is real.** The classifier reliably distinguishes
human-written text from AI-generated text — this is the easier, more
robust part of the problem.

**Gemini-vs-GPT-OSS attribution is close to chance** (AUC ≈ 0.5). The
confusion matrix shows Gemini samples are misclassified as GPT-OSS more
often (54%) than classified correctly (31%). This is consistent with the
hypothesis stated below: two modern RLHF-tuned models rewriting the same
short news snippet converge stylistically enough that classical
stylometric features and general-purpose multilingual embeddings can't
reliably tell them apart at this sample size.

This is a genuine, honest finding — not a bug to hide. It says something
real about how similar current-generation RLHF'd models' surface style
has become, and it's exactly the kind of result "optimization of existing
authorship analysis methods" points toward: knowing *when* a method
doesn't separate classes well is as informative as when it does.

## Honest note on sample size
Test set support is only 13 samples per class. This is large enough to
see the human-vs-AI signal clearly, but too small to confidently say
whether AI-vs-AI attribution is a genuinely hard problem or partly an
artifact of a small test set. Scaling up sample size is the natural next
step (see Known limitations).

## Design decisions
- **3 classes, not 4+**: two AI lineages already demonstrates the
  attribution mechanism.
- **Gemini + Groq specifically**: both free, no credit card; Groq needs
  no phone verification.
- **Class balancing before training**: the human dataset naturally has
  2x the samples of each AI class (more human articles are downloaded
  than get rewritten). Training on the imbalanced data caused the model
  to systematically over-predict "human" — downsampling every
  (origin, language) group to match the smallest group fixed this. This
  is itself a useful thing to be able to explain: it's a real example of
  diagnosing *why* a model underperforms rather than just reporting a
  number.

## Method
1. Load human news samples (`load_human_text.py`).
2. Generate same-topic rewrites from two LLM providers (`generate_ai_text.py`).
3. Extract stylometric features (`stylometric_features.py`).
4. Balance classes, concatenate stylometric + embedding features, train a
   Logistic Regression, evaluate with one-vs-rest ROC-AUC and a
   normalized confusion matrix (`train_classifier.py`).
5. Case study (not yet implemented — see Known limitations): simplified
   statistical watermarking (`watermark_demo.py`).
6. Streamlit demo (`app.py`).

## Development notes
Building this surfaced several real-world issues worth documenting:
- **Upstream dataset renaming**: `ag_news` and `gnad10` now require full
  namespaces (`fancyzhx/ag_news`, `community-datasets/gnad10`) after a
  HuggingFace Hub repo reorganization.
- **Model deprecations mid-project**: `llama-3.3-70b-versatile` was
  deprecated by Groq; `gemini-3.6-flash`'s free tier turned out to be
  only 20 requests/day, too low for this project's needs — switched to
  `gemini-3.5-flash-lite` (500 requests/day, though only 15/minute,
  requiring a longer per-request delay).
- **Windows CSV encoding**: opening output CSVs in Excel silently
  re-saves them in a non-UTF-8 encoding; fixed by always specifying
  `encoding="utf-8"` explicitly on every read/write, and by not opening
  CSVs in Excel during development.
- **Class imbalance** (see Design decisions above).
- **pandas API change**: `groupby().apply(lambda g: ...)` started
  dropping the grouping columns from the result in a recent pandas
  version; replaced with the built-in `groupby().sample()`, which is
  both more robust and more idiomatic for this exact use case.

## Known limitations
- Small sample size (13/class in the test set) — results on Gemini vs.
  GPT-OSS attribution should be treated as preliminary.
- Only 2 AI lineages; a 3rd would strengthen the cross-lineage story.
- Watermark case study is designed but not yet implemented.
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
streamlit run app.py
```