# Student Support and Course Feedback Assistant (Project 01)

Educational prototype for a teacher reviewing a class after the second assessment period.
Five Streamlit tabs: Class overview and support prediction, Learning profiles, Feedback digit,
Course comments, Teaching summary. Works fully offline; Groq Free Plan summary is optional.

**Live demo:** <add your Streamlit Community Cloud link here>

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Optional AI summary: create `.streamlit/secrets.toml` with `GROQ_API_KEY = "..."` and
`GROQ_MODEL = "openai/gpt-oss-20b"`. This file is ignored by git. Never commit keys.

## Deploy on Streamlit Community Cloud
In Advanced settings choose Python 3.12. Add `GROQ_API_KEY` under Secrets only if you want the AI summary.

## Measured test results
| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| Decision tree (max_depth=4) | 0.917 | 0.907 | class-1 precision 0.857, recall 0.900 |
| Baseline (most frequent) | 0.667 | 0.400 | never finds class 1 |
| Small CNN (5 epochs) | 0.894 | 0.894 | MNIST subset, 500 test images; uploads use MNIST-style preprocessing |
| VADER sentiment | 0.967 | 0.967 | 30 supplied comments |

KMeans: 3 learning-profile groups on G1, G2, studytime, failures (see models/cluster_profiles.csv).

## Limits
- `needs_support` (G3 below 10) is a classroom modelling target, not a promotion, admission or punishment decision. A teacher must review every suggestion.
- Group numbers are arbitrary and are not ability labels.
- The digit model is trained on MNIST. Uploaded images are auto-inverted, cropped and centred to look like MNIST, but it is still not reliable on real student handwriting and must not assign marks. The teacher confirms every digit.
- VADER and keyword topics struggle with mixed opinions, sarcasm and negation; English only.

## Data and attribution
UCI Student Performance (Cortez, CC BY 4.0), MNIST via Keras, VADER lexicon via NLTK.
See `source_notices/`. Comments are authored classroom examples, not real survey responses.
