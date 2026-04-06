# Veridect — Fake News Detector

ML-powered fake news detection system using **Naive Bayes + TF-IDF**.  
Runs entirely offline (no API keys). Fetches live news from Google News RSS.

---

## Stack

| Component     | Technology                        |
|---------------|-----------------------------------|
| UI            | Streamlit                         |
| Classifier    | Naive Bayes (hand-crafted TF-IDF) |
| News feed     | Google News RSS via feedparser    |
| Language      | Python 3.10+                      |

---

## Setup

```bash
# 1. Clone / unzip the project
cd fake_news_detector

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Project structure

```
fake_news_detector/
├── app.py           # Streamlit UI
├── model.py         # Naive Bayes + TF-IDF classifier
├── news_fetcher.py  # Google News RSS fetcher
├── requirements.txt
└── README.md
```

---

## How the ML model works

### 1. Vocabulary (TF-IDF weights)
Two hand-curated dictionaries act as learned vocabulary weights:

- `FAKE_VOCAB` — ~60 misinformation-correlated terms (e.g. "bombshell", "crisis actor", "deep state"), each with a log-likelihood weight.
- `REAL_VOCAB` — ~50 credibility signals (e.g. "according to", "peer reviewed", "clinical trial").

### 2. Naive Bayes posterior
```
P(real | text) = real_score / (real_score + fake_score)
```

### 3. Linguistic features (augment the score)
| Feature            | Effect on score               |
|--------------------|-------------------------------|
| CAPS word ratio    | High → fake score ↑           |
| Exclamation marks  | Each ! → fake score +0.45     |
| Sensational terms  | Each hit → fake score +1.3    |
| Hedge words        | Each hit → real score +0.5    |
| Long sentences     | avg > 18 words → real +1.2    |
| Long words         | avg > 5.5 chars → real +1.8   |

### 4. Verdict thresholds
| Real probability | Verdict       |
|-----------------|---------------|
| ≥ 65%           | Likely Real   |
| 36–64%          | Uncertain     |
| ≤ 35%           | Likely Fake   |

---

## Extending the model

To upgrade to a proper trained ML model:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('clf',   MultinomialNB(alpha=0.1)),
])
pipeline.fit(X_train, y_train)
```

Replace `model.py`'s `predict()` with `pipeline.predict_proba()`.

Recommended datasets:
- [LIAR dataset](https://huggingface.co/datasets/liar)
- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)
- [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
