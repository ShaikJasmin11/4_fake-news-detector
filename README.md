#  Fake News Detector

> RISE Internship Project 4 – Tamizhan Skills  
> Built with Scikit-Learn, TF-IDF, PassiveAggressiveClassifier, and Streamlit

A machine learning-based web app that classifies news articles as either **Real** or **Fake** based on their textual content. This is the fourth project from the **Machine Learning & AI** track of the RISE Internship by Tamizhan Skills.

---

##  Project Objective

To build a fake news detection model that:
  - Loads and combines real and fake news data from CSV files
  - Cleans and preprocesses the text using regex and vectorization
  - Trains a **PassiveAggressiveClassifier** using TF-IDF features
  - Provides an interactive Streamlit interface for real-time prediction

---

##  Tech Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-learn (TF-IDF + PassiveAggressiveClassifier)**
- **Joblib** (for model persistence)
- **Streamlit** (for frontend UI)

---

##  Project Structure

```bash
fake-news-detector/
├── app.py                     # Streamlit frontend for fake news prediction
├── main.py                    # Model training and evaluation script
├── requirements.txt           # All required packages
├── data/
│   ├── fake.csv               # Fake news articles
│   └── true.csv               # Real news articles
├── models/
│   ├── news_model.pkl         # Trained PassiveAggressiveClassifier model
│   └── tfidf_vectorizer.pkl   # TF-IDF vectorizer
├── src/
│   └── preprocess.py          # Text cleaning and preprocessing logic
└── README.md                  # You're reading it 😉
```

---

## Dataset

- Source: Fake and Real News Dataset – Kaggle(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- fake.csv: News articles labeled as Fake
- true.csv: News articles labeled as Real
- Merged and labeled for binary classification (0 = Fake, 1 = Real)

---

## How to Run

- Step 1: Install Dependencies
  
```bash
  pip install -r requirements.txt
```

- Step 2: Train the Model
  
```bash
  python main.py
```

- Step 3: Launch the Web App
  
```bash
  streamlit run app.py
```

  ---

## Model Performance

✅ Accuracy: ~93% on validation split
✅ Fast and efficient PassiveAggressiveClassifier
✅ TF-IDF vectorization for feature extraction
✅ Minimal false positives on curated news samples

---

## Highlights

- Cleans noisy text with custom regex logic
- Uses shuffled merged dataset from real & fake news
- Highly interpretable and efficient binary classifier
- Real-time predictions through a clean Streamlit UI
- Modular, scalable codebase for future NLP upgrades

---

## Acknowledgements

Thanks to Tamizhan Skills for the RISE Internship opportunity.

Inspired by real-world social challenges in misinformation and media credibility.

Built by @ShaikJasmin11
