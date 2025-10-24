# Fake News Detection using Machine Learning

This project detects fake news using machine learning models based on the methodology described in the 2023 ICCCI conference paper.

It implements and compares Support Vector Machine (SVM), Random Forest, Naive Bayes, XGBoost, and a final ensemble model.

## Setup

1.  **Create Environment:**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install Dependencies:**
    Install all required libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK Data:**
    The preprocessing script needs the 'stopwords' package from NLTK. Run this command once in your terminal:
    ```bash
    python -m nltk.downloader stopwords
    ```

4.  **Get the Dataset:**
    You must provide the dataset. See the instructions in the `data/README.md` file.

## How to Run

### 1. Train the Models

This script will load the data, clean it, train all five models, print their performance (Accuracy, Precision, Recall, F1-Score), and save the best models (Ensemble, XGBoost, and the TF-IDF Vectorizer) to the `saved_models/` folder.

```bash
python train.py