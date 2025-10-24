import pandas as pd
import joblib
import os
from preprocess import clean_text

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Import all models from the paper
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

# Import evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# --- Configuration ---
DATA_PATH = 'data/'
MODEL_PATH = 'saved_models/'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- 1. Load and Prepare Data ---
def load_data():
    """Loads and merges real and fake news, adding labels."""
    print("Loading data...")
    try:
        true_df = pd.read_csv(os.path.join(DATA_PATH, 'True.csv'))
        fake_df = pd.read_csv(os.path.join(DATA_PATH, 'False.csv'))
    except FileNotFoundError:
        print(f"Error: Dataset not found in '{DATA_PATH}'.")
        print("Please follow the instructions in 'data/README.md' to download the dataset.")
        return None

    # Add labels: 0 for Real, 1 for Fake
    true_df['label'] = 0
    fake_df['label'] = 1

    # Combine text columns (Title and Text)
    true_df['text'] = true_df['title'] + " " + true_df['text']
    fake_df['text'] = fake_df['title'] + " " + fake_df['text']

    # Keep only 'text' and 'label'
    true_df = true_df[['text', 'label']]
    fake_df = fake_df[['text', 'label']]

    # Merge and shuffle
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Handle any missing text
    df.dropna(subset=['text'], inplace=True)

    print(f"Data loaded: {len(df)} total articles.")
    return df

# --- 2. Preprocess and Vectorize ---
def vectorize_data(df):
    """Applies text cleaning and TF-IDF vectorization."""
    print("Cleaning and preprocessing text...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    X = df['cleaned_text']
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # --- Feature Extraction (as per paper) ---
    # We use TF-IDF as it performed best in the paper.
    print("Applying TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_df=0.7, ngram_range=(1,1), max_features=10000)
   
   
    
    # --- Alternative: Bag-of-Words (BoW) ---
    # To use BoW instead, uncomment the line below:
    # vectorizer = CountVectorizer(max_df=0.7, ngram_range=(1,2))
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Vectorization complete. Training features: {X_train_vec.shape}, Test features: {X_test_vec.shape}")

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

# --- 3. Define and Train Models ---
def train_models(X_train, y_train):
    """Initializes, trains, and returns all models from the paper."""
    print("Initializing models...")
    
    # 1. Random Forest (as in paper)
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    
    # 2. Support Vector Machine (as in paper)
    # Added probability=True for the ensemble's 'soft' voting
    clf_svm = SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)
    
    # 3. Naive Bayes (as in paper)
    clf_nb = MultinomialNB()
    
    # 4. XGBoost (the "extra technology" model)
    clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)

    # 5. Ensemble Model (as in paper: RF, SVM, NB)
    # Using 'soft' voting to average probabilities, which is often more robust than 'hard' (majority vote)
    clf_ensemble = VotingClassifier(
        estimators=[('rf', clf_rf), ('svm', clf_svm), ('nb', clf_nb)],
        voting='soft'
    )

    models = {
        "Random Forest": clf_rf,
        "Support Vector Machine (SVM)": clf_svm,
        "Naive Bayes": clf_nb,
        "XGBoost": clf_xgb,
        "Ensemble (RF+SVM+NB)": clf_ensemble
    }

    # Train all models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
    print("All models trained.")
    return models

# --- 4. Evaluate Models ---
def evaluate_models(models, X_test, y_test):
    """Calculates and prints performance metrics for all trained models."""
    print("\n" + "="*30)
    print(" MODEL EVALUATION RESULTS")
    print("="*30)

    # Define labels for classification report
    target_names = ['Real News (Class 0)', 'Fake News (Class 1)']

    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate metrics as per the paper
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        print("-"*30)

# --- 5. Save Models ---
def save_models(models, vectorizer):
    """Saves the best-performing models and the vectorizer."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    print("\nSaving models...")
    
    # Save the TF-IDF vectorizer (CRITICAL for prediction)
    vectorizer_path = os.path.join(MODEL_PATH, 'tfidf_vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to {vectorizer_path}")

    # Save the Ensemble model (highest accuracy in paper)
    ensemble_model_path = os.path.join(MODEL_PATH, 'ensemble_model.pkl')
    joblib.dump(models['Ensemble (RF+SVM+NB)'], ensemble_model_path)
    print(f"Ensemble model saved to {ensemble_model_path}")

    # Save the XGBoost model (best individual model)
    xgb_model_path = os.path.join(MODEL_PATH, 'xgb_model.pkl')
    joblib.dump(models['XGBoost'], xgb_model_path)
    print(f"XGBoost model saved to {xgb_model_path}")

# --- Main Execution ---
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        X_train_vec, X_test_vec, y_train, y_test, vectorizer = vectorize_data(df)
        models = train_models(X_train_vec, y_train)
        evaluate_models(models, X_test_vec, y_test)
        save_models(models, vectorizer)
        print("\nTraining and evaluation complete.")