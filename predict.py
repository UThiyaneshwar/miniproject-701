import joblib
import argparse
import os
from preprocess import clean_text

# --- Configuration ---
# This path MUST match the one in train.py
MODEL_PATH = 'saved_models/' 
VECTORIZER_NAME = 'tfidf_vectorizer.pkl'
ENSEMBLE_MODEL_NAME = 'ensemble_model.pkl'
XGB_MODEL_NAME = 'xgb_model.pkl'

def load_artifacts(model_choice):
    """Loads the specified model and the vectorizer from disk."""
    print(f"Loading model '{model_choice}'...")
    
    # Determine which model file to load
    if model_choice == 'xgb':
        model_file = XGB_MODEL_NAME
    else:
        model_file = ENSEMBLE_MODEL_NAME

    vectorizer_path = os.path.join(MODEL_PATH, VECTORIZER_NAME)
    model_path = os.path.join(MODEL_PATH, model_file)

    # Check if files exist
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        print(f"Error: Model files not found in '{MODEL_PATH}'.")
        print("Please check that you have run 'python train.py' successfully.")
        return None, None

    # Load the files
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        print("Model and vectorizer loaded successfully.")
        return vectorizer, model
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None

def predict_text(vectorizer, model, text):
    """Cleans, vectorizes, and predicts a single string of text."""
    
    # 1. Clean the input text
    cleaned_text = clean_text(text)
    
    # 2. Vectorize the cleaned text
    # Note: vectorizer.transform() expects an iterable (like a list)
    text_vector = vectorizer.transform([cleaned_text])
    
    # 3. Predict
    prediction = model.predict(text_vector)[0]
    
    # 4. Get probabilities (confidence score)
    # model.predict_proba returns probabilities for [Class 0, Class 1]
    probabilities = model.predict_proba(text_vector)[0]
    
    # Get the confidence for the predicted class
    confidence = probabilities[prediction]
    
    # 5. Format the result
    # Remember: 0 = Real, 1 = Fake
    label = "Fake News" if prediction == 1 else "Real News"
    
    return label, confidence

if __name__ == "__main__":
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Predict if a news headline is Real or Fake.")
    
    parser.add_argument(
        "text", 
        type=str, 
        help="The news headline or text to classify (e.g., 'Your headline here')."
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=['ensemble', 'xgb'],
        default='ensemble',
        help="Which model to use for prediction: 'ensemble' (default) or 'xgb'."
    )

    args = parser.parse_args()

    # --- Main Prediction Logic ---
    vectorizer, model = load_artifacts(args.model)
    
    if vectorizer and model:
        label, confidence = predict_text(vectorizer, model, args.text)
        
        print("\n" + "="*20)
        print(f"Input Text: \"{args.text}\"")
        print(f"> Prediction: {label} (Confidence: {confidence:.2%})")
        print("="*20)