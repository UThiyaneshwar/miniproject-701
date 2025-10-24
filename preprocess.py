import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
# You can also run: python -m nltk.downloader stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans raw text data.
    1. Converts to lowercase
    2. Removes punctuation and numbers
    3. Removes English stop words
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove punctuation and numbers
    # Keep only alphabetic characters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Remove stop words
    # Tokenize (split into words) and remove stop words
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    # Re-join words into a single string
    return " ".join(cleaned_words)