import time
import os
import re
import spacy
import pandas as pd
import wordninja
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

start_time = time.time()

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat", "entity_ruler", "entity_linker", "sentencizer"])

# Slang dictionary
slang_dict = { 
    "lol": "laughing out loud", "lmao": "laughing my ass off", "rofl": "rolling on the floor laughing",
    "omg": "oh my god", "wtf": "what the heck", "brb": "be right back", "gtg": "got to go", 
    "idk": "I don't know", "smh": "shaking my head", "btw": "by the way", "fyi": "for your information",
    "afk": "away from keyboard", "imo": "in my opinion", "imho": "in my humble opinion", 
    "tbh": "to be honest", "ikr": "I know, right?", "gg": "good game", "wp": "well played", 
    "ez": "easy", "sus": "suspicious", "yeet": "to throw something with force / an exclamation of excitement",
    "noob": "newbie / inexperienced person", "rekt": "wrecked / badly defeated", "based": "acting independently and confidently",
    "cap": "lie / false statement", "simp": "someone who does too much for someone they like",
    "stan": "overly obsessed fan", "vibe": "a feeling or atmosphere", "clapback": "a witty or critical response",
    "mood": "relatable feeling or situation"
}
# Pre-compile regex patterns 
HTML_PATTERN = re.compile(r'<[^>]+>')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
PUNCT_PATTERN = re.compile(r'[^\w\s]')

def preprocess_text(text):
    """ preprocessing function """
    # Convert to lowercase 
    text = text.lower()
    
    # Apply regex substitutions in one pass
    text = HTML_PATTERN.sub('', text)
    text = URL_PATTERN.sub('', text)
    text = PUNCT_PATTERN.sub('', text)
    
    # Process words 
    words = text.split()
    processed_words = []
    for word in words:
        # Handle slang first
        word = slang_dict.get(word, word)
        # Only split if word is long enough
        if len(word) > 10:
            processed_words.extend(wordninja.split(word))
        else:
            processed_words.append(word)
    
    return ' '.join(processed_words)

# Combined tokenizer and lemmatizer
def spacy_tokenize_lemmatize(texts, batch_size=256, n_threads=None, remove_stopwords=True, remove_short_tokens=True):
    """
    Tokenizes and lemmatizes a list of texts using spaCy with parallel processing.
    """
    if n_threads is None:
        n_threads = os.cpu_count()

    stop_words = nlp.Defaults.stop_words  

    docs = list(nlp.pipe(texts, batch_size=batch_size, n_process=n_threads))

    return [
        [token.lemma_ 
         for token in doc 
         if (not remove_stopwords or token.text not in stop_words) 
         and (not remove_short_tokens or len(token.lemma_) > 2)]
        for doc in docs
    ]

    return processed_texts

def load_datasets(data_dir):
    """Load train dataset from the specified directory."""
    train_path = os.path.join(data_dir, "train.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError("Train CSV file not found in the specified directory.")
    
    train_df = pd.read_csv(train_path)
    
    print("Datasets loaded successfully.")
    return train_df


# TF-IDF Vectorization


tfidf_vectorizer = TfidfVectorizer(
    tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False, 
    max_features=20000, ngram_range=(1,3), min_df=3, max_df=0.90
)

# Model training
def train_logistic_regression(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(C=1, penalty='l2', solver='liblinear', max_iter=200)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred))
    return model  


if __name__ == "__main__":
    data_directory = "data/raw"
    train_df = load_datasets(data_directory)

    # Apply preprocessing, tokenization, and lemmatization
    print("Processing started .")
    train_df["review_prep"] = train_df["review"].apply(preprocess_text)
    train_df["lemmatized"] = spacy_tokenize_lemmatize(train_df["review_prep"].tolist())
    print("Processing completed successfully.")

    print("TF-IDF Vectorization started .")
    train_df['lemmatized'] = train_df['lemmatized'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_df['lemmatized'])
    print("TF-IDF Vectorization completed successfully.")    

    print("Training started .")
    model = train_logistic_regression(tfidf_matrix_train, train_df['sentiment'])
    print("Training completed successfully .")
    
    output_dir = os.path.join("outputs", "models")
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Save the trained model
    model_path = os.path.join(output_dir, "logistic_regression_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    # Save the TF-IDF vectorizer
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    print(f"Vectorizer saved at: {vectorizer_path}")


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.2f} seconds")