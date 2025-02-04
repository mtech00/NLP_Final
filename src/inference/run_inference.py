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
# Optimize spaCy pipeline by only keeping necessary components
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
NUMBER_PATTERN = re.compile(r'\b\d+\b')
PUNCT_PATTERN = re.compile(r'[^\w\s]')

def preprocess_text(text):
    """ preprocessing function"""
    text = text.lower()  # Convert to lowercase first
    text = HTML_PATTERN.sub('', text)
    text = URL_PATTERN.sub('', text)
    text = PUNCT_PATTERN.sub('', text)
    
    
    words = text.split()
    processed_words = []
    for word in words:
        word = slang_dict.get(word, word)
        if len(word) > 10:  # Only split long words
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

    processed_texts = [
        [token.lemma_ for token in doc if (not remove_stopwords or token.text not in stop_words) and (not remove_short_tokens or len(token.lemma_) > 2)]
        for doc in docs
    ]

    return processed_texts

def load_datasets(data_dir):
    """Load test dataset from the specified directory."""
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError("Test CSV file not found in the specified directory.")
    
    test_df = pd.read_csv(test_path)
    
    print("Datasets loaded successfully.")
    return test_df

data_directory = "data/raw"

model = joblib.load("outputs/models/logistic_regression_model.pkl")
tfidf_vectorizer = joblib.load("outputs/models/tfidf_vectorizer.pkl")

output_dir = "outputs/predictions"
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    test_df = load_datasets(data_directory)

    true_sentiments = test_df["sentiment"]  # Save the original sentiment labels
    test_df = test_df.drop(columns=["sentiment"]) 

    # Apply preprocessing, tokenization, and lemmatization
    print("Processing started .")
    test_df["review_prep"] = test_df["review"].apply(preprocess_text)
    test_df["lemmatized"] = spacy_tokenize_lemmatize(test_df["review_prep"].tolist())
    print("Processing completed successfully.")

    print("Vectorizing test data.")
    test_df['lemmatized'] = test_df['lemmatized'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
    tfidf_matrix_test = tfidf_vectorizer.transform(test_df["lemmatized"])
    print("Vectorization completed successfully.")

    test_df["predicted_sentiment"] = model.predict(tfidf_matrix_test)
    
    test_df["true_sentiment"] = true_sentiments
    accuracy = accuracy_score(test_df["true_sentiment"], test_df["predicted_sentiment"])
    classification_rep = classification_report(test_df["true_sentiment"], test_df["predicted_sentiment"])
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_rep)
    
    predictions_path = os.path.join(output_dir, "predictions.csv")  
    test_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved at: {predictions_path}")
    
    # Saves metrics to a text file
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write("Classification Report:\n")
        f.write(classification_rep)
    print(f"Metrics saved at: {metrics_path}")
        
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.2f} seconds")