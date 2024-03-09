import pandas as pd
import spacy

# Step 1: Load spaCy model for sentiment analysis
nlp = spacy.load("en_core_web_sm")

# Step 2: Preprocess text data
# Step 2.1: Select 'review.text' column
data = pd.read_csv("amazon_product_reviews.csv")
reviews_data = data['review.text']

# Step 2.2: Remove missing values
clean_data = data.dropna(subset=['review.text'])

# Step 2.3: Define function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase and strip leading/trailing whitespaces
    text = text.lower().strip()
    # Tokenise the text
    doc = nlp(text)
    # Remove stopwords, punctuation, and lemmatise tokens
    clean_text = " ".join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)
    return clean_text

# Apply text preprocessing to the 'review.text' column
clean_data['clean_text'] = clean_data['review.text'].apply(preprocess_text)

# Step 3: Define function for sentiment analysis
def predict_sentiment(review):
    # Analyse sentiment using spaCy
    doc = nlp(review)
    polarity = doc._.polarity
    # Determine sentiment based on polarity score
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Step 4: Test the model on sample reviews
sample_reviews = [
    "This product is amazing! I love it.",
    "I am very disappointed with this purchase."
]

for review in sample_reviews:
    sentiment = predict_sentiment(review)
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")
