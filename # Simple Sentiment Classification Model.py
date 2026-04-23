# Simple Sentiment Classification Model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -----------------------
# 1. Dataset
# -----------------------
texts = [
    "I love this movie, it was amazing",
    "This is the best thing ever",
    "I really enjoyed this",
    "Absolutely fantastic experience",
    "I hate this so much",
    "This was terrible and boring",
    "Worst thing I have watched",
    "I did not like this at all",
    "It was okay, not great",
    "Pretty good overall"
]

labels = [1,1,1,1,0,0]

# -----------------------
# 2. Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)

# -----------------------
# 3. Model Pipeline
# -----------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

# -----------------------
# 4. Training
# -----------------------
model.fit(X_train, y_train)

# -----------------------
# 5. Evaluation
# -----------------------
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("Test Accuracy:", acc)

# -----------------------
# 6. Try a prediction
# -----------------------
sample = ["I really love how good this was"]
print("Prediction:", model.predict(sample))