import pandas as pd

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Convert text to numbers
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Test custom message
msg = ["Congratulations! You won a free prize"]
msg_vec = vectorizer.transform(msg)
print("Spam" if model.predict(msg_vec)[0] == 1 else "Not Spam")
