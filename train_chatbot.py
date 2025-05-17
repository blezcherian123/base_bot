import json
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Load intents
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X, y = [], []
responses = {}

# Build training data
for intent in data["intents"]:
    tags = intent["tag"]
    if isinstance(tags, str):
        tags = [tags]  # ensure it's a list

    for tag in tags:
        for pattern in intent["patterns"]:
            X.append(pattern)
            y.append(tag)

        # Store one response list per tag (same response for both "greet" and "welcome")
        responses[tag] = intent["responses"]

# Vectorize patterns
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train model
model = LinearSVC()
model.fit(X_vectors, y_encoded)

# Save components
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("responses.pkl", "wb") as f:
    pickle.dump(responses, f)

print("✅ Model trained and saved successfully.")
