import random
import pickle

# Load model components
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("responses.pkl", "rb") as f:
    responses = pickle.load(f)

def get_response(user_input):
    vector = vectorizer.transform([user_input])
    prediction = model.predict(vector)
    tag = label_encoder.inverse_transform(prediction)[0]
    return random.choice(responses[tag])

# CLI interface
if __name__ == "__main__":
    print("🤖 Chatbot is ready! Type 'quit' to exit.")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            print("Bot: Bye! 👋")
            break
        reply = get_response(msg)
        print(f"Bot: {reply}")
