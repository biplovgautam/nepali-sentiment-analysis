import pickle
from preprocess import clean_text

def load_model():
    with open("models/model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

def cli():
    print("📈 Financial Sentiment Analyzer (Naive Bayes)")
    print("Type 'exit' to quit.\n")

    model, vectorizer = load_model()

    while True:
        sentence = input("Enter a financial sentence: ")
        if sentence.lower() == 'exit':
            print("Goodbye!")
            break
        clean = clean_text(sentence)
        vec = vectorizer.transform([clean])
        prediction = model.predict(vec)[0]
        print(f"Predicted Sentiment: {prediction}\n")

if __name__ == "__main__":
    cli()
