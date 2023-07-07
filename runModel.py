from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import torch
from transformers import TFDistilBertModel, DistilBertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier

app = Flask("__name__")

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("DistilBERTModel")
model = TFDistilBertModel.from_pretrained("DistilBERTModel", from_pt=True)

# Load the logistic regression model
lr_clf = LogisticRegression()
lr_clf = lr_clf.load("logistic_regression_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer = vectorizer.load("tfidf_vectorizer.pkl")

@app.route("/")
def loadPage():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():
    input_data = request.form['input_text']
    
    # Preprocess the input data using TF-IDF vectorization
    vectorized_data = vectorizer.transform([input_data])

    # Convert the input data to BERT-based features
    tokenized_data = pd.Series([input_data]).apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    padded_data = np.array([i + [0]*(max_len - len(i)) for i in tokenized_data])
    attention_mask_data = np.where(padded_data != 0, 1, 0)
    input_ids_data = torch.tensor(padded_data)
    attention_mask_data = torch.tensor(attention_mask_data)

    with torch.no_grad():
        last_hidden_states_data = model(input_ids_data, attention_mask=attention_mask_data)

    features_data = last_hidden_states_data[0][:, 0, :].numpy()

    # Predict using the Logistic Regression model
    prediction = lr_clf.predict(features_data)[0]
    
    if prediction == 1:
        output = "Positive"
    else:
        output = "Negative"

    return render_template('home.html', input_data=input_data, output=output)

if __name__ == "__main__":
    app.run(debug=True)
