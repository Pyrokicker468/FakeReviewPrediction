from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
import numpy as np

app = Flask(__name__)
print("Hello")
# Load the saved DistilBERT model

#model_path = r"C:\Users\DanielC\Source\Repos\Pyrokicker468\FakeReviewPrediction\savedBertModel"
#model = tf.saved_model.load(model_path)

# Load the tokenizer
#tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Define a route for handling fetch requests
@app.route("/predict", methods=["POST"])
def predict():
    print("Hello")
    # Get the input text from the fetch request

    #data = request.get_json()

    # Preprocess the input data using the tokenizer

    #tokenized_data = tokenizer.encode(data["text"], add_special_tokens=True)
    #padded_data = np.array([tokenized_data + [0] * (512 - len(tokenized_data))])
    #attention_mask_data = np.where(padded_data != 0, 1, 0)
    #input_ids_data = torch.tensor(padded_data)
    #attention_mask_data = torch.tensor(attention_mask_data)

    # Pass the input through the loaded model

    #outputs = model(input_ids_data, attention_mask=attention_mask_data)

    # Get the predicted class

    #prediction = outputs[0].numpy().argmax()

    # Return the predicted class as a JSON response
    return jsonify({"prediction: 0"})
    # return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run()